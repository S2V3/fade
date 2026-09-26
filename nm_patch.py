"""
nm_patch.py -- run the UNCHANGED FADE pipeline on a newer model and a second dataset.

Nothing in the existing scripts is edited. This module patches three functions of
kaggle_run at import time and then runs the requested script, so every stage
(kaggle_run.py, run_ctc_validate.py, run_mpv.py, score_mpv.py, ...) sees the same
patched loader:

    python nm_patch.py kaggle_run.py        --model Qwen/Qwen3.5-4B ...
    python nm_patch.py run_ctc_validate.py  --model Qwen/Qwen3.5-4B ...
    FADE_TEST_DATASET=gsm_symbolic_p1 python nm_patch.py run_ctc_validate.py --split test ...

WHAT IS PATCHED, AND WHY
------------------------
1. kaggle_run.load_model  (only for models that are NOT Llama-2; Llama-2 goes
   through the original function byte-for-byte)
     - chat mode is FORCED on. autodetect_chat_mode() looks for 'chat',
       'instruct' or '-it' in the id, and 'Qwen/Qwen3.5-4B' has none of them, so
       the original would have sent raw completion text to a chat model.
     - thinking is OFF. Qwen3.5 thinks by default; every apply_chat_template call
       gets enable_thinking=False, so traces are ordinary chain-of-thought the
       equation extractor can read, comparable to Llama-2's.
     - dtype from FADE_DTYPE (fp16 default; T4 has no bf16). nm_smoke.py decides.
     - sampling defaults normalised to what Llama-2 ran with. generation.py sets
       temperature, top_p and repetition_penalty explicitly on every call; the one
       thing a model's own generation_config can still inject is top_k / min_p.
       Qwen ships top_k=20; Llama-2 ran at the transformers default 50. We set 50
       and clear min_p so the decoding rule is identical across models.
2. kaggle_run._load_split  -- when FADE_TEST_DATASET names a GSM-Symbolic variant,
   split='test' returns that dataset instead of GSM8K test. split='train' is
   ALWAYS GSM8K train, so seeds, pool and detector training never move.
3. kaggle_run.verify_identity_and_access -- Qwen is not gated. A missing or
   failing HF token is a warning for an ungated model, not a fatal error.

Everything written by a run records the model id, dtype and dataset in
nm_provenance.json next to the store (see record_provenance).
"""
from __future__ import annotations

import json
import os
import runpy
import sys
from pathlib import Path

REPO_DIR = Path(__file__).resolve().parent
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

import kaggle_run as K          # noqa: E402
import generation as G          # noqa: E402

_ORIG_LOAD_MODEL = K.load_model
_ORIG_LOAD_SPLIT = K._load_split
_ORIG_VERIFY = K.verify_identity_and_access
_PATCHED = False

# top_k Llama-2 effectively ran with (transformers default; its generation_config
# sets none). Recorded so the paper can state the decoding rule exactly.
TOP_K_NORMALISED = 50


def is_legacy_llama2(model_id: str) -> bool:
    return "llama-2" in (model_id or "").lower()


def _wrap_chat_template(tok):
    """Every apply_chat_template call defaults to enable_thinking=False."""
    if getattr(tok, "_nm_wrapped", False):
        return tok
    orig = tok.apply_chat_template

    def apply_chat_template(conversation, *args, **kwargs):
        kwargs.setdefault("enable_thinking", False)
        return orig(conversation, *args, **kwargs)

    tok.apply_chat_template = apply_chat_template
    tok._nm_wrapped = True
    return tok


def load_model(token):
    """Drop-in for kaggle_run.load_model. Llama-2 -> the original, untouched."""
    if is_legacy_llama2(K.MODEL_ID):
        return _ORIG_LOAD_MODEL(token)
    import time
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer
    transformers.logging.set_verbosity_error()
    mid = K.MODEL_ID
    dt_name = os.environ.get("FADE_DTYPE", "fp16").lower()
    dtype = {"fp16": torch.float16, "fp32": torch.float32,
             "bf16": torch.bfloat16}[dt_name]
    print(f"\nLoading {mid} ({dt_name}) via nm_patch "
          f"| transformers {transformers.__version__} | torch {torch.__version__}")
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(mid, token=token)
    if getattr(tok, "chat_template", None) is None:
        try:                                    # multimodal repos keep it on the processor
            from transformers import AutoProcessor
            proc = AutoProcessor.from_pretrained(mid, token=token)
            tok.chat_template = (getattr(proc, "chat_template", None)
                                 or getattr(getattr(proc, "tokenizer", None), "chat_template", None))
        except Exception as e:
            print(f"  (processor chat template unavailable: {e})")
    if getattr(tok, "chat_template", None) is None:
        raise SystemExit(f"  STOP: {mid} has no chat template -- cannot run chat mode.")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    _wrap_chat_template(tok)

    # one GPU is enough for a 4B model in fp16; fp32 needs both T4s
    n_gpu = torch.cuda.device_count()
    device_map = {"": 0} if (dt_name != "fp32" and n_gpu >= 1) else "auto"
    kw = dict(token=token, device_map=device_map)
    model = None
    for cls_name in ("AutoModelForCausalLM", "AutoModelForImageTextToText"):
        try:
            cls = getattr(transformers, cls_name)
            try:
                model = cls.from_pretrained(mid, dtype=dtype, **kw)
            except TypeError:
                model = cls.from_pretrained(mid, torch_dtype=dtype, **kw)
            print(f"  loaded as {cls_name} -> {type(model).__name__}")
            break
        except Exception as e:
            print(f"  {cls_name} failed: {str(e)[:200]}")
    if model is None:
        raise SystemExit(f"  STOP: could not load {mid}")
    model.eval()

    gc = model.generation_config
    before = {k: getattr(gc, k, None) for k in ("top_k", "min_p", "top_p",
                                                "temperature", "repetition_penalty")}
    gc.top_k = TOP_K_NORMALISED
    if hasattr(gc, "min_p"):
        gc.min_p = None
    print(f"  generation_config normalised: top_k {before['top_k']} -> {gc.top_k}, "
          f"min_p {before['min_p']} -> None  (temperature/top_p/repetition_penalty "
          f"are set per call by generation.py)")

    G.set_chat_mode(True)
    print(f"  loaded in {time.time() - t0:.0f}s on {next(model.parameters()).device} "
          f"| cuda={torch.cuda.is_available()} | CHAT_MODE=True (forced) | "
          f"enable_thinking=False")
    return model, tok


def _load_split(split, n_needed=None):
    ds = os.environ.get("FADE_TEST_DATASET", "gsm8k").strip().lower()
    if split == "test" and ds not in ("", "gsm8k"):
        import nm_data
        print(f"\nLoading {ds} as the TEST split (nm_patch; train stays GSM8K)")
        return nm_data.load_symbolic(ds, n_needed)
    return _ORIG_LOAD_SPLIT(split, n_needed)


def verify_identity_and_access(token):
    try:
        return _ORIG_VERIFY(token)
    except SystemExit:
        if is_legacy_llama2(K.MODEL_ID):
            raise
        try:
            from huggingface_hub import HfApi
            HfApi().model_info(K.MODEL_ID)
            print(f"  (HF identity check failed, but {K.MODEL_ID} is public -- continuing)")
        except Exception as e:
            print(f"  !! {K.MODEL_ID} not reachable without a token: {e}")
            raise


_ORIG_RETRY_ONCE = K.retry_once


def retry_once(*args, **kwargs):
    """[NM-1] BUG FIX for kaggle_run.py on main. run_full logs a retry with
        _detail_record(..., **retry_once.last_accept, ..., hash_appended=_happ, ...)
    and last_accept (apply_cure's meta) ALSO carries 'hash_appended', so the very
    first retry raises 'got multiple values for keyword argument hash_appended'
    and the train run dies. The two values are the same number (retry_once returns
    meta['hash_appended'] as _happ), so dropping the duplicate loses nothing.
    Fix it in kaggle_run.py itself by excluding the key in that dict comprehension."""
    out = _ORIG_RETRY_ONCE(*args, **kwargs)
    la = dict(getattr(_ORIG_RETRY_ONCE, "last_accept", None) or {})
    la.pop("hash_appended", None)
    retry_once.last_accept = la
    return out


def apply():
    global _PATCHED
    if _PATCHED:
        return
    K.load_model = load_model
    K._load_split = _load_split
    K.verify_identity_and_access = verify_identity_and_access
    K.retry_once = retry_once
    _PATCHED = True
    print(f"  [nm_patch] active | FADE_DTYPE={os.environ.get('FADE_DTYPE', 'fp16')} "
          f"| FADE_TEST_DATASET={os.environ.get('FADE_TEST_DATASET', 'gsm8k')}")


def record_provenance(store_dir, **extra):
    """nm_provenance.json: which model, dtype, dataset and library versions wrote
    this store. Merged, never overwritten, so each stage adds its own entry."""
    p = Path(store_dir) / "nm_provenance.json"
    blob = {}
    if p.exists():
        try:
            blob = json.loads(p.read_text())
        except Exception:
            blob = {}
    entry = {"model": K.MODEL_ID, "dtype": os.environ.get("FADE_DTYPE", "fp16"),
             "test_dataset": os.environ.get("FADE_TEST_DATASET", "gsm8k"),
             "top_k_normalised": TOP_K_NORMALISED,
             "chat_mode": bool(G.CHAT_MODE), "enable_thinking": False}
    try:
        import torch
        import transformers
        entry.update({"transformers": transformers.__version__, "torch": torch.__version__})
    except Exception:
        pass
    entry.update(extra)
    blob.setdefault("stages", []).append(entry)
    Path(store_dir).mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(blob, indent=2))


def _store_arg(argv):
    for flag in ("--store-dir", "--store"):
        if flag in argv:
            i = argv.index(flag)
            if i + 1 < len(argv):
                return argv[i + 1]
    return None


def main():
    if len(sys.argv) < 2:
        raise SystemExit("usage: python nm_patch.py <script.py> [args...]")
    script = sys.argv[1]
    args = sys.argv[2:]
    apply()
    path = Path(script)
    if not path.is_absolute():
        path = REPO_DIR / path
    store = _store_arg(args)
    if "--model" in args:
        K.MODEL_ID = args[args.index("--model") + 1]
    if store:
        try:
            record_provenance(store, script=path.name, argv=args)
        except Exception as e:
            print(f"  (provenance not recorded: {e})")
    sys.argv = [str(path)] + args
    if path.name == "kaggle_run.py":
        K.main()                 # the imported (patched) module, not a fresh copy
    else:
        runpy.run_path(str(path), run_name="__main__")


if __name__ == "__main__":
    main()
