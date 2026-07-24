"""
Supports local runs (no GPU) for importing STRATEGY_NAMES and
STRATEGY_EXEMPLAR_NEEDS. Torch is imported safely with a fallback.

generate() signature:
    generate(
        problem          : str,
        strategy_id      : int,       0-15
        manual_exemplars : List[Dict], pre-selected seeds for this strategy
        pool_exemplars   : List[Dict], accumulated good pool for KNN
        category_info    : Dict,       from categorizer.categorize()
        max_new_tokens   : int,        default 512
        return_logprobs  : bool,       default True
        knn_fn           : callable,   selector.get_knn_exemplars
    )
"""

import re
from collections import Counter
from typing import List, Dict, Optional, Tuple

# Safe torch import — allows local use without GPU
try:
    import torch
    _TORCH_AVAILABLE = True
except (ImportError, OSError):
    torch = None
    _TORCH_AVAILABLE = False

import loopguard  # anchor-free loop truncation + stopping


# ── Strategy metadata ─────────────────────────────────────────────────────────
# Defined at module level so they can be imported without instantiating anything

STRATEGY_NAMES: Dict[int, str] = {
    0:  'zero_shot',
    1:  'zero_shot_cot',
    2:  'few_shot',
    3:  'few_shot_dynamic',
    4:  'self_consistency',
    5:  'temp_low',
    6:  'temp_high',
    7:  'explicit_steps',
    8:  'scratchpad',
    9:  'math_first',
    10: 'category_context',
    11: 'worked_example',
    12: 'reversed_qa',
    13: 'cot_few_shot',
    14: 'cot_self_consistency',
    15: 'full_pipeline',
}

# n_per_category for each strategy
# n=0 → zero-shot (no exemplars needed)
# n>0 → select n exemplars from EACH of 8 category types (total = n×8)
STRATEGY_EXEMPLAR_NEEDS: Dict[int, int] = {
    0:  0,
    1:  0,
    2:  5,
    3:  3,
    4:  0,
    5:  0,
    6:  0,
    7:  3,
    8:  3,
    9:  3,
    10: 2,
    11: 2,
    12: 1,
    13: 5,
    14: 3,
    15: 5,
}


# ── Shared answer parser ──────────────────────────────────────────────────────

def extract_final_answer(text: str) -> Optional[str]:
    """Extract final numeric answer from generated trace."""
    m = re.search(r'####\s*([\d,]+(?:\.\d+)?)', text)
    if m:
        return m.group(1).replace(',', '').strip()
    m = re.search(
        r'(?:the\s+)?answer\s*(?:is|=|:)\s*([\d,]+(?:\.\d+)?)',
        text, re.IGNORECASE
    )
    if m:
        return m.group(1).replace(',', '').strip()
    nums = re.findall(r'[\d,]+(?:\.\d+)?', text)
    return nums[-1].replace(',', '').strip() if nums else None


def majority_vote(answers: List[Optional[str]]) -> Optional[str]:
    valid = [a for a in answers if a is not None]
    return Counter(valid).most_common(1)[0][0] if valid else None


# ── Prompt formatting helpers ─────────────────────────────────────────────────

# Universal solving protocol — included in EVERY strategy's prompt.
# This is part of the experimental conditions: identical across all arms.
_HEADER = (
    "You are an expert math tutor. Solve the problem using this exact process:\n"
    "1. Read the question carefully and restate what is being asked.\n"
    "2. Recall the method needed to solve it.\n"
    "3. Think it through, then solve step by step, writing every calculation.\n"
    "4. Do NOT round intermediate values. Keep exact fractions "
    "(e.g. 50/60 = 5/6, not 0.83) and round only the final answer if needed.\n"
    "5. Don't repeat any steps, unless required.\n"
    "6. Check the solution once; if you find an error, correct it once.\n"
    "7. Finish with the final answer on its own line as: #### <number>\n"
    "   Write it as a plain number only - no units, no currency symbol, "
    "no words, and nothing after it.\n\n"
)

# System message used when a chat/instruct model is active but the assembled
# prompt doesn't carry the standard header (e.g. a bare retry prompt).
_SYSTEM_DEFAULT = (
    "You are an expert math tutor. Solve the problem step by step, writing "
    "every calculation. Do not round intermediate values. Finish with the "
    "final answer on its own line as: #### <number>"
)

# Decoding controls — applied identically to every arm (log as one decision):
#  - REPETITION_PENALTY breaks greedy-decoding degeneration loops (base-model
#    doubling spirals / repeated sentences seen in the zero-shot sanity run)
#  - STOP_MARKERS end generation when the model starts a new "Question:" of
#    its own (few-shot scaffold gives it this termination pattern to imitate)
GENERATION_VERSION = "v4-chat"  # kaggle_run's guard checks this
# The decisive anti-loop fix: STOP the instant "#### <number>" is generated.
# (Prompt instructions can't fix looping -- a base model imitates, it does
# not obey -- and no_repeat_ngram can't catch spirals whose numbers mutate.)
ANSWER_STOP_RE = re.compile(r"####\s*\$?-?\d[\d,]*(?:\.\d+)?")
REPETITION_PENALTY = 1.15            # stronger push against repetition
NO_REPEAT_NGRAM    = 6               # HARD loop-killer: no exact 6-token
                                     # repetition can ever be generated --
                                     # this is what actually breaks the
                                     # "\begin{code} 72+72=144" cycles
                                     # (stop_strings can't: a looping model
                                     # never emits "Question:")
BAN_STRINGS = ["\\begin{code}", "\\end{code}", "```"]   # never generate these
N_SHOTS = 8                          # few-shot strategies show 8 exemplars
STOP_MARKERS = ["\nQuestion:", "\nQ:", "\nExample", "\nProblem:"]

# --------------------------------------------------------------------------- #
# CHAT / INSTRUCT MODEL SUPPORT
# A base model (Llama-2-7b-hf) continues text, so a plain completion prompt is
# correct for it. An instruct/chat model (Llama-2-7b-chat-hf, Mistral-Instruct,
# ...) was fine-tuned on a specific dialogue structure -- feeding it the same raw
# "Question:/Solution:" text puts it OFF-distribution and costs real accuracy.
# When CHAT_MODE is on we re-wrap the SAME assembled prompt through the
# tokenizer's own chat template: the protocol header becomes the system message,
# the exemplars + question become the user message. Nothing about strategy
# construction or retry changes -- this is purely the final encoding step, so it
# applies identically to every arm.
# --------------------------------------------------------------------------- #
CHAT_MODE = False


def set_chat_mode(enabled: bool) -> None:
    global CHAT_MODE
    CHAT_MODE = bool(enabled)


def autodetect_chat_mode(tokenizer, model_id: str = "") -> bool:
    """Enable chat mode when the model is an instruct/chat variant AND the
    tokenizer actually ships a chat template."""
    name = (model_id or "").lower()
    looks_chat = any(k in name for k in ("chat", "instruct", "-it"))
    has_tpl = getattr(tokenizer, "chat_template", None) is not None
    set_chat_mode(bool(looks_chat and has_tpl))
    return CHAT_MODE


def _to_chat_prompt(tokenizer, prompt: str) -> str:
    """Re-wrap an assembled completion prompt using the model's chat template.
    Falls back to the original prompt if anything goes wrong."""
    if prompt.startswith(_HEADER):
        system, user = _HEADER.strip(), prompt[len(_HEADER):].strip()
    else:
        system, user = _SYSTEM_DEFAULT, prompt.strip()
    try:
        return tokenizer.apply_chat_template(
            [{"role": "system", "content": system},
             {"role": "user", "content": user}],
            tokenize=False, add_generation_prompt=True)
    except Exception:
        return prompt


def _truncate_trace(text: str) -> str:
    """Keep only the model's FIRST attempt. Robust to loops that emit no
    '####' and no literal 'Question:' (base-model prose / repeated-step
    re-solves) via loopguard; safe on a clean single solution."""
    return loopguard.truncate_trace(text)

def _fmt_exemplars(exemplars: List[Dict], n: int) -> str:
    out = ""
    for i, ex in enumerate(exemplars[:n], 1):
        trace = ex.get('trace', ex.get('raw_answer', ''))
        out += f"Example {i}:\nQuestion: {ex['question']}\nSolution: {trace}\n\n"
    return out

def _fmt_category_hint(cat_info: Dict) -> str:
    if not cat_info:
        return ""
    return (
        f"[Problem type: {cat_info.get('category_type','?').upper()} | "
        f"Complexity: {cat_info.get('complexity','?')}]\n\n"
    )


# ── Prompt builder ────────────────────────────────────────────────────────────

def _build_prompt(
    sid              : int,
    problem          : str,
    manual_exemplars : List[Dict],
    pool_exemplars   : List[Dict],
    cat_info         : Optional[Dict],
    knn_fn           = None,
) -> Tuple[str, Dict]:
    """
    Build prompt for strategy sid.
    Returns (prompt_string, generation_kwargs_overrides).
    """
    gen_kw = {}

    def _dyn(k=3):
        """Get exemplars via KNN if available, else fall back to manual."""
        if knn_fn and pool_exemplars:
            return knn_fn(
                problem, pool_exemplars, k,
                cat_info.get('category_type') if cat_info else None
            )
        return manual_exemplars[:k]

    # GROUP 1 — PROMPT-BASED
    if sid == 0:
        prompt = f"{_HEADER}Question: {problem}\nAnswer:"

    elif sid == 1:
        prompt = f"{_HEADER}Question: {problem}\nLet's think step by step.\n"

    elif sid == 2:
        prompt = (
            f"{_HEADER}"
            f"{_fmt_exemplars(manual_exemplars, N_SHOTS)}"
            f"Question: {problem}\nSolution:"
        )

    elif sid == 3:
        exs = _dyn(k=3)
        prompt = (
            f"{_HEADER}Here are similar problems:\n\n"
            f"{_fmt_exemplars(exs, 3)}"
            f"Now solve:\nQuestion: {problem}\nSolution:"
        )

    # GROUP 2 — SAMPLING-BASED
    elif sid == 4:
        prompt = f"{_HEADER}Question: {problem}\nLet's think step by step.\n"
        gen_kw['num_return_sequences'] = 5
        gen_kw['temperature']          = 0.9

    elif sid == 5:
        prompt = f"{_HEADER}Question: {problem}\nLet's think step by step.\n"
        gen_kw['temperature'] = 0.3

    elif sid == 6:
        prompt = f"{_HEADER}Question: {problem}\nLet's think step by step.\n"
        gen_kw['temperature'] = 1.0  # capped from 1.2 — safer for LLaMA-2

    # GROUP 3 — FORMATTING-BASED
    elif sid == 7:
        ex_block = ""
        if manual_exemplars:
            ex = manual_exemplars[0]
            ex_block = (
                f"Example:\nQuestion: {ex['question']}\n"
                f"Solution: {ex.get('trace', ex.get('raw_answer',''))}\n\n"
            )
        prompt = (
            f"{_HEADER}{ex_block}"
            f"Use numbered steps.\nQuestion: {problem}\n\nStep 1:"
        )

    elif sid == 8:
        prompt = f"{_HEADER}Question: {problem}\n\nScratch:\n"

    elif sid == 9:
        prompt = f"{_HEADER}Question: {problem}\n\nKey equation(s): "

    # GROUP 4 — CONTEXT-BASED
    elif sid == 10:
        hint = _fmt_category_hint(cat_info)
        prompt = (
            f"{_HEADER}{hint}"
            f"Question: {problem}\nLet's think step by step.\n"
        )

    elif sid == 11:
        worked = ""
        if manual_exemplars:
            best = manual_exemplars[0]
            worked = (
                f"Worked example:\nQuestion: {best['question']}\n"
                f"Full solution:\n{best.get('trace', best.get('raw_answer',''))}\n\n"
                f"Apply the same approach:\n\n"
            )
        prompt = f"{_HEADER}{worked}Question: {problem}\nFull solution:\n"

    elif sid == 12:
        ex_ans = ""
        if manual_exemplars:
            ex     = manual_exemplars[0]
            ans_val = extract_final_answer(ex.get('trace', ex.get('raw_answer', '')))
            ex_ans = (
                f"[Example — answer is {ans_val}, "
                f"question: {ex['question']}]\n\n"
            )
        prompt = f"{_HEADER}{ex_ans}Question: {problem}\nAnswer:"

    # GROUP 5 — HYBRID
    elif sid == 13:
        exs = _dyn(k=3)
        prompt = (
            f"{_HEADER}Examples with step-by-step reasoning:\n\n"
            f"{_fmt_exemplars(exs, 3)}"
            f"Now solve step by step:\n"
            f"Question: {problem}\nLet's think step by step.\n"
        )

    elif sid == 14:
        exs = _dyn(k=2)
        prompt = (
            f"{_HEADER}{_fmt_exemplars(exs, 2)}"
            f"Question: {problem}\nLet's think step by step.\n"
        )
        gen_kw['num_return_sequences'] = 3
        gen_kw['temperature']          = 0.9

    elif sid == 15:
        hint = _fmt_category_hint(cat_info)
        exs  = _dyn(k=3)
        prompt = (
            f"{_HEADER}{hint}"
            f"Similar solved problems:\n\n"
            f"{_fmt_exemplars(exs, 3)}"
            f"Solve with numbered steps, end with #### <answer>:\n\n"
            f"Question: {problem}\n\nStep 1:"
        )

    else:
        raise ValueError(f"Unknown strategy_id: {sid}")

    return prompt, gen_kw


# ── Main generator class ──────────────────────────────────────────────────────

class ExemplarGenerator:
    """
    Generates reasoning traces using one of 16 prompting strategies.
    Requires a loaded HuggingFace model and tokenizer (Kaggle/GPU only).
    On local machines, only STRATEGY_NAMES and STRATEGY_EXEMPLAR_NEEDS
    are used — the class itself is never instantiated.
    """

    def __init__(self, model=None, tokenizer=None):
        if not _TORCH_AVAILABLE:
            raise RuntimeError(
                "PyTorch is not available. ExemplarGenerator requires a GPU "
                "environment (Kaggle). For local runs, only import "
                "STRATEGY_NAMES and STRATEGY_EXEMPLAR_NEEDS."
            )
        self.model     = model
        self.tokenizer = tokenizer
        if model is not None:
            self.device = next(model.parameters()).device
        else:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    class _StopOnAnswer:
        """StoppingCriteria: halt when the GENERATED text contains a completed
        '#### <number>' or any stop marker. Independent of stop_strings
        support; only attached for single-sequence generation."""
        def __init__(self, tokenizer, prompt_len):
            self.tok = tokenizer
            self.plen = prompt_len
        def __call__(self, input_ids, scores, **kwargs):
            new_ids = input_ids[0, self.plen:]
            if len(new_ids) < 3:
                return False
            tail = self.tok.decode(new_ids[-32:], skip_special_tokens=True)
            m = ANSWER_STOP_RE.search(tail)
            if m and (m.end() < len(tail) - 1 or len(new_ids) > 8):
                # require a char after the number OR enough tokens, so we
                # don't cut "#### 1" while "#### 12" is still being written
                if tail[m.end():m.end() + 1] not in ("", ".", ","):
                    if not tail[m.end():m.end() + 1].isdigit():
                        return True
            return any(mk in tail for mk in STOP_MARKERS)

    def _banned_token_ids(self):
        """Token-id sequences for BAN_STRINGS (cached; empty on any failure
        so generation never breaks because of this)."""
        if not hasattr(self, "_ban_cache"):
            ids = []
            try:
                for s in BAN_STRINGS:
                    for variant in (s, "\n" + s):
                        toks = self.tokenizer(variant,
                                              add_special_tokens=False).input_ids
                        if toks:
                            ids.append(toks)
            except Exception:
                ids = []
            self._ban_cache = ids
        return self._ban_cache

    def _run_model(
        self,
        prompt               : str,
        temperature          : float = 0.7,
        max_new_tokens       : int   = 512,
        num_return_sequences : int   = 1,
        return_logprobs      : bool  = False,
    ) -> Tuple[List[str], Optional[List]]:
        """
        Run model.generate(). Returns (list_of_decoded_traces, logprobs_or_None).
        Traces contain only the newly generated tokens (prompt stripped).
        """
        enc = self.tokenizer(
            _to_chat_prompt(self.tokenizer, prompt) if CHAT_MODE else prompt,
            return_tensors = "pt",
            truncation     = True,
            max_length     = 3072,
        ).to(self.device)
        prompt_len = enc['input_ids'].shape[1]

        gen_kwargs = dict(
            max_new_tokens          = max_new_tokens,
            temperature             = max(temperature, 1e-4),
            top_p                   = 0.95,
            do_sample               = temperature > 0.01,
            num_return_sequences    = num_return_sequences,
            pad_token_id            = self.tokenizer.eos_token_id,
            output_scores           = return_logprobs,
            return_dict_in_generate = return_logprobs,
            repetition_penalty      = REPETITION_PENALTY,
            no_repeat_ngram_size    = NO_REPEAT_NGRAM,
        )
        bad_words = self._banned_token_ids()
        if bad_words:
            gen_kwargs["bad_words_ids"] = bad_words
        if num_return_sequences == 1:
            from transformers import StoppingCriteriaList
            gen_kwargs["stopping_criteria"] = StoppingCriteriaList([
                self._StopOnAnswer(self.tokenizer, prompt_len),
                loopguard.RepeatLineStopping(self.tokenizer, prompt_len),
            ])
        with torch.no_grad():
            try:
                # stop generating the moment the model starts its own next
                # "Question:" — saves tokens; needs transformers >= 4.39
                out = self.model.generate(
                    **enc, **gen_kwargs,
                    stop_strings = STOP_MARKERS,
                    tokenizer    = self.tokenizer,
                )
            except (TypeError, ValueError):
                out = self.model.generate(**enc, **gen_kwargs)

        if return_logprobs:
            sequences = out.sequences
            scores    = out.scores
        else:
            sequences = out
            scores    = None

        traces = []
        for seq in sequences:
            new_tokens = seq[prompt_len:]
            traces.append(_truncate_trace(
                self.tokenizer.decode(new_tokens, skip_special_tokens=True)
            ))

        logprobs_out = None
        if return_logprobs and scores is not None:
            logprobs_out  = []
            generated_ids = sequences[0][prompt_len:]
            for tok_id, score_tensor in zip(generated_ids, scores):
                log_probs = torch.log_softmax(score_tensor[0], dim=-1)
                lp        = log_probs[tok_id].item()
                tok_str   = self.tokenizer.decode([tok_id])
                logprobs_out.append((tok_str, lp))

        return traces, logprobs_out

    def generate(
        self,
        problem          : str,
        strategy_id      : int,
        manual_exemplars : List[Dict] = None,
        pool_exemplars   : List[Dict] = None,
        category_info    : Optional[Dict] = None,
        max_new_tokens   : int  = 512,
        return_logprobs  : bool = True,
        knn_fn           = None,
    ) -> Dict:
        """
        Generate a reasoning trace for `problem` using `strategy_id`.

        Args:
            problem          : GSM8K question string
            strategy_id      : integer 0-15
            manual_exemplars : pre-selected seed exemplars for this strategy
                               (from manual_exemplars.json, keyed by strategy_id)
            pool_exemplars   : accumulated good pool exemplars
                               (used by KNN strategies 3, 13, 14, 15)
            category_info    : dict from QuestionCategorizer.categorize()
            max_new_tokens   : max tokens to generate
            return_logprobs  : if True, collect token log-probs for REx K metric
            knn_fn           : ExemplarSelector.get_knn_exemplars or None

        Returns:
            {
                'trace'         : str   — generated reasoning text
                'answer'        : str   — extracted final answer
                'strategy_id'   : int
                'strategy_name' : str
                'num_tokens'    : int   — tokens in generated trace
                'candidates'    : list  — all traces (self-consistency only)
                'token_logprobs': list  — [(token_str, log_prob), ...]
            }
        """
        manual_exemplars = manual_exemplars or []
        pool_exemplars   = pool_exemplars   or []

        prompt, gen_kw = _build_prompt(
            strategy_id,
            problem,
            manual_exemplars,
            pool_exemplars,
            category_info,
            knn_fn,
        )

        temperature          = gen_kw.get('temperature', 0.7)
        num_return_sequences = gen_kw.get('num_return_sequences', 1)

        traces, logprobs = self._run_model(
            prompt,
            temperature          = temperature,
            max_new_tokens       = max_new_tokens,
            num_return_sequences = num_return_sequences,
            return_logprobs      = return_logprobs and (num_return_sequences == 1),
        )

        # Self-consistency: majority vote
        if num_return_sequences > 1:
            answers    = [extract_final_answer(t) for t in traces]
            best_ans   = majority_vote(answers)
            best_trace = next(
                (t for t, a in zip(traces, answers) if a == best_ans),
                traces[0]
            )
            logprobs = None
        else:
            best_trace = traces[0]
            best_ans   = extract_final_answer(best_trace)

        return {
            'trace'         : best_trace,
            'answer'        : best_ans,
            'strategy_id'   : strategy_id,
            'strategy_name' : STRATEGY_NAMES[strategy_id],
            'num_tokens'    : (len(self.tokenizer.encode(best_trace))
                               if self.tokenizer else len(best_trace.split())),
            'candidates'    : traces if num_return_sequences > 1 else [],
            'token_logprobs': logprobs,
        }