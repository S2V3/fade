# FADE — GSM8K run report

**Model** `Qwen/Qwen3.5-4B` · **frozen** (no weight updates)  
**Strategy** 2 = `few_shot` · **Retrieval** `3stage` · **Eval split** `train`  
**Generation** `v5-hashline` · **Embeddings** `minilm`

## 1. Headline results

| Metric | Value |
|---|---|
| Problems evaluated | 250 |
| **Pass-1 accuracy** (first attempt) | **88.4%** (221/250) |
| **Final accuracy** (after retry) | **92.4%** (231/250) |
| Recovered by retry | +10 (4.0% of all problems) |
| GOOD traces in pool | 95 |

## 2. Retry-worthiness — does the classifier predict which failures are worth retrying?

MEDIUM-labelled failures should recover far more often than BAD ones. If the gap is ~0, the label is not a property.

| Iter | MEDIUM recovered | BAD recovered | Gap (MEDIUM−BAD) |
|---|---|---|---|
| 1 | 0/1 (0%) | 10/28 (36%) | **-0.357** |

## 3. Recovery by diagnosed failure type

| Diagnosis | Recovered on retry |
|---|---|
| AL | 5 |
| TR | 4 |
| CE | 1 |

## 4. Pass-1 distribution

| Diagnosis (non-GOOD traces) | Count |
|---|---|
| AL | 21 (64%) |
| TR | 7 (21%) |
| NR | 2 (6%) |
| CE | 2 (6%) |
| ST | 1 (3%) |

## 5. Accuracy by problem category

| Category | Correct / Total | Accuracy |
|---|---|---|
| logic | 5/5 | 100.0% |
| counting | 98/106 | 92.5% |
| monetary | 50/60 | 83.3% |
| rate | 10/12 | 83.3% |
| time | 54/67 | 80.6% |

## 6. Extraction health

Confirms the accuracy number reflects the model, not parsing bugs.

| Check | Value |
|---|---|
| Traces emitting `####` | 250/250 |
| Did the work (E≥0.8) but scored wrong | 14 |
| Gold value in trace but scored wrong | 13 |

## 7. Cost

| Metric | Value |
|---|---|
| Generations | 9 |
| Seconds / generation | 61.17 |
| Wall time | 550 s (0.2 h) |
| Projected full-GSM8K-train | 126.97 GPU-h |

---
*Reproduce: config in `config_snapshot.json`; per-attempt detail in `results.jsonl`; per-problem outcomes in `results.csv`.*
