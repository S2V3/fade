# FADE — GSM8K run report

**Model** `Qwen/Qwen3.5-4B` · **frozen** (no weight updates)  
**Strategy** 2 = `few_shot` · **Retrieval** `3stage` · **Eval split** `train`  
**Generation** `v5-hashline` · **Embeddings** `minilm`

## 1. Headline results

| Metric | Value |
|---|---|
| Problems evaluated | 250 |
| **Pass-1 accuracy** (first attempt) | **91.2%** (228/250) |
| **Final accuracy** (after retry) | **93.2%** (233/250) |
| Recovered by retry | +5 (2.0% of all problems) |
| GOOD traces in pool | 108 |

## 2. Retry-worthiness — does the classifier predict which failures are worth retrying?

MEDIUM-labelled failures should recover far more often than BAD ones. If the gap is ~0, the label is not a property.

| Iter | MEDIUM recovered | BAD recovered | Gap (MEDIUM−BAD) |
|---|---|---|---|
| 1 | 0/3 (0%) | 5/19 (26%) | **-0.263** |

## 3. Recovery by diagnosed failure type

| Diagnosis | Recovered on retry |
|---|---|
| TR | 3 |
| AL | 2 |

## 4. Pass-1 distribution

| Diagnosis (non-GOOD traces) | Count |
|---|---|
| AL | 20 (67%) |
| TR | 6 (20%) |
| WP | 2 (7%) |
| CE | 1 (3%) |
| UNCLASSIFIED | 1 (3%) |

## 5. Accuracy by problem category

| Category | Correct / Total | Accuracy |
|---|---|---|
| rate | 16/16 | 100.0% |
| monetary | 52/57 | 91.2% |
| counting | 97/110 | 88.2% |
| time | 51/62 | 82.3% |
| logic | 4/5 | 80.0% |

## 6. Extraction health

Confirms the accuracy number reflects the model, not parsing bugs.

| Check | Value |
|---|---|
| Traces emitting `####` | 250/250 |
| Did the work (E≥0.8) but scored wrong | 13 |
| Gold value in trace but scored wrong | 13 |

## 7. Cost

| Metric | Value |
|---|---|
| Generations | 2 |
| Seconds / generation | 55.0 |
| Wall time | 110 s (0.0 h) |
| Projected full-GSM8K-train | 114.17 GPU-h |

---
*Reproduce: config in `config_snapshot.json`; per-attempt detail in `results.jsonl`; per-problem outcomes in `results.csv`.*
