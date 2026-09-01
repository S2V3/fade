# FADE — GSM8K run report

**Model** `meta-llama/Llama-2-7b-chat-hf` · **frozen** (no weight updates)  
**Strategy** 2 = `few_shot` · **Retrieval** `3stage` · **Eval split** `train`  
**Generation** `v5-hashline` · **Embeddings** `minilm`

## 1. Headline results

| Metric | Value |
|---|---|
| Problems evaluated | 1500 |
| **Pass-1 accuracy** (first attempt) | **32.8%** (492/1500) |
| **Final accuracy** (after retry) | **40.9%** (614/1500) |
| Recovered by retry | +122 (8.1% of all problems) |
| GOOD traces in pool | 344 |

## 2. Retry-worthiness — does the classifier predict which failures are worth retrying?

MEDIUM-labelled failures should recover far more often than BAD ones. If the gap is ~0, the label is not a property.

| Iter | MEDIUM recovered | BAD recovered | Gap (MEDIUM−BAD) |
|---|---|---|---|
| 1 | 35/195 (18%) | 87/813 (11%) | **+0.072** |

## 3. Recovery by diagnosed failure type

| Diagnosis | Recovered on retry |
|---|---|
| CE | 34 |
| SM | 23 |
| WP | 18 |
| AL | 13 |
| ST | 13 |
| UNCLASSIFIED | 11 |
| NR | 7 |
| TR | 3 |

## 4. Pass-1 distribution

| Diagnosis (non-GOOD traces) | Count |
|---|---|
| WP | 254 (24%) |
| SM | 253 (24%) |
| CE | 136 (13%) |
| ST | 114 (11%) |
| AL | 94 (9%) |
| UNCLASSIFIED | 88 (8%) |
| NR | 73 (7%) |
| TR | 56 (5%) |

## 5. Accuracy by problem category

| Category | Correct / Total | Accuracy |
|---|---|---|
| percentage | 1/2 | 50.0% |
| monetary | 116/334 | 34.7% |
| counting | 202/694 | 29.1% |
| time | 88/347 | 25.4% |
| rate | 20/98 | 20.4% |
| logic | 5/25 | 20.0% |

## 6. Extraction health

Confirms the accuracy number reflects the model, not parsing bugs.

| Check | Value |
|---|---|
| Traces emitting `####` | 1500/1500 |
| Did the work (E≥0.8) but scored wrong | 69 |
| Gold value in trace but scored wrong | 117 |

## 7. Cost

| Metric | Value |
|---|---|
| Generations | 432 |
| Seconds / generation | 18.36 |
| Wall time | 7932 s (2.2 h) |
| Projected full-GSM8K-train | 38.11 GPU-h |

---
*Reproduce: config in `config_snapshot.json`; per-attempt detail in `results.jsonl`; per-problem outcomes in `results.csv`.*
