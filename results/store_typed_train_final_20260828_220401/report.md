# FADE — GSM8K run report

**Model** `meta-llama/Llama-2-7b-chat-hf` · **frozen** (no weight updates)  
**Strategy** 2 = `few_shot` · **Retrieval** `3stage` · **Eval split** `train`  
**Generation** `v5-hashline` · **Embeddings** `minilm`

## 1. Headline results

| Metric | Value |
|---|---|
| Problems evaluated | 1500 |
| **Pass-1 accuracy** (first attempt) | **20.5%** (308/1500) |
| **Final accuracy** (after retry) | **24.2%** (363/1500) |
| Recovered by retry | +55 (3.7% of all problems) |
| GOOD traces in pool | 170 |

## 2. Retry-worthiness — does the classifier predict which failures are worth retrying?

MEDIUM-labelled failures should recover far more often than BAD ones. If the gap is ~0, the label is not a property.

| Iter | MEDIUM recovered | BAD recovered | Gap (MEDIUM−BAD) |
|---|---|---|---|
| 1 | 9/133 (7%) | 46/1059 (4%) | **+0.024** |

## 3. Recovery by diagnosed failure type

| Diagnosis | Recovered on retry |
|---|---|
| UNCLASSIFIED | 47 |
| CE | 7 |
| ST | 1 |

## 4. Pass-1 distribution

| Diagnosis (non-GOOD traces) | Count |
|---|---|
| UNCLASSIFIED | 713 (52%) |
| CE | 296 (22%) |
| NR | 183 (13%) |
| SM | 168 (12%) |
| ST | 7 (1%) |

## 5. Accuracy by problem category

| Category | Correct / Total | Accuracy |
|---|---|---|
| percentage | 51/227 | 22.5% |
| monetary | 91/443 | 20.5% |
| rate | 21/103 | 20.4% |
| time | 92/539 | 17.1% |
| counting | 27/188 | 14.4% |

## 6. Extraction health

Confirms the accuracy number reflects the model, not parsing bugs.

| Check | Value |
|---|---|
| Traces emitting `####` | 1353/1500 |
| Did the work (E≥0.8) but scored wrong | 84 |
| Gold value in trace but scored wrong | 139 |

## 7. Cost

| Metric | Value |
|---|---|
| Generations | 0 |
| Seconds / generation | 0.49 |
| Wall time | 0 s (0.0 h) |
| Projected full-GSM8K-train | 1.01 GPU-h |

---
*Reproduce: config in `config_snapshot.json`; per-attempt detail in `results.jsonl`; per-problem outcomes in `results.csv`.*
