# FADE — failure-type stability

300 questions x k=5 samples at T=0.8.

## Headline

| quantity | value |
|---|---|
| Fleiss kappa, full label space | 0.2369 |
| **Fleiss kappa, failure TYPE (wrong samples only)** | **0.2239** |
| Cohen kappa, sample 1 vs sample 2 | 0.2348 |
| questions with >=2 wrong samples | 278 |
| mean modal-rating share | 0.657 |
| mean entropy | 1.014 bits |
| all 5 samples agree | 36/300 |

The bolded row is the ceiling on any question-only failure-type predictor.

## Elicitation gap

| | |
|---|---|
| accuracy@1 | 0.163 |
| majority@5 | 0.227 |
| pass@5 | 0.407 |

## Verdict

**WEAKLY STABLE.** kappa=0.224. Some reproducible signal, fair at best. A predictor could in principle reach ~0.22 agreement -- report the ceiling alongside the gate, and only build the predictor if that margin is worth the claim.

*Caveat: Measured at T=0.80. Stability is temperature-dependent; a greedy (T=0) run is deterministic and would trivially give kappa=1, which is why this uses the same sampling regime a real deployment would.*
