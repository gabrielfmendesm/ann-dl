### Table 8 — The chosen network's threshold on the out-of-fold validation probabilities, for three costs of a miss

| cost of a miss | threshold | recall (1) | precision (1) | flagged (% of patients) | cost per patient |
|---|---|---|---|---|---|
| 10 | 0.6711 | 0.6432 | 0.1714 | 18.27 | 0.3251 |
| 20 | 0.4732 | 0.8392 | 0.1257 | 32.51 | 0.4408 |
| 50 | 0.2099 | 0.9849 | 0.0862 | 55.65 | 0.5453 |

### Table 9 — The test set, used once: AP, ROC-AUC and the decisions at each model's validation threshold

| model | AP | ROC-AUC | threshold (validation) | TN | FP | FN | TP | cost per patient |
|---|---|---|---|---|---|---|---|---|
| Network (seed 42) | 0.2997 | 0.8432 | 0.4732 | 687 | 285 | 11 | 39 | 0.4941 |
| Network (mean of 5 seeds) | 0.2622 | 0.8413 | 0.4732 | 697.4 | 274.6 | 10.0 | 40.0 | 0.4644 |
| Majority class | 0.0489 | 0.5000 | 0.0000 | 0 | 972 | 0 | 50 | 0.9511 |
| Logistic regression | 0.2675 | 0.8440 | 0.4280 | 680 | 292 | 8 | 42 | 0.4423 |
| Gradient boosting | 0.2075 | 0.8213 | 0.0188 | 614 | 358 | 10 | 40 | 0.5460 |

### Table 10 — Precision, recall and F1 per class on the test set (0 = no stroke, 1 = stroke)

| model | precision (0) | recall (0) | F1 (0) | precision (1) | recall (1) | F1 (1) | macro F1 |
|---|---|---|---|---|---|---|---|
| Network (seed 42) | 0.9842 | 0.7068 | 0.8228 | 0.1204 | 0.7800 | 0.2086 | 0.5157 |
| Network (mean of 5 seeds) | 0.9859 | 0.7175 | 0.8305 | 0.1272 | 0.8000 | 0.2195 | 0.5250 |
| Majority class | 0.0000 | 0.0000 | 0.0000 | 0.0489 | 1.0000 | 0.0933 | 0.0466 |
| Logistic regression | 0.9884 | 0.6996 | 0.8193 | 0.1257 | 0.8400 | 0.2188 | 0.5190 |
| Gradient boosting | 0.9840 | 0.6317 | 0.7694 | 0.1005 | 0.8000 | 0.1786 | 0.4740 |

### Table 11 — The chosen network retrained with 5 seeds, on the test set

| seed | AP | ROC-AUC | recall (1) | false alarms | cost per patient |
|---|---|---|---|---|---|
| 42 | 0.2997 | 0.8432 | 0.7800 | 285 | 0.4941 |
| 43 | 0.2531 | 0.8344 | 0.8000 | 262 | 0.4521 |
| 44 | 0.2773 | 0.8438 | 0.8000 | 278 | 0.4677 |
| 45 | 0.2474 | 0.8443 | 0.8200 | 271 | 0.4413 |
| 46 | 0.2335 | 0.8408 | 0.8000 | 277 | 0.4667 |
