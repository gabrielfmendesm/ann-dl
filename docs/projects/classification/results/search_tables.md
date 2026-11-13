### Table 3 — The network of step 2, layer by layer (23 inputs)

| layer | shape | activation | initialization | parameters |
|---|---|---|---|---|
| hidden 1 | 23 → 32 | ReLU | He (Kaiming) normal | 768 |
| hidden 2 | 32 → 16 | ReLU | He (Kaiming) normal | 528 |
| output | 16 → 1 | none (logit) | Glorot (Xavier) normal | 17 |

### Table 4 — Early stopping of the network of step 2 on each fold (losses and AP at the restored epoch)

| fold | validation positives | epochs run | best epoch | training loss | validation loss | validation AP |
|---|---|---|---|---|---|---|
| 1 | 40 | 44 | 14 | 0.8408 | 0.9554 | 0.1947 |
| 2 | 40 | 46 | 16 | 0.8359 | 0.9578 | 0.2312 |
| 3 | 40 | 40 | 10 | 0.8744 | 0.9121 | 0.1569 |
| 4 | 39 | 46 | 16 | 0.8371 | 0.9129 | 0.1375 |
| 5 | 40 | 48 | 18 | 0.8177 | 0.9767 | 0.2460 |

### Table 5 — The network of step 2 (32-16, lr 0.001) with each regularization (5 validation folds; the loss gap is at the restored epoch)

| regularization | AP (mean) | AP (sd) | ROC-AUC (mean) | best epoch (median) | validation − training loss | weights kept by the decay |
|---|---|---|---|---|---|---|
| none | 0.1933 | 0.0465 | 0.8291 | 16 | 0.1018 | 1.0000 |
| weight decay 0.1 | 0.1951 | 0.0416 | 0.8312 | 16 | 0.1064 | 0.9593 |
| weight decay 1 | 0.2038 | 0.0417 | 0.8357 | 20 | 0.0720 | 0.5944 |
| weight decay 10 | 0.1814 | 0.0380 | 0.8235 | 25 | 0.0062 | 0.0015 |
| dropout 0.1 | 0.1957 | 0.0417 | 0.8292 | 17 | 0.0977 | 1.0000 |
| dropout 0.3 | 0.1693 | 0.0242 | 0.8289 | 39 | 0.1177 | 1.0000 |
| dropout 0.5 | 0.1721 | 0.0266 | 0.8319 | 70 | 0.1092 | 1.0000 |
| weight decay 1 + dropout 0.3 | 0.1899 | 0.0396 | 0.8386 | 43 | 0.0585 | 0.3268 |

### Table 6 — The search: 48 configurations on the 5 validation folds, sorted by mean AP

| rank | learning rate | hidden layers | regularization | parameters | AP (mean) | AP (sd) | ROC-AUC (mean) | best epoch (median) | within one SE | chosen |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 0.001 | 32-16 | weight decay 1 | 1,313 | 0.2038 | 0.0417 | 0.8357 | 20 | yes |  |
| 2 | 0.003 | 16 | none | 401 | 0.2033 | 0.0286 | 0.8369 | 8 | yes | ◀ |
| 3 | 0.003 | 16 | weight decay 0.1 | 401 | 0.2031 | 0.0268 | 0.8360 | 12 | yes |  |
| 4 | 0.001 | 16 | weight decay 0.1 | 401 | 0.2017 | 0.0245 | 0.8358 | 35 | yes |  |
| 5 | 0.001 | 16 | none | 401 | 0.1998 | 0.0247 | 0.8363 | 23 | yes |  |
| 6 | 0.001 | 16 | dropout 0.1 | 401 | 0.1982 | 0.0251 | 0.8354 | 41 | yes |  |
| 7 | 0.003 | 32-16 | dropout 0.1 | 1,313 | 0.1969 | 0.0363 | 0.8293 | 8 | yes |  |
| 8 | 0.001 | 32-16 | dropout 0.1 | 1,313 | 0.1957 | 0.0417 | 0.8292 | 17 | yes |  |
| 9 | 0.001 | 32-16 | weight decay 0.1 | 1,313 | 0.1951 | 0.0416 | 0.8312 | 16 | yes |  |
| 10 | 0.003 | 16 | dropout 0.1 | 401 | 0.1948 | 0.0229 | 0.8360 | 12 | yes |  |
| 11 | 0.003 | 16 | weight decay 1 | 401 | 0.1944 | 0.0410 | 0.8384 | 29 | yes |  |
| 12 | 0.003 | 32-16 | weight decay 0.1 | 1,313 | 0.1942 | 0.0370 | 0.8305 | 6 | yes |  |
| 13 | 0.003 | 32-16 | weight decay 1 | 1,313 | 0.1939 | 0.0306 | 0.8375 | 6 | yes |  |
| 14 | 0.003 | 16 | weight decay 1 + dropout 0.3 | 401 | 0.1936 | 0.0457 | 0.8386 | 38 | yes |  |
| 15 | 0.001 | 32-16 | none | 1,313 | 0.1933 | 0.0465 | 0.8291 | 16 | yes |  |
| 16 | 0.001 | 16 | weight decay 1 | 401 | 0.1928 | 0.0415 | 0.8375 | 82 | yes |  |
| 17 | 0.003 | 32-16 | none | 1,313 | 0.1923 | 0.0390 | 0.8291 | 6 | yes |  |
| 18 | 0.003 | 16 | dropout 0.5 | 401 | 0.1906 | 0.0372 | 0.8345 | 32 | yes |  |
| 19 | 0.001 | 32-16 | weight decay 1 + dropout 0.3 | 1,313 | 0.1899 | 0.0396 | 0.8386 | 43 | yes |  |
| 20 | 0.003 | 32-16 | weight decay 1 + dropout 0.3 | 1,313 | 0.1886 | 0.0347 | 0.8390 | 18 | yes |  |
| 21 | 0.001 | 16 | weight decay 1 + dropout 0.3 | 401 | 0.1878 | 0.0396 | 0.8368 | 109 | yes |  |
| 22 | 0.001 | 16 | dropout 0.5 | 401 | 0.1864 | 0.0358 | 0.8338 | 71 | yes |  |
| 23 | 0.001 | 64-32 | weight decay 10 | 3,649 | 0.1864 | 0.0438 | 0.8317 | 18 | yes |  |
| 24 | 0.003 | 16 | dropout 0.3 | 401 | 0.1861 | 0.0318 | 0.8348 | 19 | yes |  |
| 25 | 0.001 | 64-32 | weight decay 1 + dropout 0.3 | 3,649 | 0.1852 | 0.0506 | 0.8351 | 17 | yes |  |
| 26 | 0.001 | 16 | dropout 0.3 | 401 | 0.1852 | 0.0341 | 0.8343 | 57 | yes |  |
| 27 | 0.003 | 64-32 | weight decay 1 | 3,649 | 0.1839 | 0.0405 | 0.8327 | 4 |  |  |
| 28 | 0.003 | 64-32 | none | 3,649 | 0.1832 | 0.0504 | 0.8304 | 2 |  |  |
| 29 | 0.003 | 64-32 | weight decay 0.1 | 3,649 | 0.1829 | 0.0520 | 0.8300 | 2 |  |  |
| 30 | 0.003 | 64-32 | dropout 0.5 | 3,649 | 0.1814 | 0.0373 | 0.8357 | 11 |  |  |
| 31 | 0.001 | 32-16 | weight decay 10 | 1,313 | 0.1814 | 0.0380 | 0.8235 | 25 |  |  |
| 32 | 0.003 | 64-32 | dropout 0.1 | 3,649 | 0.1810 | 0.0527 | 0.8295 | 3 |  |  |
| 33 | 0.003 | 64-32 | weight decay 1 + dropout 0.3 | 3,649 | 0.1808 | 0.0448 | 0.8366 | 10 |  |  |
| 34 | 0.003 | 64-32 | dropout 0.3 | 3,649 | 0.1787 | 0.0440 | 0.8344 | 5 |  |  |
| 35 | 0.003 | 64-32 | weight decay 10 | 3,649 | 0.1782 | 0.0383 | 0.8290 | 27 |  |  |
| 36 | 0.001 | 64-32 | dropout 0.1 | 3,649 | 0.1771 | 0.0477 | 0.8309 | 10 |  |  |
| 37 | 0.001 | 64-32 | weight decay 1 | 3,649 | 0.1762 | 0.0429 | 0.8308 | 8 |  |  |
| 38 | 0.003 | 32-16 | weight decay 10 | 1,313 | 0.1752 | 0.0327 | 0.8205 | 10 |  |  |
| 39 | 0.001 | 64-32 | dropout 0.5 | 3,649 | 0.1749 | 0.0402 | 0.8326 | 41 |  |  |
| 40 | 0.001 | 32-16 | dropout 0.5 | 1,313 | 0.1721 | 0.0266 | 0.8319 | 70 |  |  |
| 41 | 0.003 | 32-16 | dropout 0.5 | 1,313 | 0.1720 | 0.0314 | 0.8322 | 21 |  |  |
| 42 | 0.003 | 32-16 | dropout 0.3 | 1,313 | 0.1703 | 0.0290 | 0.8289 | 11 |  |  |
| 43 | 0.001 | 32-16 | dropout 0.3 | 1,313 | 0.1693 | 0.0242 | 0.8289 | 39 |  |  |
| 44 | 0.001 | 64-32 | dropout 0.3 | 3,649 | 0.1691 | 0.0335 | 0.8306 | 15 |  |  |
| 45 | 0.001 | 64-32 | weight decay 0.1 | 3,649 | 0.1683 | 0.0436 | 0.8277 | 7 |  |  |
| 46 | 0.001 | 64-32 | none | 3,649 | 0.1680 | 0.0443 | 0.8274 | 7 |  |  |
| 47 | 0.001 | 16 | weight decay 10 | 401 | 0.1647 | 0.0223 | 0.7951 | 57 |  |  |
| 48 | 0.003 | 16 | weight decay 10 | 401 | 0.1642 | 0.0254 | 0.7990 | 29 |  |  |

### Table 7 — The chosen configuration with the adult BMIs below 12 set to missing (5 validation folds)

| data | AP (mean) | AP (sd) | ROC-AUC (mean) | best epoch (median) |
|---|---|---|---|---|
| as recorded | 0.2033 | 0.0286 | 0.8369 | 8 |
| the 2 adult BMIs below 12 set to missing | 0.1990 | 0.0263 | 0.8368 | 8 |
