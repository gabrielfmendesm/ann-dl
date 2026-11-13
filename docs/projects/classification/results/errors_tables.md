### Table 12 — Errors of the network by subgroup on the test set, at the validation threshold

| feature | subgroup | patients | strokes | missed | false alarms | miss rate (%) | false-alarm rate (%) | error rate (%) | cost per patient |
|---|---|---|---|---|---|---|---|---|---|
| age | under 18 | 182 | 1 | 1 | 0 | 100.00 | 0.00 | 0.55 | 0.1099 |
| age | 18–59 | 568 | 11 | 10 | 60 | 90.91 | 10.77 | 12.32 | 0.4577 |
| age | 60 or more | 272 | 38 | 0 | 225 | 0.00 | 96.15 | 82.72 | 0.8272 |
| hypertension | no | 921 | 38 | 10 | 209 | 26.32 | 23.67 | 23.78 | 0.4441 |
| hypertension | yes | 101 | 12 | 1 | 76 | 8.33 | 85.39 | 76.24 | 0.9505 |
| heart disease | no | 967 | 39 | 11 | 247 | 28.21 | 26.62 | 26.68 | 0.4829 |
| heart disease | yes | 55 | 11 | 0 | 38 | 0.00 | 86.36 | 69.09 | 0.6909 |
| BMI | recorded | 991 | 47 | 10 | 272 | 21.28 | 28.81 | 28.46 | 0.4763 |
| BMI | missing | 31 | 3 | 1 | 13 | 33.33 | 46.43 | 45.16 | 1.0645 |
| glucose | below 175 | 906 | 32 | 11 | 212 | 34.38 | 24.26 | 24.61 | 0.4768 |
| glucose | 175 or more | 116 | 18 | 0 | 73 | 0.00 | 74.49 | 62.93 | 0.6293 |
| gender | female | 599 | 29 | 5 | 166 | 17.24 | 29.12 | 28.55 | 0.4441 |
| gender | male | 423 | 21 | 6 | 119 | 28.57 | 29.60 | 29.55 | 0.5650 |

### Table 13 — Share of the non-strokes the network flags, with and without each clinical flag, by age band (test set)

| age | flag | non-strokes with it | median age with it | flagged with it (%) | non-strokes without it | median age without it | flagged without it (%) |
|---|---|---|---|---|---|---|---|
| 18–59 | hypertension | 39 | 52.0 | 66.67 | 518 | 39.0 | 6.56 |
| 18–59 | heart disease | 14 | 52.0 | 64.29 | 543 | 40.0 | 9.39 |
| 18–59 | glucose 175 or more | 45 | 49.0 | 51.11 | 512 | 40.0 | 7.23 |
| 18–59 | any of the three | 85 | 51.0 | 52.94 | 472 | 38.0 | 3.18 |
| 60 or more | hypertension | 50 | 70.5 | 100.00 | 184 | 69.0 | 95.11 |
| 60 or more | heart disease | 29 | 72.0 | 100.00 | 205 | 69.0 | 95.61 |
| 60 or more | glucose 175 or more | 50 | 73.5 | 100.00 | 184 | 69.0 | 95.11 |
| 60 or more | any of the three | 97 | 72.0 | 100.00 | 137 | 67.0 | 93.43 |

### Table 14 — The worst errors: the 5 strokes with the lowest probability and the 5 non-strokes with the highest

| error | id | age | gender | hypertension | heart_disease | avg_glucose_level | bmi | smoking_status | probability |
|---|---|---|---|---|---|---|---|---|---|
| missed stroke | 49669 | 14 | Female | 0 | 0 | 57.93 | 30.9 | Unknown | 0.0283 |
| missed stroke | 39912 | 32 | Female | 0 | 0 | 76.13 | 29.9 | smokes | 0.1475 |
| missed stroke | 54385 | 45 | Male | 0 | 0 | 64.14 | 29.4 | never smoked | 0.1648 |
| missed stroke | 33879 | 42 | Male | 0 | 0 | 83.41 | 25.4 | Unknown | 0.1733 |
| missed stroke | 33943 | 39 | Female | 0 | 0 | 83.24 | 26.3 | never smoked | 0.1760 |
| false alarm | 65955 | 81 | Male | 1 | 1 | 220.64 | 30.0 | never smoked | 0.9297 |
| false alarm | 19271 | 82 | Female | 1 | 1 | 101.56 | 31.5 | never smoked | 0.9282 |
| false alarm | 57080 | 81 | Female | 1 | 1 | 59.11 | 20.7 | formerly smoked | 0.9268 |
| false alarm | 15120 | 81 | Female | 1 | 0 | 210.23 | 30.7 | never smoked | 0.9065 |
| false alarm | 69284 | 81 | Female | 1 | 0 | 174.54 | 26.4 | never smoked | 0.9057 |

### Table 15 — Permutation importance on the test set (30 permutations per feature; test AP 0.2997)

| feature | AP drop (mean) | AP drop (sd) |
|---|---|---|
| age | 0.2113 | 0.0237 |
| avg_glucose_level | 0.0491 | 0.0220 |
| work_type | 0.0259 | 0.0123 |
| heart_disease | 0.0242 | 0.0177 |
| smoking_status | 0.0186 | 0.0084 |
| ever_married | 0.0185 | 0.0122 |
| bmi | 0.0158 | 0.0105 |
| gender | 0.0072 | 0.0087 |
| Residence_type | 0.0048 | 0.0218 |
| hypertension | 0.0041 | 0.0180 |
