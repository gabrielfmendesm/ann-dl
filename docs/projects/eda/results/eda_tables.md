### Table 2 — Missing values per column (full file)

| column | missing (count) | missing (%) |
|---|---|---|
| id | 0 | 0.00 |
| gender | 0 | 0.00 |
| age | 0 | 0.00 |
| hypertension | 0 | 0.00 |
| heart_disease | 0 | 0.00 |
| ever_married | 0 | 0.00 |
| work_type | 0 | 0.00 |
| Residence_type | 0 | 0.00 |
| avg_glucose_level | 0 | 0.00 |
| bmi | 201 | 3.93 |
| smoking_status | 0 | 0.00 |
| stroke | 0 | 0.00 |

### Table 3 — Duplicates, impossible and inconsistent values (full file)

| check | rows |
|---|---|
| Fully duplicated rows | 0 |
| Duplicated rows ignoring id | 0 |
| Repeated id | 0 |
| Constant columns | 0 |
| Values outside the documented vocabulary | 0 |
| age <= 0 or age > 120 | 0 |
| Fractional age | 115 |
| avg_glucose_level <= 0 | 0 |
| bmi < 12 (observed) | 3 |
| bmi > 60 (observed) | 13 |
| work_type = children with age >= 18 | 0 |
| ever_married = Yes with age < 18 | 0 |
| smoking_status = smokes with age < 12 | 1 |
| smoking_status = Unknown | 1,544 |
| smoking_status = Unknown with age < 18 | 682 |

### Table 4 — Target distribution in the full file and after the stratified split

| partition | rows | stroke = 0 | stroke = 1 | stroke = 1 (%) |
|---|---|---|---|---|
| Full file | 5,110 | 4,861 | 249 | 4.87 |
| Train | 4,088 | 3,889 | 199 | 4.87 |
| Test | 1,022 | 972 | 50 | 4.89 |

### Table 5 — Descriptive statistics of the numerical features (train, observed values)

| feature | count | missing | mean | median | std | min | Q1 | Q3 | max | skewness | excess kurtosis |
|---|---|---|---|---|---|---|---|---|---|---|---|
| age | 4,088 | 0 | 43.35 | 45.00 | 22.60 | 0.08 | 26.00 | 61.00 | 82.00 | −0.16 | −0.98 |
| avg_glucose_level | 4,088 | 0 | 106.32 | 91.94 | 45.26 | 55.12 | 77.31 | 114.20 | 271.74 | 1.56 | 1.62 |
| bmi | 3,918 | 170 | 28.92 | 28.00 | 7.93 | 10.30 | 23.60 | 33.10 | 97.60 | 1.12 | 3.84 |

### Table 6 — Frequencies and cardinality of the categorical features (train)

| feature | category | count | share (%) | cardinality | rare (< 1%) |
|---|---|---|---|---|---|
| gender | Female | 2,395 | 58.59 | 3 |  |
| gender | Male | 1,692 | 41.39 | 3 |  |
| gender | Other | 1 | 0.02 | 3 | yes |
| hypertension | 0 | 3,691 | 90.29 | 2 |  |
| hypertension | 1 | 397 | 9.71 | 2 |  |
| heart_disease | 0 | 3,867 | 94.59 | 2 |  |
| heart_disease | 1 | 221 | 5.41 | 2 |  |
| ever_married | Yes | 2,700 | 66.05 | 2 |  |
| ever_married | No | 1,388 | 33.95 | 2 |  |
| work_type | Private | 2,332 | 57.05 | 5 |  |
| work_type | Self-employed | 667 | 16.32 | 5 |  |
| work_type | children | 554 | 13.55 | 5 |  |
| work_type | Govt_job | 522 | 12.77 | 5 |  |
| work_type | Never_worked | 13 | 0.32 | 5 | yes |
| Residence_type | Urban | 2,069 | 50.61 | 2 |  |
| Residence_type | Rural | 2,019 | 49.39 | 2 |  |
| smoking_status | never smoked | 1,501 | 36.72 | 4 |  |
| smoking_status | Unknown | 1,247 | 30.50 | 4 |  |
| smoking_status | formerly smoked | 714 | 17.47 | 4 |  |
| smoking_status | smokes | 626 | 15.31 | 4 |  |

### Table 7 — Pairwise correlations of the numerical features (train, observed values)

| pair | n | Pearson r | Spearman ρ |
|---|---|---|---|
| age × avg_glucose_level | 4,088 | 0.233 | 0.141 |
| age × bmi | 3,918 | 0.336 | 0.381 |
| avg_glucose_level × bmi | 3,918 | 0.172 | 0.113 |
| age × bmi, adults only (age ≥ 18) | 3,259 | 0.032 | 0.083 |

### Table 8 — Stroke rate per category (train) with 95% Wilson intervals

| feature | category | n | strokes | stroke rate (%) | 95% CI (%) |
|---|---|---|---|---|---|
| gender | Female | 2,395 | 112 | 4.68 | 3.9–5.6 |
| gender | Male | 1,692 | 87 | 5.14 | 4.2–6.3 |
| gender | Other | 1 | 0 | 0.00 | 0.0–79.3 |
| hypertension | 0 | 3,691 | 145 | 3.93 | 3.3–4.6 |
| hypertension | 1 | 397 | 54 | 13.60 | 10.6–17.3 |
| heart_disease | 0 | 3,867 | 163 | 4.22 | 3.6–4.9 |
| heart_disease | 1 | 221 | 36 | 16.29 | 12.0–21.7 |
| ever_married | No | 1,388 | 23 | 1.66 | 1.1–2.5 |
| ever_married | Yes | 2,700 | 176 | 6.52 | 5.6–7.5 |
| work_type | Govt_job | 522 | 28 | 5.36 | 3.7–7.6 |
| work_type | Never_worked | 13 | 0 | 0.00 | 0.0–22.8 |
| work_type | Private | 2,332 | 115 | 4.93 | 4.1–5.9 |
| work_type | Self-employed | 667 | 55 | 8.25 | 6.4–10.6 |
| work_type | children | 554 | 1 | 0.18 | 0.0–1.0 |
| Residence_type | Rural | 2,019 | 91 | 4.51 | 3.7–5.5 |
| Residence_type | Urban | 2,069 | 108 | 5.22 | 4.3–6.3 |
| smoking_status | Unknown | 1,247 | 38 | 3.05 | 2.2–4.2 |
| smoking_status | formerly smoked | 714 | 56 | 7.84 | 6.1–10.0 |
| smoking_status | never smoked | 1,501 | 71 | 4.73 | 3.8–5.9 |
| smoking_status | smokes | 626 | 34 | 5.43 | 3.9–7.5 |

### Table 9 — Association of each categorical feature with the target (train, χ² test of independence)

| feature | χ² | dof | p-value | Cramér's V | min expected count | p-value without the sparse categories |
|---|---|---|---|---|---|---|
| hypertension | 72.4 | 1 | 1.7e-17 | 0.133 | 19.33 | — |
| heart_disease | 65.8 | 1 | 5.0e-16 | 0.127 | 10.76 | — |
| ever_married | 46.8 | 1 | 7.9e-12 | 0.107 | 67.57 | — |
| work_type | 43.7 | 4 | 7.5e-09 | 0.103 | 0.63 | 2.6e-09 (without Never_worked) |
| smoking_status | 23.1 | 3 | 3.9e-05 | 0.075 | 30.47 | — |
| Residence_type | 1.1 | 1 | 0.29 | 0.017 | 98.28 | — |
| gender | 0.5 | 2 | 0.77 | 0.011 | 0.05 | 0.50 (without Other) |

### Table 10 — Numerical features by target class (train): location and spread

| feature | group | n | median | Q1 | Q3 | IQR | Mann–Whitney p | AUC (feature alone) |
|---|---|---|---|---|---|---|---|---|
| age | stroke = 0 | 3,889 | 44.00 | 24.00 | 59.00 | 35.00 | 3.4e-57 | 0.834 |
| age | stroke = 1 | 199 | 70.00 | 58.50 | 78.00 | 19.50 | 3.4e-57 | 0.834 |
| avg_glucose_level | stroke = 0 | 3,889 | 91.65 | 77.28 | 112.98 | 35.70 | 3.4e-06 | 0.597 |
| avg_glucose_level | stroke = 1 | 199 | 104.86 | 78.44 | 195.47 | 117.03 | 3.4e-06 | 0.597 |
| bmi | stroke = 0 | 3,756 | 27.95 | 23.50 | 33.10 | 9.60 | 2.7e-04 | 0.584 |
| bmi | stroke = 1 | 162 | 29.90 | 26.52 | 33.70 | 7.18 | 2.7e-04 | 0.584 |

### Table 11 — Numerical features grouped by categorical features (train): location and spread

| feature | group | n | median | Q1 | Q3 | IQR |
|---|---|---|---|---|---|---|
| age | work_type = Govt_job | 522 | 51.00 | 41.00 | 62.00 | 21.00 |
| age | work_type = Never_worked | 13 | 17.00 | 16.00 | 18.00 | 2.00 |
| age | work_type = Private | 2,332 | 45.00 | 30.00 | 59.00 | 29.00 |
| age | work_type = Self-employed | 667 | 63.00 | 50.00 | 75.00 | 25.00 |
| age | work_type = children | 554 | 6.00 | 2.00 | 11.00 | 9.00 |
| age | smoking_status = Unknown | 1,247 | 24.00 | 8.00 | 52.00 | 44.00 |
| age | smoking_status = formerly smoked | 714 | 57.00 | 42.00 | 69.00 | 27.00 |
| age | smoking_status = never smoked | 1,501 | 47.00 | 31.00 | 62.00 | 31.00 |
| age | smoking_status = smokes | 626 | 48.00 | 34.00 | 59.00 | 25.00 |
| avg_glucose_level | hypertension = 0 | 3,691 | 91.09 | 77.03 | 112.07 | 35.04 |
| avg_glucose_level | hypertension = 1 | 397 | 103.89 | 79.79 | 196.01 | 116.22 |

### Table 12 — Stroke rate per category within disjoint adult age bands (train)

| feature | category | rate, all ages (%) | n, 18–59 | rate, 18–59 (%) | n, ≥ 60 | rate, ≥ 60 (%) | test p, ≥ 60 |
|---|---|---|---|---|---|---|---|
| hypertension | 0 | 3.93 | 2,149 | 2.00 | 869 | 11.62 | 0.016 (Fisher) |
| hypertension | 1 | 13.60 | 161 | 7.45 | 235 | 17.87 | 0.016 (Fisher) |
| heart_disease | 0 | 4.22 | 2,263 | 2.25 | 930 | 11.94 | 0.026 (Fisher) |
| heart_disease | 1 | 16.29 | 47 | 8.51 | 174 | 18.39 | 0.026 (Fisher) |
| ever_married | No | 1.66 | 631 | 0.79 | 83 | 20.48 | 0.041 (Fisher) |
| ever_married | Yes | 6.52 | 1,679 | 2.98 | 1,021 | 12.34 | 0.041 (Fisher) |
| work_type | Govt_job | 5.36 | 362 | 2.49 | 156 | 12.18 | 0.84 (χ²) |
| work_type | Private | 4.93 | 1,663 | 2.29 | 569 | 13.53 | 0.84 (χ²) |
| work_type | Self-employed | 8.25 | 281 | 2.85 | 379 | 12.40 | 0.84 (χ²) |
| smoking_status | Unknown | 3.05 | 471 | 2.34 | 229 | 11.35 | 0.76 (χ²) |
| smoking_status | formerly smoked | 7.84 | 387 | 3.10 | 304 | 14.47 | 0.76 (χ²) |
| smoking_status | never smoked | 4.73 | 981 | 1.73 | 420 | 12.86 | 0.76 (χ²) |
| smoking_status | smokes | 5.43 | 471 | 3.18 | 151 | 12.58 | 0.76 (χ²) |
