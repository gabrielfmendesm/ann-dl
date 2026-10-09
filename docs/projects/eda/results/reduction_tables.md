### Outliers in the training set (fences fitted on train; no row removed)

| feature | lower fence | upper fence | flagged (1.5×IQR) | flagged (%) | stroke rate among flagged (%) | flagged (modified z > 3.5) |
|---|---|---|---|---|---|---|
| age | -26.50 | 113.50 | 0 | 0.00 | 0.00 | 0 |
| avg_glucose_level | 21.98 | 169.53 | 503 | 12.30 | 13.12 | 458 |
| bmi | 9.35 | 47.35 | 90 | 2.20 | 3.33 | 47 |

### Skewness of the right-skewed features before and after log1p (train, observed values)

| feature | skewness (raw) | skewness (log1p) | max/median (raw) | max/median (log1p) |
|---|---|---|---|---|
| avg_glucose_level | 1.56 | 0.88 | 2.96 | 1.24 |
| bmi | 1.12 | 0.04 | 3.49 | 1.36 |

### Parameters fitted by the pipeline (training rows only)

| feature | imputation median | scaler mean | scaler std |
|---|---|---|---|
| age | 45.000 | 43.3533 | 22.5941 |
| log1p(avg_glucose_level) | 91.945 | 4.6048 | 0.3587 |
| log1p(bmi) | 28.000 | 3.3655 | 0.2515 |

### PCA explained variance (first 8 of 24 components)

| component | explained (%) | cumulative (%) |
|---|---|---|
| PC1 | 30.80 | 30.80 |
| PC2 | 15.20 | 46.00 |
| PC3 | 10.91 | 56.91 |
| PC4 | 8.21 | 65.13 |
| PC5 | 8.03 | 73.15 |
| PC6 | 5.35 | 78.51 |
| PC7 | 4.74 | 83.24 |
| PC8 | 3.26 | 86.51 |

### Largest PCA loadings (eigenvector coefficients) of PC1 and PC2

| feature | PC1 | PC2 |
|---|---|---|
| log__avg_glucose_level | +0.303 | +0.940 |
| num__age | +0.626 | -0.139 |
| log__bmi | +0.541 | -0.215 |
| cat__ever_married_No | -0.256 | +0.088 |
| cat__ever_married_Yes | +0.256 | -0.088 |
| cat__work_type_children | -0.181 | +0.090 |
| cat__smoking_status_Unknown | -0.161 | +0.071 |
| cat__work_type_Private | +0.084 | -0.062 |
| cat__gender_Female | +0.014 | -0.084 |
| cat__gender_Male | -0.013 | +0.083 |

### Projection diagnostics (train, 4,088 rows; base rate 4.87%)

| space | trustworthiness k=5 | trustworthiness k=30 | positives among 10-NN of positives (%) | same categorical profile among 10-NN (%) |
|---|---|---|---|---|
| Original 24 dimensions | — | — | 8.89 | 75.5 |
| PCA (2 components) | 0.817 | 0.820 | 8.84 | 4.5 |
| t-SNE perplexity 5 | 0.996 | 0.954 | 10.20 | 78.2 |
| t-SNE perplexity 30 | 0.997 | 0.977 | 9.70 | 77.7 |
| t-SNE perplexity 50 | 0.996 | 0.978 | 10.20 | 76.7 |
| UMAP n_neighbors 5 | 0.994 | 0.956 | 10.65 | 76.9 |
| UMAP n_neighbors 15 | 0.987 | 0.969 | 9.05 | 74.2 |
| UMAP n_neighbors 50 | 0.981 | 0.971 | 10.50 | 68.2 |
| Control: t-SNE perplexity 30 on independently shuffled columns | 0.992 | 0.944 | 3.97 | 59.7 |
