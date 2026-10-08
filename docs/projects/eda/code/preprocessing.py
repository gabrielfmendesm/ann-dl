"""Data loading, the train/test split and the preprocessing pipeline of the Stroke EDA.

Importable: nothing is fitted at import time. Typical use from the modeling deliverable:

    from preprocessing import load_raw, split, build_preprocessor
    X_train, X_test, y_train, y_test = split(load_raw())
    preprocess = build_preprocessor()
    X_train_t = preprocess.fit_transform(X_train)   # fit on train only
    X_test_t = preprocess.transform(X_test)         # test is only transformed

Every choice in build_preprocessor() is motivated in section 4A of the report.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import MissingIndicator, SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

SEED = 42
DATA_PATH = Path(__file__).resolve().parents[4] / "data" / "stroke-prediction" / "healthcare-dataset-stroke-data.csv"

TARGET = "stroke"
ID_COLUMN = "id"
SKEWED_FEATURES = ["avg_glucose_level", "bmi"]      # right-skewed (skewness > 1): log1p, then standardize
SYMMETRIC_FEATURES = ["age"]                         # near-symmetric: standardize only
NUMERICAL_FEATURES = SYMMETRIC_FEATURES + SKEWED_FEATURES
CATEGORICAL_FEATURES = ["gender", "hypertension", "heart_disease", "ever_married",
                        "work_type", "Residence_type", "smoking_status"]
FEATURES = NUMERICAL_FEATURES + CATEGORICAL_FEATURES


def load_raw(path=DATA_PATH):
    """Read the CSV exactly as published. The text 'N/A' in bmi is parsed as NaN by pandas."""
    return pd.read_csv(path)


def split(df, test_size=0.2):
    """Stratified 80/20 split with a fixed seed; returns X_train, X_test, y_train, y_test.

    The file is sorted by the target (the 249 positives are its first rows), so the split
    must shuffle; stratification keeps the 4.87% minority rate identical in both parts.
    """
    X = df[FEATURES]
    y = df[TARGET]
    return train_test_split(X, y, test_size=test_size, stratify=y, shuffle=True, random_state=SEED)


def build_preprocessor():
    """Return an unfitted ColumnTransformer whose output is a dense, finite matrix.

    - age: median imputation (a safeguard: the column has no missing value) + standardization;
    - avg_glucose_level, bmi: median imputation + log1p + standardization;
    - bmi missing indicator: 0/1, kept unscaled;
    - categorical: constant imputation + full one-hot (no dropped level, so no category is
      placed closer to the others than they are to each other); a category never seen in
      training is encoded as an all-zero block, so it is never confused with a known one.
    """
    symmetric = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    skewed = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("log1p", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
        ("scale", StandardScaler()),
    ])
    categorical = Pipeline([
        ("impute", SimpleImputer(strategy="constant", fill_value="missing")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer([
        ("num", symmetric, SYMMETRIC_FEATURES),
        ("log", skewed, SKEWED_FEATURES),
        ("missing", MissingIndicator(features="all"), ["bmi"]),
        ("cat", categorical, CATEGORICAL_FEATURES),
    ], remainder="drop")
