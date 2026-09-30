"""Exercise 3 — Preparing Real-World Data for a Neural Network.

Loads the Spaceship Titanic training file, describes it, splits it 80/20
(stratified, fixed seed) BEFORE any transformation, then imputes, encodes,
engineers TotalSpend, applies log(1 + x) to the spending columns and scales
everything to [-1, 1] for a tanh network — every statistic fitted on the
training split only. Produces Figure 6 and the numbers reported in the text.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

HERE = Path(__file__).resolve().parent
FIGURES_DIR = HERE.parent / "figures"
FIGURES_DIR.mkdir(exist_ok=True)
DATA_PATH = HERE.parents[3] / "data" / "spaceship-titanic" / "train.csv"  # repository root / data / ...

# Fixed seed; the generator is used for the stratified split
rng = np.random.default_rng(42)

TARGET = "Transported"
SPENDING = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]
NUMERICAL = ["Age"] + SPENDING
CATEGORICAL = ["HomePlanet", "CryoSleep", "Destination", "VIP"]
DROPPED = ["Cabin", "Name", "PassengerId"]

# ---------------------------------------------------------------- A: get to know the data
df = pd.read_csv(DATA_PATH)
print(f"Loaded {DATA_PATH.name}: {df.shape[0]} rows x {df.shape[1]} columns")

balance = df[TARGET].value_counts()
print(f"\nTarget balance: True = {balance[True]} ({balance[True] / len(df):.2%}), "
      f"False = {balance[False]} ({balance[False] / len(df):.2%})")

print("\nColumn types:")
print(f"  numerical   : {NUMERICAL}")
print(f"  categorical : {CATEGORICAL} (+ Cabin, a high-cardinality code with {df['Cabin'].nunique()} unique values)")
print(f"  identifiers : PassengerId, Name (dropped)")

missing = pd.DataFrame({"missing": df.isna().sum(), "percent": (df.isna().mean() * 100).round(2)})
print("\nMissing values per column:")
print(missing.to_string())

spending_stats = df[SPENDING].agg(["mean", "median", "max", "skew"]).T.round(2)
print("\nSpending columns (full train.csv):")
print(spending_stats.to_string())

# ---------------------------------------------------------------- B: split before you transform
# Stratified 80/20 split done by hand with the seeded generator: shuffle the
# rows of each class separately and send 20% of each class to the test set.
y_all = df[TARGET].astype(int).to_numpy()
test_idx = []
for label in (0, 1):
    idx = np.flatnonzero(y_all == label)
    idx = rng.permutation(idx)
    test_idx.append(idx[: int(round(0.2 * len(idx)))])
test_idx = np.sort(np.concatenate(test_idx))
train_mask = np.ones(len(df), dtype=bool)
train_mask[test_idx] = False

train_df = df[train_mask].reset_index(drop=True)
test_df = df[~train_mask].reset_index(drop=True)
y_train, y_test = train_df[TARGET].astype(int).to_numpy(), test_df[TARGET].astype(int).to_numpy()
print(f"\nSplit: train = {len(train_df)} rows ({len(train_df) / len(df):.1%}), "
      f"test = {len(test_df)} rows ({len(test_df) / len(df):.1%})")
print(f"Positive share: full = {y_all.mean():.4f}, train = {y_train.mean():.4f}, test = {y_test.mean():.4f}")

fc_train = train_df["FoodCourt"]
print(f"FoodCourt on the training set, before transforming: mean = {fc_train.mean():.2f}, "
      f"median = {fc_train.median():.2f}, max = {fc_train.max():.0f}")

# ---------------------------------------------------------------- C: preprocess (fit on train, apply to test)
# 1) Missing data: median for numerical columns, most frequent value for categorical ones
medians = train_df[NUMERICAL].median()
modes = train_df[CATEGORICAL].mode().iloc[0]
print("\nImputation values learned on the training set:")
print("  medians :", medians.round(2).to_dict())
print("  modes   :", modes.to_dict())


def impute(frame):
    out = frame.copy()
    out[NUMERICAL] = out[NUMERICAL].fillna(medians)
    out[CATEGORICAL] = out[CATEGORICAL].fillna(modes)
    return out


train_imp, test_imp = impute(train_df), impute(test_df)

# 2) Feature engineering: TotalSpend = sum of the five spending columns; drop identifiers
for frame in (train_imp, test_imp):
    frame["TotalSpend"] = frame[SPENDING].sum(axis=1)
    frame.drop(columns=DROPPED, inplace=True)
LOG_COLS = SPENDING + ["TotalSpend"]
NUM_FINAL = ["Age"] + LOG_COLS

# 3) Heavy tails: log(1 + x) on the spending columns (and on their sum)
fc_raw_train = train_imp["FoodCourt"].to_numpy().copy()  # kept for Figure 6
for frame in (train_imp, test_imp):
    frame[LOG_COLS] = np.log1p(frame[LOG_COLS])
fc_log_train = train_imp["FoodCourt"].to_numpy().copy()
print(f"\nFoodCourt (train) after log(1 + x): mean = {fc_log_train.mean():.3f}, median = {np.median(fc_log_train):.3f}, "
      f"max = {fc_log_train.max():.3f}, skew = {pd.Series(fc_log_train).skew():.2f} "
      f"(raw skew = {pd.Series(fc_raw_train).skew():.2f})")

# 4) Categorical features: one-hot encoding fitted on the training categories.
#    handle_unknown="ignore" -> a category seen only in the test set produces an
#    all-zero row for that feature instead of an error.
encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype=float)
encoder.fit(train_imp[CATEGORICAL].astype(str))
onehot_names = list(encoder.get_feature_names_out(CATEGORICAL))
print(f"\nOne-hot columns ({len(onehot_names)}): {onehot_names}")
unseen = {c: sorted(set(test_imp[c].astype(str)) - set(train_imp[c].astype(str))) for c in CATEGORICAL}
print("Categories present in test but absent from train:", unseen)

# 5) Scaling: min-max normalization to [-1, 1], fitted on the training numerical columns
scaler = MinMaxScaler(feature_range=(-1, 1))
scaler.fit(train_imp[NUM_FINAL])


def transform(frame):
    num = scaler.transform(frame[NUM_FINAL])
    cat = encoder.transform(frame[CATEGORICAL].astype(str))
    return np.hstack([num, cat])


X_train, X_test = transform(train_imp), transform(test_imp)
feature_names = NUM_FINAL + onehot_names

# ---------------------------------------------------------------- D: verify and visualize
print("\nFinal checks:")
print(f"  NaN in X_train: {np.isnan(X_train).sum()} | NaN in X_test: {np.isnan(X_test).sum()}")
print(f"  X_train shape: {X_train.shape} | X_test shape: {X_test.shape} | features: {len(feature_names)}")
print(f"  X_train min/max: {X_train.min():.4f} / {X_train.max():.4f}")
print(f"  X_test  min/max: {X_test.min():.4f} / {X_test.max():.4f}")
num_test = X_test[:, : len(NUM_FINAL)]
print(f"  numerical block only — train: [{X_train[:, :len(NUM_FINAL)].min():.4f}, {X_train[:, :len(NUM_FINAL)].max():.4f}], "
      f"test: [{num_test.min():.4f}, {num_test.max():.4f}]")
print(f"  columns of X_test outside [-1, 1]: "
      f"{[feature_names[j] for j in range(X_test.shape[1]) if X_test[:, j].min() < -1 or X_test[:, j].max() > 1]}")

# Figure 6: FoodCourt on the training set — raw, after log(1 + x), after log(1 + x) + scaling
fc_scaled_train = X_train[:, feature_names.index("FoodCourt")]
fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
panels = [
    (fc_raw_train, "before preprocessing (raw values)", "FoodCourt (credits spent)"),
    (fc_log_train, "after log(1 + x)", "log(1 + FoodCourt)"),
    (fc_scaled_train, "after log(1 + x) and scaling to [-1, 1]", "scaled FoodCourt"),
]
for ax, (values, title, xlabel) in zip(axes, panels):
    bins = np.linspace(values.min(), values.max(), 40)
    for label, color, name in [(0, "tab:blue", "Transported = False"), (1, "tab:orange", "Transported = True")]:
        ax.hist(values[y_train == label], bins=bins, alpha=0.6, color=color, label=name)
    ax.set(title=title, xlabel=xlabel, ylabel="count (training set)")
    ax.legend()
axes[0].set_yscale("log")
axes[0].set_ylabel("count (training set, log scale)")
fig.suptitle("Figure 6 — FoodCourt on the training set before and after preprocessing")
fig.tight_layout()
fig.savefig(FIGURES_DIR / "fig6.png", dpi=150)

print("\nFigure saved to", FIGURES_DIR / "fig6.png")
