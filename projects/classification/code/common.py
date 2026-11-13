"""Settings fixed before training, the EDA's data and pipeline, the cross-validation folds and shared helpers.

The data, the train/test split and the preprocessing pipeline are imported unchanged from the EDA's code
(docs/projects/eda/code/preprocessing.py): this deliverable starts from that pipeline, as the statement requires.
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # render straight to files, so the scripts also run without a display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

HERE = Path(__file__).resolve().parent
REPORT = HERE.parent / "index.md"
FIGURES_DIR = HERE.parent / "figures"
RESULTS_DIR = HERE.parent / "results"
EDA_CODE = HERE.parents[1] / "eda" / "code"
EDA_SPLIT = HERE.parents[1] / "eda" / "results" / "split.csv"

# Appended after this folder: `common` stays this file, and `preprocessing` is the EDA's module. The EDA's own
# common.py cannot be imported under the same name, so the small report helpers below repeat it.
sys.path.append(str(EDA_CODE))
from preprocessing import FEATURES, SEED, build_preprocessor, load_raw, split  # noqa: E402

__all__ = ["HERE", "REPORT", "FIGURES_DIR", "RESULTS_DIR", "FEATURES", "SEED", "build_preprocessor", "MAIN_METRIC",
           "CRITERION_GAIN", "COST_MISS", "N_FOLDS", "SEEDS", "CLASS_COLORS", "MODEL_COLORS", "NEUTRAL",
           "load_raw", "load_data", "folds", "save_figure", "md_table", "Results"]

# --- Step 0: fixed before any model was trained --------------------------------------------------
MAIN_METRIC = "average precision"   # the decision of step 0, computed with sklearn's average_precision_score
CRITERION_GAIN = 0.15   # the network's mean test AP (over SEEDS) must be at least 15% above the best baseline's,
#                         and the gap must be larger than the network's standard deviation across SEEDS
COST_MISS = 20          # a missed stroke costs as much as 20 false alarms (a false alarm costs 1)
N_FOLDS = 5             # stratified, shuffled folds of the training rows: the validation sets
SEEDS = (42, 43, 44, 45, 46)

CLASS_COLORS = {0: "#2a78d6", 1: "#eb6834"}   # the EDA's colorblind-safe pair (blue / orange)
MODEL_COLORS = {"Network": "#eb6834", "Logistic regression": "#2a78d6", "Gradient boosting": "#1baf7a",
                "Majority class": "#898781"}
NEUTRAL = "#898781"

plt.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 150, "axes.grid": True, "grid.alpha": 0.3,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 10,
    "axes.titlesize": 11, "legend.fontsize": 8, "legend.framealpha": 0.9,
})


def load_data():
    """The EDA's split, checked row by row against the one the EDA saved in results/split.csv."""
    X_train, X_test, y_train, y_test = split(load_raw())
    saved = pd.read_csv(EDA_SPLIT)
    for name, part in (("train", X_train), ("test", X_test)):
        if list(part.index) != saved.loc[saved["partition"] == name, "row"].tolist():
            raise ValueError(f"the {name} rows differ from the EDA's split")
    return X_train, X_test, y_train.to_numpy(), y_test.to_numpy()


def folds(y_train):
    """The validation folds every model uses: N_FOLDS stratified, shuffled folds of the training rows."""
    splitter = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    return list(splitter.split(np.zeros(len(y_train)), y_train))


def save_figure(fig, number, title):
    """Give the figure its numbered title and write it to figures/fig{number}.png."""
    FIGURES_DIR.mkdir(exist_ok=True)
    fig.suptitle(f"Figure {number} — {title}", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.3 / fig.get_figheight()))   # keep the panels clear of the title
    fig.savefig(FIGURES_DIR / f"fig{number}.png", bbox_inches="tight")
    plt.close(fig)


def md_table(df, floatfmt=None):
    """Render a DataFrame as a Markdown table; floatfmt maps a column to its format spec (default .4f).

    Integers get a thousands separator, negative numbers a true minus sign, as in the prose, and a missing value
    a dash.
    """
    floatfmt = floatfmt or {}
    header = "| " + " | ".join(str(c) for c in df.columns) + " |"
    rule = "|" + "|".join("---" for _ in df.columns) + "|"
    rows = []
    for values in df.astype(object).itertuples(index=False):   # object dtype keeps each cell's own type
        cells = []
        for c, v in zip(df.columns, values):
            if isinstance(v, (float, np.floating)) and np.isnan(v):
                text = "—"
            elif isinstance(v, (float, np.floating)):
                text = format(v, floatfmt.get(c, ".4f"))
            elif isinstance(v, (int, np.integer)) and not isinstance(v, (bool, np.bool_)):
                text = f"{v:,}"
            else:
                text = str(v)
            if isinstance(v, (int, float, np.number)) and text.startswith("-"):
                text = "−" + text[1:]
            cells.append(text)
        rows.append("| " + " | ".join(cells) + " |")
    return "\n".join([header, rule, *rows])


class Results:
    """Collects the numbered tables and the numbers one step of the report quotes, then writes them to results/.

    Each table goes to results/table{NN}.csv and, rendered in Markdown, to results/{name}_tables.md; the numbers
    go to results/{name}_metrics.json. The table numbers are the ones the report uses.
    """

    def __init__(self, name):
        self.name = name
        self.metrics = {}
        self.tables = []
        RESULTS_DIR.mkdir(exist_ok=True)

    def table(self, number, title, df, floatfmt=None):
        """Register Table `number` and write its CSV."""
        self.tables.append(f"### Table {number} — {title}\n\n{md_table(df, floatfmt)}\n")
        df.to_csv(RESULTS_DIR / f"table{number:02d}.csv", index=False)

    def write(self):
        """Write the metrics JSON and the Markdown file with every table of this step."""
        (RESULTS_DIR / f"{self.name}_metrics.json").write_text(
            json.dumps(self.metrics, indent=2, default=float) + "\n")
        (RESULTS_DIR / f"{self.name}_tables.md").write_text("\n".join(self.tables))
