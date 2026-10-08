"""Paths, colors and small helpers shared by eda.py and reduction.py."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # render straight to files, so the scripts also run without a display
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
FIGURES_DIR = HERE.parent / "figures"
RESULTS_DIR = HERE.parent / "results"
FIGURES_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

CLASS_COLORS = {0: "#2a78d6", 1: "#eb6834"}   # colorblind-safe pair (blue / orange)
CLASS_LABELS = {0: "No stroke (0)", 1: "Stroke (1)"}
GROUP_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
NEUTRAL = "#898781"
UNITS = {"age": "Age (years)", "avg_glucose_level": "Average glucose level (mg/dL, inferred)", "bmi": "BMI (kg/m²)"}

plt.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 150, "axes.grid": True, "grid.alpha": 0.3,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 10,
    "axes.titlesize": 11, "legend.fontsize": 8, "legend.framealpha": 0.9,
})


def save_figure(fig, number, slug, title):
    """Number and title the figure, then write figures/figNN_slug.png."""
    fig.suptitle(f"Figure {number} — {title}", fontsize=13, fontweight="bold")
    fig.tight_layout()
    path = FIGURES_DIR / f"fig{number:02d}_{slug}.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def md_table(df, floatfmt=None):
    """Render a DataFrame as a Markdown table; floatfmt maps column -> format string."""
    floatfmt = floatfmt or {}
    header = "| " + " | ".join(str(c) for c in df.columns) + " |"
    rule = "|" + "|".join("---" for _ in df.columns) + "|"
    rows = []
    for _, row in df.iterrows():
        cells = []
        for c in df.columns:
            v = row[c]
            if c in floatfmt and isinstance(v, (float, np.floating)):
                cells.append(format(v, floatfmt[c]))
            elif isinstance(v, (float, np.floating)):
                cells.append(f"{v:.2f}")
            else:
                cells.append(str(v))
        rows.append("| " + " | ".join(cells) + " |")
    return "\n".join([header, rule, *rows])


def wilson_interval(successes, n, z=1.96):
    """95% Wilson score interval for a proportion (stable for small n and rates near 0)."""
    if n == 0:
        return 0.0, 0.0
    p = successes / n
    center = (p + z**2 / (2 * n)) / (1 + z**2 / n)
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / (1 + z**2 / n)
    return max(0.0, center - half), min(1.0, center + half)


class Results:
    """Collects every number and Markdown table the report quotes, then writes them to results/."""

    def __init__(self, name):
        self.name = name
        self.metrics = {}
        self.tables = []

    def table(self, title, df, floatfmt=None):
        self.tables.append(f"### {title}\n\n{md_table(df, floatfmt)}\n")
        df.to_csv(RESULTS_DIR / f"{self.name}_{len(self.tables):02d}.csv", index=False)

    def write(self):
        (RESULTS_DIR / f"{self.name}_metrics.json").write_text(
            json.dumps(self.metrics, indent=2, default=float) + "\n")
        (RESULTS_DIR / f"{self.name}_tables.md").write_text("\n".join(self.tables))
