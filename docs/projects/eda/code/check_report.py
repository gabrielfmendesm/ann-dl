"""Check that the report quotes exactly what the scripts produced.

Usage, from the repository root, after main.py:
    python docs/projects/eda/code/check_report.py

Every table in results/*_tables.md must appear verbatim in index.md, every figure file must be shown exactly
once, and the results-summary table must carry the numbers stored in results/*_metrics.json.
"""

import json
import re
import sys

from common import FIGURES_DIR, REPORT, RESULTS_DIR

N_FIGURES = 13


def main():
    report = REPORT.read_text()
    problems = []

    tables = 0
    for stage in ("eda", "reduction"):
        text = (RESULTS_DIR / f"{stage}_tables.md").read_text()
        for number, body in re.findall(r"^### Table (\d+) — .*\n\n((?:\|.*\n)+)", text, re.M):
            tables += 1
            if body.strip() not in report:
                problems.append(f"Table {number} in the report differs from results/table{int(number):02d}.csv")

    for n in range(1, N_FIGURES + 1):
        if not (FIGURES_DIR / f"fig{n}.png").is_file():
            problems.append(f"figures/fig{n}.png is missing")
        if report.count(f"](figures/fig{n}.png)") != 1:
            problems.append(f"Figure {n} is not shown exactly once")
        if f"**Conclusion — Figure {n}.**" not in report:
            problems.append(f"Figure {n} has no conclusion")

    m = {}
    for stage in ("eda", "reduction"):
        m.update(json.loads((RESULTS_DIR / f"{stage}_metrics.json").read_text()))
    summary = report.split("## Results summary")[1]
    expected = {
        "row 3": f"`bmi`: {m['bmi_missing']} missing, {m['bmi_missing_pct']:.2f}% of the file",
        "row 5": f"{m['n_pos']} of {m['raw_shape'][0]:,} = {m['pos_pct']:.2f}%",
        "row 6": f"Train {m['train_rows']:,} ({m['train_pos']} positives) · "
                 f"test {m['test_rows']:,} ({m['test_pos']} positives)",
        "row 7": f"{m['top_pair']}, Spearman ρ = {m['top_pair_rho']:.3f}",
        "row 8": f"{m['outlier_rows']} training rows ({m['outlier_rows_pct']:.2f}%)",
        "row 9": f"{m['pca_pc12']:.2f}% (PC1 {m['pca_pc1']:.2f}% + PC2 {m['pca_pc2']:.2f}%)",
        "row 10": f"train ({m['train_shape'][0]}, {m['train_shape'][1]}) · "
                  f"test ({m['test_shape'][0]}, {m['test_shape'][1]})",
    }
    for row, text in expected.items():
        if text not in summary:
            problems.append(f"results summary {row} does not contain '{text}'")

    if problems:
        print("\n".join(problems))
        sys.exit(1)
    print(f"Report consistent with results/: {tables} tables, {N_FIGURES} figures and the results summary.")


if __name__ == "__main__":
    main()
