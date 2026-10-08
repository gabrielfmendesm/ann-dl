"""Reproduce every table, number and figure of the EDA report.

Usage, from the repository root:
    python docs/projects/eda/code/main.py

Writes Figures 1-13 to figures/ and the tables (Markdown and CSV) and numbers (JSON) to results/.
"""

import eda
import reduction

if __name__ == "__main__":
    eda.main()
    reduction.main()
