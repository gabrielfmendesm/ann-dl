"""Build docs/projects/eda/eda.ipynb and docs/projects/classification/classification.ipynb from each report's code.

Usage, from the repository root:
    python tools/build_notebooks.py

The notebooks are assembled from the same modules the reports run (docs/projects/<project>/code/), one cell per
top-level definition, so they do exactly what `main.py` does and write the same figures/ and results/. Nothing is
rewritten here: each cell holds the module's own source, with only the imports between those modules removed (a
notebook has one namespace) and `__file__`, which a notebook does not have, resolved to the same paths.
"""

import ast
import json
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PROJECTS = REPO / "docs" / "projects"
NOTICE = """> **The same work as the report, in a notebook.** Nothing here is new: every cell below is the
> code of `code/`, as it is there, in the order `main.py` runs it, one cell per definition. Running the whole
> notebook writes the same figures to `figures/` and the same tables and numbers to `results/` as
> `python code/main.py` does. The report itself is [`index.md`](index.md), next to this notebook."""

LOCAL = {"common", "preprocessing", "network", "validation", "eda", "reduction", "baselines", "errors",
         "evaluate", "search", "starting_point", "check_report"}


def clean(source, module):
    """The module's source as notebook cells use it.

    The imports between these modules go (a notebook has one namespace), `__file__` becomes the same paths the
    scripts resolve, the `__main__` guard goes, and the entry point run()/main() — every module names it the same —
    becomes <module>_run / <module>_main so one module does not overwrite another.
    """
    tree = ast.parse(source)
    lines = source.splitlines(keepends=True)
    drop = set()
    for node in tree.body:   # only top-level statements
        local = (isinstance(node, ast.ImportFrom) and node.module in LOCAL) or \
                (isinstance(node, ast.Import) and any(a.name in LOCAL for a in node.names))
        guard = (isinstance(node, ast.If) and isinstance(node.test, ast.Compare)
                 and getattr(node.test.left, "id", None) == "__name__")
        path_hack = (isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
                     and ast.unparse(node.value).startswith("sys.path.append"))
        if local or guard or path_hack:
            drop.update(range(node.lineno - 1, node.end_lineno))
    source = "".join(line for i, line in enumerate(lines) if i not in drop)
    source = re.sub(r"^def (run|main)\(", rf"def {module}_\1(", source, flags=re.M)
    source = source.replace("Path(__file__).resolve().parents[4]", "REPO")
    source = source.replace("Path(__file__).resolve().parent", "CODE_DIR")
    return re.sub(r"\n{3,}", "\n\n\n", source).strip("\n")


def blocks(source):
    """The cleaned module split into cells: everything before the first definition, then one cell per definition."""
    lines = source.splitlines(keepends=True)
    tops = [n for n in ast.parse(source).body if isinstance(n, (ast.FunctionDef, ast.ClassDef))]
    cells, start = [], 0
    for node in tops:
        first = min([node.lineno] + [d.lineno for d in node.decorator_list]) - 1
        while first > start and lines[first - 1].lstrip().startswith("#"):   # keep a comment with its definition
            first -= 1
        if "".join(lines[start:first]).strip():
            cells.append(("".join(lines[start:first]).strip("\n"), None))
        cells.append(("".join(lines[first:node.end_lineno]).strip("\n"), node.name))
        start = node.end_lineno
    if "".join(lines[start:]).strip():
        cells.append(("".join(lines[start:]).strip("\n"), None))
    return cells


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def code(text):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
            "source": text.splitlines(keepends=True)}


DEFINED = {}
EMITTED = []


def module_cells(project, module, title, source_project=None):
    """A heading for the module, then its definitions, each in its own cell.

    source_project names the project the file comes from when it is not this one: the classification deliverable
    reuses the EDA's preprocessing.py, exactly as its common.py imports it.
    """
    path = PROJECTS / (source_project or project) / "code" / f"{module}.py"
    raw = path.read_text()
    for node in ast.parse(raw).body:   # the cells run in order: every module it imports must already be above
        if isinstance(node, ast.ImportFrom) and node.module in LOCAL:
            assert node.module in EMITTED, f"{module}.py imports {node.module}.py, which comes later"
    EMITTED.append(module)
    source = clean(raw, module)
    for node in ast.parse(source).body:   # one namespace: nothing may be defined twice
        names = ([node.name] if isinstance(node, (ast.FunctionDef, ast.ClassDef)) else
                 [x.id for x in node.targets if isinstance(x, ast.Name)] if isinstance(node, ast.Assign) else [])
        for name in names:
            assert name not in DEFINED, f"{module}.py redefines {name}, already defined by {DEFINED[name]}.py"
            DEFINED[name] = module
    where = (f"../{source_project}/code/{module}.py" if source_project else f"code/{module}.py")
    cells = [md(f"## [`{'../eda/code' if source_project else 'code'}/{module}.py`]({where}) — {title}")]
    for text, name in blocks(source):
        if name:
            cells.append(md(f"### `{name}`"))
        cells.append(code(text))
    return cells


def setup_cell(project):
    return code(f'''from pathlib import Path
from IPython.display import Image, Markdown, display

# The notebook runs from anywhere inside the repository and uses the same paths as the scripts.
REPO = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / "mkdocs.yml").is_file())
CODE_DIR = REPO / "docs" / "projects" / "{project}" / "code"


def show_figures(*numbers):
    """Show the figures the scripts just wrote to figures/."""
    for n in numbers:
        display(Image(filename=str(FIGURES_DIR / f"fig{{n}}.png")))


def show_tables(step):
    """Show the tables of one step, as the report prints them."""
    display(Markdown((RESULTS_DIR / f"{{step}}_tables.md").read_text()))''')


def notebook(cells):
    # nbformat 4.5 requires an id on every cell. Without one, any tool that opens the notebook writes a random id
    # of its own and the file changes on every rebuild; a positional id keeps two builds byte for byte identical.
    for position, cell in enumerate(cells, start=1):
        cell["id"] = f"cell-{position:03d}"
    return {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python",
                                                        "name": "python3"},
                                         "language_info": {"name": "python"}},
            "nbformat": 4, "nbformat_minor": 5}


def build_eda():
    cells = [md("# EDA — Stroke Prediction\n\n" + NOTICE + """

Stages 1 to 3 of the report — the target and the split, each feature, then the features against each other and
against the target — come from `eda.py`; the outliers, the pipeline checks and the dimensionality reduction come
from `reduction.py`."""), setup_cell("eda")]
    cells += module_cells("eda", "preprocessing", "the data, the split and the preprocessing pipeline")
    cells += module_cells("eda", "common", "paths, colors, figure saving and result collection")
    cells += module_cells("eda", "eda", "stages 1 to 3: Tables 2–12 and Figures 1–8")
    cells += [md("""# Run — stages 1 to 3

`eda.main()`, one stage per cell, with the figures of each stage shown as they are written."""),
              code('''res = Results("eda")
raw = load_raw()
assert raw.shape == (5110, 12) and raw[ID_COLUMN].is_unique
res.metrics["seed"] = SEED
res.metrics["features"] = FEATURES
raw.head()'''),
              md("### Stage 1 — the target, the split and data quality (Table 2–4, Figure 1)"),
              code("train = stage1(raw, res)\nshow_figures(1)"),
              md("### Stage 2 — the features one by one (Figures 2–3)"),
              code("stage2(train, res)\nshow_figures(2, 3)"),
              md("### Stage 3 — the features against each other and against the target (Figures 4–8)"),
              code("stage3(train, res)\nshow_figures(4, 5, 6, 7, 8)"),
              code('res.write()\nprint(f"Stages 1-3 done: {len(res.tables)} tables, Figures 1-8 written.")'),
              md("### The tables of stages 1 to 3"), code('show_tables("eda")')]
    cells += module_cells("eda", "reduction", "outliers, the pipeline checks and the dimensionality reduction")
    cells += [md("""# Run — outliers, pipeline and reduction

`reduction.main()`: Tables 13–20 and Figures 9–13."""),
              code("reduction_main()\nshow_figures(9, 10, 11, 12, 13)"),
              md("### The tables of this stage"), code('show_tables("reduction")')]
    return notebook(cells)


def build_classification():
    cells = [md("# Classification — Stroke Prediction\n\n" + NOTICE + """

One section per step of the report. Steps 0 to 4 use only the training rows and their validation folds; steps 5
and 6 are the only ones that receive the test rows."""), setup_cell("classification")]
    cells += module_cells("classification", "preprocessing", "the EDA's data, split and pipeline, reused unchanged",
                          source_project="eda")
    cells += module_cells("classification", "common", "the settings fixed in step 0, the folds and the helpers")
    cells += module_cells("classification", "network", "the network and its training loop")
    cells += module_cells("classification", "validation", "the cross-validation every model shares")
    cells += [md("""# The data and the validation folds

The first lines of `main()`: the EDA's split, checked row by row, and the 5 stratified folds of the training
rows."""),
              code("""X_train, X_test, y_train, y_test = load_data()
cv_folds = folds(y_train)
print(f"train {len(y_train)} rows ({int(y_train.sum())} strokes) · test {len(y_test)} rows "
      f"({int(y_test.sum())} strokes)")""")]
    cells += module_cells("classification", "baselines", "step 1: the trivial, linear and boosting baselines; "
                          "step 0 compares them too, so they are defined first")
    cells += module_cells("classification", "starting_point", "step 0: the BMI-missing indicator")
    cells += [md("## Run — step 0"),
              code("""make_preprocessor = starting_point_run(X_train, y_train, cv_folds)
n_inputs = make_preprocessor().fit_transform(X_train).shape[1]
mats = fold_matrices(X_train, y_train, cv_folds, make_preprocessor)
print(f"the pipeline outputs {n_inputs} columns")"""),
              code('show_tables("starting_point")')]
    cells += [md("## Run — step 1\n\nThe baselines of `code/baselines.py`, defined above."),
              code("oof_baselines, best_baseline = baselines_run(mats, y_train)"),
              code('show_tables("baselines")')]
    cells += module_cells("classification", "search", "steps 2 to 4: the network, its training and the search")
    cells += [md("""## Run — steps 2 to 4

The regularization comparison and the 48-configuration search; this is the slow cell (a few minutes on a
laptop CPU)."""),
              code("chosen = search_run(y_train, mats, n_inputs)\n"
                   "show_figures(1, 2)"),
              code('show_tables("search")')]
    cells += module_cells("classification", "evaluate", "step 5: the threshold, the test set and the seeds")
    cells += [md("""## Run — step 5

The first and only time the test rows are read."""),
              code("final = evaluate_run(X_train, X_test, y_train, y_test, make_preprocessor, chosen,\n"
                   "                     oof_baselines, best_baseline)\nshow_figures(3, 4, 5)"),
              code('show_tables("evaluation")')]
    cells += module_cells("classification", "errors", "step 6: the errors by subgroup and permutation importance")
    cells += [md("## Run — step 6"),
              code("errors_run(X_test, y_test, final)\nshow_figures(6, 7)"),
              code('show_tables("errors")'),
              md("## The split sizes the report quotes\n\nThe last lines of `main()`."),
              code('''import json

(RESULTS_DIR / "split_sizes.json").write_text(json.dumps({
    "train": len(y_train), "train_positives": int(y_train.sum()), "test": len(y_test),
    "test_positives": int(y_test.sum()), "validation_folds": [len(v) for _, v in cv_folds],
    "validation_positives": [int(y_train[v].sum()) for _, v in cv_folds]}, indent=2) + "\\n")
print((RESULTS_DIR / "split_sizes.json").read_text())''')]
    return notebook(cells)


def main():
    global DEFINED
    builders = (("eda", build_eda), ("classification", build_classification))
    for name, build in builders:
        DEFINED, EMITTED[:] = {}, []
        nb = build()
        path = PROJECTS / name / f"{name}.ipynb"
        path.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n")
        print(f"{path.relative_to(REPO)}: {len(nb['cells'])} cells")


if __name__ == "__main__":
    main()
