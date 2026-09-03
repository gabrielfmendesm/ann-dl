# ANN & Deep Learning — Deliverables

Individual deliverables for the *Artificial Neural Networks and Deep Learning* course (Insper, 2026.2).

**Published site:** https://gabrielfmendesm.github.io/ann-dl

## Setup

Create and activate a virtual environment, then install the dependencies:

``` shell
python3 -m venv env
source ./env/bin/activate
python3 -m pip install -r requirements.txt --upgrade
```

## Local preview

The site is built with [MkDocs](https://www.mkdocs.org/) + Material. To preview locally:

``` shell
mkdocs serve -o
```

Every push to `main` publishes the site automatically via GitHub Actions (`mkdocs gh-deploy`).

## Layout

```
docs/
  index.md                # landing page
  exercises/
    data/
      index.md            # the report
      code/               # the sources actually run
      figures/            # the figures the report shows
    perceptron/
    mlp/
    vae/
  projects/
mkdocs.yml
requirements.txt
```
