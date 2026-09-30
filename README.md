# ANN & Deep Learning — Deliverables

Deliverables for the *Artificial Neural Networks and Deep Learning* course (Insper, 2026.2): the individual exercises and the team projects.

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

## Reproducing the results

Each report's code lives in `docs/exercises/<exercise>/code/` and rewrites the figures in `docs/exercises/<exercise>/figures/`. From the repository root:

``` shell
python docs/exercises/data/code/exercise1.py        # also exercise2.py and exercise3.py
python docs/exercises/perceptron/code/main.py
```

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
data/
  spaceship-titanic/train.csv   # Kaggle Spaceship Titanic training file (Data, Exercise 3)
```

## Data

`data/spaceship-titanic/train.csv` is the training file of the [Spaceship Titanic](https://www.kaggle.com/competitions/spaceship-titanic) Kaggle competition, committed so the script of the Data activity's Exercise 3 runs from a clean checkout.
