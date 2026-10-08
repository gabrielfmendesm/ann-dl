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
python docs/projects/eda/code/main.py               # EDA project: Figures 1-13 and results/
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
    eda/                  # same layout: index.md, code/, figures/, plus results/
mkdocs.yml
requirements.txt
data/
  spaceship-titanic/train.csv   # Kaggle Spaceship Titanic training file (Data, Exercise 3)
  stroke-prediction/healthcare-dataset-stroke-data.csv   # Kaggle Stroke Prediction Dataset (EDA project)
```

## Data

`data/spaceship-titanic/train.csv` is the training file of the [Spaceship Titanic](https://www.kaggle.com/competitions/spaceship-titanic) Kaggle competition, committed so the script of the Data activity's Exercise 3 runs from a clean checkout.

`data/stroke-prediction/healthcare-dataset-stroke-data.csv` is the [Stroke Prediction Dataset](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset) (fedesoriano, Kaggle), committed so the EDA project runs from a clean checkout. SHA-256: `644d473b05d2797006bd94865e4f8bb057f0c721617911613c82c8fcfc707420`.
