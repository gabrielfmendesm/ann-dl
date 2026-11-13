---
project: classification
dataset: https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset
team:
  - Gabriel Fernando Missaka Mendes
  - Eduardo Takei Yaginuma
  - Luca Santana Feltrin
ai_use: "Claude assisted with the code implementation and the drafting of this report; all results and analyses were reviewed and validated by the team."
---

# Classification — Stroke Prediction

**Gabriel Fernando Missaka Mendes · Eduardo Takei Yaginuma · Luca Santana Feltrin** — Insper, Artificial Neural Networks and Deep Learning, 2026.2

This report trains a neural network to predict the `stroke` label of the dataset explored in the [EDA](../eda/index.md), on the same task, the same split and the same preprocessing pipeline.

## 0. Starting point

**Main metric and success criterion, fixed before any model was trained** (constants `MAIN_METRIC`, `CRITERION_GAIN` and `COST_MISS` in `code/common.py`):

- **Main metric: average precision** (AP, the area under the precision–recall curve). Only 4.87% of the rows are positive, so accuracy rewards a model that never predicts a stroke, and ROC-AUC hides false alarms behind the large number of true negatives; AP is read against the 4.87% that a random ranking scores.
- **Success criterion: the network's mean test AP over 5 seeds is at least 15% higher than the test AP of the best baseline** (the baseline with the highest validation AP), **and the gap is larger than the network's standard deviation across the 5 seeds.**
- **Cost of the errors, for the decision threshold: a missed stroke costs as much as 20 false alarms.** A false alarm sends a patient to a follow-up consultation; a missed stroke withholds preventive care from someone who will have one.
