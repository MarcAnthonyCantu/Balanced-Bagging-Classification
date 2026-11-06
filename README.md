# Imbalanced Classification on Census-Income (KDD)

Comparing three strategies for learning from heavily imbalanced data — Balanced
Bagging, Balanced Random Forest, and SMOTE oversampling — on the Census-Income
(KDD) dataset, where the positive class is only ~6% of cases. The evaluation
tracks **minority- and majority-class accuracy separately**, because overall
accuracy is misleading when one class dominates.

## Overview

On Census-Income (KDD), a model that predicts "majority class" for everyone
scores ~94% accuracy while being useless, so this project measures each class's
accuracy on its own and asks which imbalance handling method actually recovers
the rare class. The most interesting result is a cautionary one: a popular fix
(SMOTE oversampling) **hurt** minority-class performance here.

- **Data:** Census-Income (KDD) (UCI id 117), ~6% minority class.
- **Metric:** per class accuracy (minority `pmin`, majority `pmaj`) plus
  ROC/AUC, rather than raw accuracy.
- **Methods compared:** Balanced Bagging (Random Forest base), Balanced Random
  Forest, and SMOTE-NC oversampling.

## Method & findings

1. **Preprocessing** — Categorical features are ordinal-encoded and the target
   label-encoded; missing values are dropped. A class stratified validation set
   (10% of each class) is held out so the rare class is represented.
2. **Balanced Bagging** — A `BalancedBaggingClassifier` with Random Forest base
   estimators undersamples the majority class per bag. Each base estimator's
   minority/majority accuracy is plotted against the full ensemble's, showing how
   aggregation tightens performance. Tuned result: **majority 81.0% / minority
   90.7%**.
3. **Hyperparameter & ensemble-size tuning** — Base estimator parameters
   (`n_estimators`, `min_samples_leaf`, `max_depth`) and bagging size
   (51 → 301) are swept for the best minority/majority balance.
4. **Balanced Random Forest** — As a second approach, a
   `BalancedRandomForestClassifier` reaches **majority 81.6% / minority 90.3%**,
   essentially on par with Balanced Bagging. ROC/AUC is close as well:
   **AUC ≈ 0.94** for Balanced Random Forest vs. **≈ 0.93** for Balanced Bagging.
5. **SMOTE oversampling (the key finding)** — Synthetically oversampling the
   minority class with SMOTE-NC to a 0.5 ratio and retraining made minority-class
   accuracy *drop* to **78.3%** (majority rose to 87.6%); a plain Random Forest on
   the oversampled data fell further, to **65.9%** minority. Oversampling to a
   balanced ratio makes the model train on a distribution (0.5) that no longer
   matches the true test distribution (~0.06), a **train/test distribution
   mismatch** that costs exactly the performance oversampling was meant to add.

**Takeaway:** methods built *into* the learner for imbalance (Balanced Bagging,
Balanced Random Forest) outperformed bolting SMOTE on top; more aggressive
balancing is not automatically better once the train/test distribution gap is
accounted for.

## Repository contents

| File | Description |
|------|-------------|
| `code.ipynb` | Full analysis — preprocessing, the three methods, tuning, and ROC/AUC comparison |
| `Report.pdf` | Written report with figures (per-estimator scatter, ROC curves) and discussion |

## Tech stack

- **Python** — NumPy, pandas, matplotlib
- **scikit-learn** — `RandomForestClassifier`, model-selection and metrics
- **imbalanced-learn** — `BalancedBaggingClassifier`, `BalancedRandomForestClassifier`, `SMOTENC`
- **ucimlrepo** — dataset fetch
- Jupyter Notebook

## Running it

```bash
pip install numpy pandas matplotlib scikit-learn imbalanced-learn ucimlrepo jupyter
jupyter notebook code.ipynb
```

The dataset is fetched automatically at runtime via `ucimlrepo` (UCI id 117),
so no local data file is needed, an internet connection is required on first run.

## Background

This project was developed for a graduate course in mathematical machine
learning. The emphasis is on evaluating imbalanced classification
per class rather than in aggregate and on understanding why a standard
technique underperformed, rather than on a single headline score.
