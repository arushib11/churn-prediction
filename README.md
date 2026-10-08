# Beta Bank Customer Churn Prediction

Predicting which bank customers are likely to leave, on a dataset where only one customer in five does. The project compares four ways of handling that class imbalance and reaches the target on a held-out test set: **F1 = 0.596, AUC-ROC = 0.851**.

Built for the Supervised Learning sprint of the TripleTen AI/ML Bootcamp.

## The problem

Beta Bank is losing customers every month, and keeping a customer costs less than winning a new one. The task is to predict whether a customer will leave (`Exited`) from their profile and account behavior, with a target of **F1 ≥ 0.59** on the test set. AUC-ROC is tracked alongside F1.

## Data

10,000 customers, 14 columns (`Churn_supervised.csv`).

| Feature | Description |
|---------|-------------|
| CreditScore | Credit score |
| Geography | Country of residence (France, Germany, Spain) |
| Gender | Gender |
| Age | Age |
| Tenure | Years with the bank |
| Balance | Account balance |
| NumOfProducts | Number of bank products used |
| HasCrCard | Has a credit card |
| IsActiveMember | Active member flag |
| EstimatedSalary | Estimated salary |
| **Exited** | **Target: 1 = left, 0 = stayed** |

`RowNumber`, `CustomerId` and `Surname` are dropped as identifiers.

## Approach

1. **Missing values.** `Tenure` is missing for 909 customers. Their churn rate (20.1%) is almost the same as everyone else's (20.4%), so those rows were dropped, leaving 9,091.
2. **Class balance.** 7,237 customers stayed and 1,854 left: roughly 80/20.
3. **Encoding and split.** Categorical features encoded; data split 60/20/20 into train (5,454), validation (1,818) and test (1,819).
4. **Baseline.** A Random Forest with no imbalance handling.
5. **Imbalance techniques.** Class weighting, upsampling, downsampling and threshold adjustment, each tried with Random Forest and Logistic Regression.
6. **Final check.** The best configuration is evaluated once on the test set.

## Results on the validation set

| Approach | Model | F1 | AUC-ROC |
|---|---|---|---|
| Baseline, no imbalance handling | Random Forest | 0.496 | 0.847 |
| **Class weighting** | **Random Forest** | **0.595** | **0.844** |
| Class weighting | Logistic Regression | 0.478 | 0.760 |
| Upsampling | Random Forest | 0.509 | 0.840 |
| Upsampling | Logistic Regression | 0.399 | 0.762 |
| Downsampling | Random Forest | 0.560 | 0.841 |
| Downsampling | Logistic Regression | 0.481 | 0.762 |
| Threshold 0.30 | Logistic Regression | 0.449 | n/a |
| Class weighting + threshold search | Random Forest | 0.595 (best at 0.50) | 0.844 |

Random Forest settings throughout: 50 trees, max depth 8.

## Final model on the test set

Random Forest with `class_weight="balanced"`:

| | F1 | AUC-ROC |
|---|---|---|
| Validation | 0.595 | 0.844 |
| **Test** | **0.596** | **0.851** |

The target of F1 ≥ 0.59 is met, and the test score matches validation, so the result is not an artifact of tuning on the validation set.

## What I found

- **The baseline looked better than it was.** AUC-ROC was already 0.85 with no imbalance handling, but F1 was only 0.50. The model could rank customers well; it was the default cut-off that failed the minority class.
- **Class weighting gave the largest gain for the least effort:** F1 rose from 0.50 to 0.60 with one parameter.
- **Upsampling helped least.** Duplicating churners ten times raised F1 by only 0.01 for Random Forest, and it was the weakest of the three rebalancing methods for Logistic Regression.
- **Threshold search confirmed 0.50** as the best cut-off once classes were weighted: precision 0.56, recall 0.64. Lowering it to 0.30 catches 88% of churners at a precision of 0.36, which may be the better trade if a retention offer is cheap.

## What I would improve

- **Encoding.** I ordinal-encoded every column, including numeric ones. Tree models tolerate that, but it likely held Logistic Regression back. One-hot encoding the two categorical features and scaling the numeric ones would give it a fairer test.
- **Fit the encoder on training data only,** inside a scikit-learn `Pipeline`.
- **Stratify the splits** so each set keeps the same churn rate.
- **Try gradient boosting** and tune hyperparameters with cross-validation.

## Files

| File | Contents |
|---|---|
| `SupervisedLearning_notebook.ipynb` | Full analysis, with outputs |
| `Churn_supervised.csv` | Dataset |

## Run it

```bash
pip install pandas scikit-learn jupyter
jupyter notebook SupervisedLearning_notebook.ipynb
```

The notebook reads the data from `/datasets/Churn.csv`. To run it locally, change that path to `Churn_supervised.csv`.

## Tech

Python, pandas, scikit-learn (RandomForestClassifier, LogisticRegression, OrdinalEncoder), Jupyter.
