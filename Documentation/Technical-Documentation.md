# Technical Documentation

**Project:** Effect of SMOTE on Imbalanced Diabetes Prediction
**Course:** Applied Machine Learning Lab (CSE2512P), IILM University, Gurugram
**Team:** Akash Sivanandan (2481026), Dhananjay Kumar (2481033), Girish Khanna (2481150), Soumen Kumar Dutta (2481248)

The full write-up, with literature review, discussion and references, is in [`Research-Paper.pdf`](Research-Paper.pdf). This document covers the technical side: environment, data processing, pipeline design, evaluation and how to reproduce the results.

---

## 1. Problem Statement

In medical screening, missing a diabetic patient (a false negative) costs far more than a false alarm. The Pima Indians Diabetes dataset is imbalanced (34.9% positive), so classifiers trained on it lean towards the majority "non-diabetic" class and have low recall. This project tests whether SMOTE, applied correctly, improves minority-class detection across three common classifiers, and what it costs in precision.

## 2. Environment

| Item | Version |
|---|---|
| Python | 3.13 (Kaggle Notebooks) |
| scikit-learn | 1.6.1 |
| imbalanced-learn | 0.14.2 |
| pandas | 2.3.3 |
| NumPy | 2.1.3 |
| Matplotlib | 3.10.0 |
| Seaborn | 0.13.2 |

Install the packages with `pip install -r Source-Code/requirements.txt`.

## 3. Data Processing

| Step | Detail | Why |
|---|---|---|
| Load | 768 rows, 8 features, target `Outcome` | — |
| Duplicate / null check | No duplicates and no explicit nulls | — |
| Invalid zeros → NaN | Glucose, BloodPressure, SkinThickness, Insulin, BMI | A value of 0 is physically impossible for these measurements, so it really means "missing" |
| Train/test split | 80/20, stratified on `Outcome`, `random_state=42` | Keeps the class ratio the same in both sets |
| Outlier capping (hold-out experiment) | IQR rule (Q1 − 1.5·IQR, Q3 + 1.5·IQR) on Glucose, BloodPressure, SkinThickness, Insulin, BMI and DiabetesPedigreeFunction. Caps are computed on the training set and then applied to both sets | Stops extreme values from dominating, without leaking test data |
| Imputation | `SimpleImputer(strategy="median")`, inside the pipeline | Fitted on training folds only |
| Scaling | `StandardScaler`, inside the pipeline | Logistic Regression is sensitive to feature scale |

## 4. Pipeline Design

Each of the six models is a single pipeline object, so every preprocessing step is fitted again on the training data of each split:

```
Baseline:  ColumnTransformer(median impute → scale) → Classifier
SMOTE:     ColumnTransformer(median impute → scale) → SMOTE(random_state=42) → Classifier
```

- Baseline pipelines use `sklearn.pipeline.Pipeline`.
- SMOTE pipelines use `imblearn.pipeline.Pipeline`. During `fit` it calls `SMOTE.fit_resample`, but it skips SMOTE during `predict`. Synthetic samples are therefore created only from training data, and every test fold keeps its real class distribution.

| Classifier | Settings |
|---|---|
| Logistic Regression | `max_iter=1000`, `random_state=42` |
| Decision Tree | `random_state=42` (default depth) |
| Random Forest | `random_state=42` (default 100 trees) |

Diagrams: `Project-Design/system_architecture.png`, `workflow_flowchart.png`, `uml_pipeline_class_diagram.png`.

## 5. Evaluation Protocol

**Experiment A: Hold-out (notebook sections 6–8).** The six pipelines are trained on the 80% split and evaluated on the untouched 20% (154 rows, 54 diabetic). The notebook reports accuracy, precision, recall, F1 and ROC-AUC, plus a confusion matrix for each model. With so few positive cases these numbers are noisy, so they are used only for illustration.

**Experiment B: Repeated stratified cross-validation (notebook section 9, main results).** `RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=42)` gives 50 train/test runs per model. All six models see exactly the same folds. Results are reported as mean ± standard deviation. A second table gives the change caused by SMOTE (SMOTE mean − baseline mean).

**Metrics:** accuracy alone is misleading on imbalanced data, so the analysis focuses on recall (diabetic patients detected), precision, F1 (the balance between the two) and ROC-AUC (ranking ability, independent of threshold).

## 6. Results

| Model | Accuracy | Precision | Recall | F1-score | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.767 ± 0.029 | 0.713 ± 0.058 | 0.558 ± 0.067 | 0.624 ± 0.054 | 0.835 ± 0.027 |
| Decision Tree | 0.700 ± 0.038 | 0.571 ± 0.058 | 0.582 ± 0.072 | 0.574 ± 0.055 | 0.673 ± 0.041 |
| Random Forest | 0.763 ± 0.026 | 0.709 ± 0.056 | 0.548 ± 0.056 | 0.616 ± 0.046 | 0.840 ± 0.026 |
| Logistic Regression + SMOTE | 0.755 ± 0.035 | 0.633 ± 0.049 | 0.716 ± 0.063 | 0.670 ± 0.047 | 0.835 ± 0.028 |
| Decision Tree + SMOTE | 0.695 ± 0.032 | 0.559 ± 0.045 | 0.604 ± 0.058 | 0.579 ± 0.044 | 0.674 ± 0.035 |
| Random Forest + SMOTE | 0.757 ± 0.031 | 0.622 ± 0.040 | 0.779 ± 0.049 | 0.691 ± 0.037 | 0.835 ± 0.029 |

**Change caused by SMOTE (SMOTE mean − baseline mean):**

| Algorithm | Accuracy | Precision | Recall | F1-score | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | −0.012 | −0.081 | +0.157 | +0.046 | 0.000 |
| Decision Tree | −0.005 | −0.012 | +0.022 | +0.005 | +0.001 |
| Random Forest | −0.006 | −0.086 | +0.231 | +0.075 | −0.004 |

*(Values are copied from the notebook, which calculates them from unrounded means. That is why a few of them differ by 0.001 from subtracting the rounded figures in the table above.)*

**Interpretation**
- SMOTE improves recall substantially for Logistic Regression and Random Forest, so many more diabetic patients are correctly flagged.
- The cost is lower precision (more false alarms). F1 still rises, so the net effect is positive.
- ROC-AUC is essentially unchanged. SMOTE shifts where the models draw the line between classes more than it improves how well they rank patients.
- A single unpruned Decision Tree overfits and benefits least.

## 7. Reproducing the Results

1. `pip install -r Source-Code/requirements.txt`
2. Open `Source-Code/research-paper-final.ipynb`.
3. If running outside Kaggle, change the loading line in cell 2 to `df = pd.read_csv("../Dataset/diabetes.csv")`.
4. Run all cells. With seed 42 everywhere (split, models, SMOTE, CV), the numbers above are reproduced exactly on the same library versions.

## 8. Limitations and Future Work

- The dataset is small (768 rows) and comes from a single population (Pima women aged 21 and over), so the results may not generalise.
- Hyperparameters were left at their defaults so that only the effect of SMOTE was measured. Tuning could change the absolute numbers.
- Possible extensions: other resampling methods (Borderline-SMOTE, ADASYN, SMOTE-Tomek), cost-sensitive learning (`class_weight="balanced"`), threshold tuning, and gradient-boosting models.
