# Effect of SMOTE and Cost-Sensitive Learning on Imbalanced Diabetes Prediction

Applied Machine Learning Lab (CSE2512P) project, IILM University, Gurugram.

This project measures how SMOTE oversampling changes the performance of Logistic Regression, Decision Tree and Random Forest classifiers on the Pima Indians Diabetes dataset. Models are compared with and without SMOTE on precision, recall, F1-score and ROC-AUC, using stratified 5-fold cross-validation repeated 10 times.

## Team

| Name | URN |
|---|---|
| Akash Sivanandan | 2481026 |
| Dhananjay Kumar | 2481033 |
| Girish Khanna | 2481150 |
| Soumen Kumar Dutta | 2481248 |

## Repository contents

| File | Description |
|---|---|
| `research-paper-final.ipynb` | Full experiment: EDA, preprocessing, models with and without SMOTE, cross-validation results |
| `requirements.txt` | Python packages needed to run the notebook |

## Dataset

Pima Indians Diabetes dataset (768 records, 8 features, 34.9% diabetic), originally from the National Institute of Diabetes and Digestive and Kidney Diseases.

Source: https://www.kaggle.com/datasets/sztuanakurun/diabates

The notebook downloads it automatically with `kagglehub`.

## Environment

- Python 3.13.15 (Kaggle Notebooks)
- scikit-learn 1.6.1, imbalanced-learn 0.14.2, numpy 2.1.3, pandas 2.3.3, matplotlib 3.10.0, seaborn 0.13.2

## How to reproduce

### Option 1: Kaggle (recommended)

1. Open a new Kaggle notebook and import `research-paper-final.ipynb` (File → Import Notebook).
2. Add the dataset `sztuanakurun/diabates` as an input (Add Input → Datasets).
3. Click **Run All**.

### Option 2: Local machine

1. Clone the repository and install the packages:

   ```bash
   git clone https://github.com/AkashSivanandan245/aml-project.git
   cd aml-project
   pip install -r requirements.txt
   ```

2. The notebook reads the data from the Kaggle path `/kaggle/input/...`. To run locally, change the loading line in the second cell to:

   ```python
   df = pd.read_csv(os.path.join(path, "diabetes.csv"))
   ```

   and add `import os` at the top. `kagglehub` may ask for your Kaggle API credentials the first time.

3. Start Jupyter and run all cells from top to bottom:

   ```bash
   jupyter notebook research-paper-final.ipynb
   ```

## Where each result comes from

| Paper item | Notebook section |
|---|---|
| Fig. 1 (class counts), Fig. 2 (correlation heatmap) | Section 2: Univariate EDA |
| Fig. 5 (confusion matrices) | Sections 6–7: baseline and SMOTE models |
| Table IV (cross-validation results) | Section 9: repeated stratified cross-validation |
| Table V (change caused by SMOTE) | Section 9 |
| Fig. 4 (recall box plot) | Section 9 |

## Reproducibility

- Random seed 42 is used for the train/test split, all models, SMOTE and cross-validation.
- SMOTE, imputation and scaling are fitted on training data only, inside an `imblearn` pipeline, so the test data is never used during training.
- A full run takes about 1–2 minutes on a standard CPU.
