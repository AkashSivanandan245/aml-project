# Dataset: Pima Indians Diabetes

| Property | Value |
|---|---|
| File | `diabetes.csv` |
| Records | 768 |
| Features | 8 numeric predictors |
| Target | `Outcome` (1 = diabetic, 0 = non-diabetic) |
| Class balance | 268 diabetic (34.9%) / 500 non-diabetic (65.1%) |
| Original source | National Institute of Diabetes and Digestive and Kidney Diseases |
| Kaggle source used | https://www.kaggle.com/datasets/sztuanakurun/diabates |

## Columns
| Column | Description |
|---|---|
| Pregnancies | Number of pregnancies |
| Glucose | Plasma glucose concentration (2-hour oral glucose tolerance test) |
| BloodPressure | Diastolic blood pressure (mm Hg) |
| SkinThickness | Triceps skin-fold thickness (mm) |
| Insulin | 2-hour serum insulin (µU/ml) |
| BMI | Body mass index (kg/m²) |
| DiabetesPedigreeFunction | Score of diabetes likelihood based on family history |
| Age | Age in years |
| Outcome | Class label |

## Data quality note
A value of 0 is physically impossible for Glucose, BloodPressure, SkinThickness, Insulin and BMI, so the notebook treats these zeros as missing values (NaN). They are then filled with the median, which is calculated on training data only.
