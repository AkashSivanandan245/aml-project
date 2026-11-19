# Screenshots

All images are outputs of `Source-Code/research-paper-final.ipynb`.

| File | Shows | Paper item |
|---|---|---|
| `01_class_counts.png` | Outcome class counts (class imbalance) | Fig. 1 |
| `02_boxplot_<feature>.png` | Distribution of each feature by Outcome (8 plots) | EDA |
| `03_correlation_heatmap.png` | Correlation between features and Outcome | Fig. 2 |
| `04_confusion_matrix_<model>_baseline.png` | Hold-out confusion matrices without SMOTE | Fig. 5 |
| `05_confusion_matrix_<model>_SMOTE.png` | Hold-out confusion matrices with SMOTE | Fig. 5 |
| `06_recall_boxplot_cross_validation.png` | Recall across 50 cross-validation runs, all six models | Fig. 4 |

## Class imbalance
![](01_class_counts.png)

## Correlation heatmap
![](03_correlation_heatmap.png)

## Confusion matrices: Random Forest without vs with SMOTE
| Without SMOTE | With SMOTE |
|---|---|
| ![](04_confusion_matrix_RandomForest_baseline.png) | ![](05_confusion_matrix_RandomForest_SMOTE.png) |

## Recall across 50 cross-validation runs
![](06_recall_boxplot_cross_validation.png)
