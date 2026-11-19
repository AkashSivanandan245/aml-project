# Source Code

| File | Description |
|---|---|
| `research-paper-final.ipynb` | Complete experiment: data loading, EDA, cleaning, train/test split, baseline and SMOTE pipelines, hold-out evaluation, repeated stratified cross-validation |
| `requirements.txt` | Python packages needed to run the notebook |

## Run on Kaggle (recommended)
1. Open Kaggle, then File → Import Notebook, and select `research-paper-final.ipynb`.
2. Add Input → Datasets → `sztuanakurun/diabates`.
3. Run All. A full run takes about 1–2 minutes on a CPU.

## Run locally
```bash
pip install -r requirements.txt
jupyter notebook research-paper-final.ipynb
```
In the second cell, replace the Kaggle path with the copy in this repository:
```python
df = pd.read_csv("../Dataset/diabetes.csv")
```
and remove (or keep) the `kagglehub.dataset_download(...)` line.
