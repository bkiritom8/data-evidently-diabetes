# Data Lab — Evidently AI Drift Detection on Diabetes Dataset

A Jupyter notebook that uses **Evidently AI** to detect data drift between a reference and production dataset using the **Diabetes dataset** from scikit-learn.

## What Changed from the Original

The original lab used the **Adult dataset** (fetched from OpenML), which requires an internet connection and has mixed numerical/categorical columns. This version uses the **Diabetes dataset** from `sklearn`, which is fully numerical and available offline.

| | Original | This Lab |
|---|---|---|
| Dataset | Adult (OpenML, 48842 rows, mixed types) | Diabetes (sklearn, 442 rows, all numerical) |
| Features | Numerical + Categorical | Numerical only |
| Source | `fetch_openml(name='adult')` | `load_diabetes()` |
| Drift simulation | Split by education level | BMI and BP scaling (+30% / +0.05) |

## Dataset

**Diabetes** from `sklearn.datasets.load_diabetes`

- **442 samples**, **10 numerical features**
- Features: `age`, `sex`, `bmi`, `bp`, `s1`, `s2`, `s3`, `s4`, `s5`, `s6`
- Target: disease progression score (continuous)

## Drift Simulation

The first 60% of data is the **reference** (training distribution).
The remaining 40% is treated as **production** with an injected shift:
- `bmi` scaled by ×1.3 — simulates an increase in patient BMI over time
- `bp` shifted by +0.05 — simulates a slight increase in blood pressure readings

## How to Run

### Step 1 — Install dependencies

```bash
pip install -r requirements.txt
```

### Step 2 — Launch Jupyter

```bash
jupyter notebook Lab1.ipynb
```

### Step 3 — Run all cells

The final cell saves the drift report as `diabetes_drift_report.html` in the same directory. Open it in any browser.

### Optional — Evidently Cloud

If you have an Evidently Cloud account, uncomment the last cell to push the report for persistent dashboards and alerts.

## Project Structure

```
Data_Lab/
├── Lab1.ipynb                  # Main notebook
├── requirements.txt
├── README.md
└── diabetes_drift_report.html  # Generated after running the notebook
```

## Dependencies

```
evidently
scikit-learn
pandas
numpy
jupyter
```
