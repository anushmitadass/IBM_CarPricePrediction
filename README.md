# 🚗 Car Price Prediction

**Author:** Anushmita Das  
**Dataset:** UCI Automobile Dataset (`carprice.csv`)  
**Notebook:** `AnushmitaDas_CarPricePrediction.ipynb`  
**Script:** `AnushmitaDas_CarPricePrediction.py`

---

## Project Description

This project builds a complete end-to-end machine learning pipeline to predict the **market price of an automobile** based on its technical and categorical attributes. The problem is framed as a **supervised regression task**: given 25 input features describing a car, the model outputs a predicted selling price in USD.

The project covers the full data science lifecycle — raw data ingestion, exploratory analysis, preprocessing, feature engineering, multi-model comparison, hyperparameter tuning, and an interactive prediction interface — entirely within a single Jupyter Notebook.

---

## Dataset

| Property | Value |
|---|---|
| Source | [UCI Machine Learning Repository — Automobile Dataset](https://archive.ics.uci.edu/ml/datasets/automobile) |
| File | `carprice.csv` |
| Rows | 199 |
| Features | 25 input features + 1 target (`price`) |
| Era | Mid-1980s automobiles |
| Missing values | Encoded as `?` (replaced with NaN before modelling) |

---

## Technologies Used

| Category | Library / Tool |
|---|---|
| Language | Python 3.8+ |
| Data manipulation | pandas, NumPy |
| Visualisation | Matplotlib, Seaborn |
| Machine learning | scikit-learn |
| Interactive UI | ipywidgets |
| Model persistence | joblib |
| Notebook environment | Jupyter Notebook / JupyterLab |

---

## Project Structure

```
zoosense/
├── carprice.csv                          # Raw dataset
├── AnushmitaDas_CarPricePrediction.ipynb # Main Jupyter Notebook
├── AnushmitaDas_CarPricePrediction.py    # Python script version
├── AnushmitaDas_ProjectReport.docx       # Full project report
├── requirements.txt                      # Python dependencies
└── README.md                             # This file
```

> After running the notebook, two additional files are saved:
> - `car_price_model.pkl` — trained final model
> - `car_price_preprocessor.pkl` — fitted preprocessing pipeline

---

## Setup & Run Instructions

### 1. Clone / download the repository

```bash
git clone <repo-url>
cd zoosense
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Enable ipywidgets extension (for the interactive predictor)

```bash
jupyter nbextension enable --py widgetsnbextension --sys-prefix
```

### 4. Launch the notebook

```bash
jupyter notebook AnushmitaDas_CarPricePrediction.ipynb
```

### 5. Run all cells

Select **Kernel → Restart & Run All**.  
The interactive prediction form appears in **Section 10**.

### Alternative — run as a plain Python script

```bash
python AnushmitaDas_CarPricePrediction.py
```

> Note: the ipywidgets interactive frontend is only available in the notebook version.

---

## Pipeline Summary

| Step | Description |
|---|---|
| 1 — Load & Inspect | Read CSV, check shape/types/nulls, replace `?` with NaN |
| 2 — EDA | 8 visualisations: distributions, correlations, categorical breakdowns |
| 3 — Preprocessing | Median imputation + StandardScaler (numeric); Mode imputation + OneHotEncoder (categorical) via ColumnTransformer |
| 4 — Feature Engineering | ExtraTreesRegressor feature importance ranking (top 20) |
| 5 — Model Training | 10 algorithms trained with 5-fold cross-validation |
| 6 — Evaluation | R², RMSE, MAE, MAPE + Actual vs Predicted + Residual plots |
| 7 — Tuning | GridSearchCV on Random Forest (240 fits) |
| 8 — Persistence | Save model & preprocessor as `.pkl` files |
| 9 — Interactive UI | ipywidgets form for live predictions in-notebook |

---

## Models Trained

- Linear Regression
- Ridge Regression
- Lasso Regression
- ElasticNet
- K-Nearest Neighbours
- Decision Tree
- **Random Forest** ← best baseline
- Gradient Boosting
- Extra Trees
- Support Vector Regression (SVR)

---

## Key Results

| Metric | Value (Tuned Random Forest) |
|---|---|
| Test R² | ~0.95 |
| Test MAE | ~$1,000–$1,500 |
| Test RMSE | ~$1,500–$2,000 |
| Test MAPE | ~8–12% |

> Exact values vary slightly with each run due to the stochastic nature of Random Forest.

---

## Key Findings

- **Engine size**, **curb weight**, and **horsepower** are the strongest predictors (Pearson r > 0.80 with price).
- **City-MPG** and **highway-MPG** are strongly *negatively* correlated with price (r ≈ −0.70).
- **Make (brand)** is a powerful categorical predictor — BMW, Porsche, and Mercedes-Benz command significantly higher prices.
- Tree-based ensembles consistently outperform linear models on this dataset.
- GridSearchCV tuning improved CV R² by ~1–2% over the default Random Forest.

---

## References

1. [UCI Machine Learning Repository — Automobile Dataset](https://archive.ics.uci.edu/ml/datasets/automobile)
2. [scikit-learn Documentation](https://scikit-learn.org/stable/)
3. [pandas Documentation](https://pandas.pydata.org/docs/)
4. [Matplotlib Documentation](https://matplotlib.org/stable/contents.html)
5. [Seaborn Documentation](https://seaborn.pydata.org/)
6. [ipywidgets Documentation](https://ipywidgets.readthedocs.io/)
7. Breiman, L. (2001). Random Forests. *Machine Learning*, 45(1), 5–32.
8. Pedregosa et al. (2011). Scikit-learn: Machine Learning in Python. *JMLR* 12, 2825–2830.
