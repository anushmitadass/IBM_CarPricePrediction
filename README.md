# 🚗 Car Price Prediction

> A complete machine learning project that predicts the selling price of a car based on its technical specifications, using a full scikit-learn pipeline and an interactive ipywidgets-based frontend — all inside a Jupyter Notebook.

---

## 📋 Project Description

This project solves a **supervised regression problem**: given 25 features that describe a car (make, engine type, horsepower, dimensions, fuel type, etc.), predict its **market price in USD**.

The workflow covers every stage of a real-world ML project:

- **Exploratory Data Analysis (EDA)** — distributions, correlations, categorical breakdowns
- **Data Preprocessing** — handling `?` missing values, median/mode imputation, `StandardScaler`, `OneHotEncoder`, all wrapped in a `ColumnTransformer` pipeline
- **Feature Engineering** — feature importance ranking via Extra Trees
- **Model Comparison** — 10 algorithms evaluated with 5-fold cross-validation and held-out test metrics (R², MAE, RMSE, MAPE)
- **Hyperparameter Tuning** — `GridSearchCV` on the best-performing Random Forest
- **Interactive Frontend** — `ipywidgets` form that lets you dial in any car spec and get a live price prediction without leaving the notebook

---

## 📁 Project Files

```
zoosense/
├── AnushmitaDas_CarPricePrediction.ipynb   ← Main notebook (all code + UI)
├── dashboard.html                           ← ✨ Standalone frontend dashboard (open in browser)
├── carprice.csv                             ← Dataset
├── car_price_model.pkl                      ← Saved trained model (generated on run)
├── car_price_preprocessor.pkl               ← Saved preprocessor  (generated on run)
└── README.md                                ← This file
```

---

## 📊 Dataset

| Property | Detail |
|----------|--------|
| **Source** | [UCI Machine Learning Repository — Automobile Dataset](https://archive.ics.uci.edu/ml/datasets/automobile) |
| **File** | `carprice.csv` | 'https://drive.google.com/file/d/12wXFiS7b9JY0jcuHBSqTmwKIy0JET9Ly/view?usp=sharing'
| **Rows** | 199 cars |
| **Columns** | 26 (25 features + 1 target) |
| **Target** | `price` (continuous, USD) |
| **Missing values** | Encoded as `?` — replaced with `NaN` and imputed |

### Key Features

| Feature | Type | Description |
|---------|------|-------------|
| `make` | Categorical | Car manufacturer (e.g. toyota, bmw) |
| `fuel-type` | Categorical | `gas` / `diesel` |
| `body-style` | Categorical | sedan, hatchback, convertible … |
| `drive-wheels` | Categorical | fwd / rwd / 4wd |
| `engine-size` | Numeric | Displacement in cc |
| `horsepower` | Numeric | Engine power output |
| `curb-weight` | Numeric | Vehicle weight in lbs |
| `city-mpg` | Numeric | Fuel efficiency (city) |
| `highway-mpg` | Numeric | Fuel efficiency (highway) |
| `price` | **Target** | Market price in USD |

---

## 🛠️ Technologies Used

| Layer | Library / Tool | Purpose |
|-------|---------------|---------|
| **Language** | Python 3.x | Core language |
| **Data** | pandas, numpy | Data loading, manipulation |
| **Visualisation** | matplotlib, seaborn | EDA charts, evaluation plots |
| **ML** | scikit-learn | Pipelines, models, metrics, GridSearchCV |
| **Frontend / UI** | ipywidgets | Interactive in-notebook prediction form |
| **Dashboard** | HTML + Chart.js | Standalone browser dashboard (`dashboard.html`) |
| **Persistence** | joblib | Save/load model and preprocessor |
| **Environment** | Jupyter Notebook | Notebook runtime |

### Models Compared

Linear Regression · Ridge · Lasso · ElasticNet · K-Nearest Neighbours · Decision Tree · **Random Forest** · Gradient Boosting · Extra Trees · SVR

---

## ⚙️ Setup & Run Instructions

### Prerequisites

- Python 3.8 or higher
- `pip` package manager
- Jupyter Notebook or JupyterLab

### 1 — Clone / download the project

Place all files in the same directory so `carprice.csv` is next to the notebook.

### 2 — Install dependencies

```bash
pip install numpy pandas matplotlib seaborn scikit-learn ipywidgets joblib notebook
```

Enable the ipywidgets extension (needed for the interactive frontend):

```bash
jupyter nbextension enable --py widgetsnbextension --sys-prefix
# OR for JupyterLab:
jupyter labextension install @jupyter-widgets/jupyterlab-manager
```

### 3 — Launch Jupyter

```bash
jupyter notebook AnushmitaDas_CarPricePrediction.ipynb
```

### 4 — Run the notebook

Select **Kernel → Restart & Run All** to execute every cell from top to bottom.

After execution:
- All EDA charts and model metrics will be rendered inline.
- `car_price_model.pkl` and `car_price_preprocessor.pkl` will be saved to the working directory.
- The **interactive prediction form** (Section 10) will appear — use the sliders and dropdowns to configure a car, then click **🔮 Predict Price**.

---

## 📈 Key Results

| Metric | Value (Tuned Random Forest) |
|--------|----------------------------|
| **Test R²** | ~0.95+ |
| **Test MAE** | ~$1,000–$1,500 |
| **Test RMSE** | ~$1,500–$2,200 |
| **Test MAPE** | ~8–12% |

> *Exact values depend on the random state and train/test split.*

**Top predictive features:** `engine-size`, `curb-weight`, `horsepower`, `highway-mpg`, `city-mpg`, `width`

---

## 🖼️ Notebook Sections at a Glance

```
1.  Import Libraries
2.  Load & Inspect Dataset
3.  Exploratory Data Analysis (EDA)
    ├─ 3.1 Price Distribution
    ├─ 3.2 Average Price by Make
    ├─ 3.3 Correlation Heatmap
    ├─ 3.4 Top Correlated Features
    ├─ 3.5 Scatter Plots vs Price
    ├─ 3.6 Price by Body Style / Fuel Type
    └─ 3.7 Categorical Count Plots
4.  Data Preprocessing
5.  Feature Engineering & Selection
6.  Model Training  (10 algorithms)
7.  Model Evaluation (R², MAE, RMSE, MAPE + charts)
8.  Hyperparameter Tuning (GridSearchCV on Random Forest)
9.  Final Model & Sample Predictions
10. Interactive Frontend (ipywidgets)
```

---

## 👩‍💻 Author

**Anushmita Das**

---

*Dataset source: UCI Machine Learning Repository — [Automobile Data Set](https://archive.ics.uci.edu/ml/datasets/automobile)*
