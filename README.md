# Carrier Pay Prediction (Super Dispatch-style)

This project provides a reproducible ML pipeline (in a Jupyter Notebook) to predict `Carrier_Pay` from logistics quote data. It follows best practices: cleaning, feature engineering with `ColumnTransformer` and `FunctionTransformer`, model comparison (Linear Regression, Random Forest, Gradient Boosting, XGBoost), hyperparameter tuning, and model persistence with `joblib`.

## Expected Columns

- Quote_Id (dropped)
- Customer
- Vehicle_Types
- Mode(no HH)
- Total_Vehicles
- Origin_City
- Destination_City
- Origin_State
- Destination_State
- Origin_Zip
- Destination_Zip
- Miles
- Tariff
- GP
- Carrier_Pay (target)

## Setup

1. (Optional) Create a virtual environment
   - Windows (PowerShell):
     ```powershell
     py -m venv .venv
     .\.venv\Scripts\Activate.ps1
     ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Data

- Place your CSV at `data/shipments.csv` (create the `data/` folder if needed), or edit `DATA_CSV_PATH` at the top of the notebook to point to your file.
- The notebook will auto-load a small in-memory sample if the CSV is not found so you can run end-to-end immediately.
- Cleaning removes rows with non-positive or missing `Miles` and missing `Carrier_Pay`.

## Run

1. Start Jupyter and open the notebook:
   ```bash
   jupyter notebook
   ```
2. Open `carrier_pay_model.ipynb` and Run All Cells.

## What the Notebook Does

- Cleans data and enforces types (ZIPs as strings, numerics converted).
- Feature engineering via `FunctionTransformer`:
  - `tariff_per_mile`, `gp_per_mile`, `is_same_state`.
- Preprocessing via `ColumnTransformer`:
  - One-hot encode categoricals.
  - Impute + scale numerics.
- Builds pipelines for: LinearRegression, RandomForestRegressor, GradientBoostingRegressor, XGBRegressor (if installed).
- Cross-validated metrics: RMSE, R².
- Hyperparameter tuning with `RandomizedSearchCV` on the best base model (by RMSE).
- Saves final model and metrics:
  - `models/carrier_pay_model.joblib`
  - `models/carrier_pay_model_metrics.json`

## Using the Saved Model

```python
import joblib
import pandas as pd

pipe = joblib.load('models/carrier_pay_model.joblib')
# df_new should contain the same input columns used by the pipeline (categoricals + numeric base).
preds = pipe.predict(df_new)
```

## Notes

- If you see a NameError for `display`, add this cell once at the top of the notebook:
  ```python
  from IPython.display import display
  ```
- XGBoost is optional. If not installed, the notebook will skip it automatically.
- Adjust `get_search_space()` and the models list to suit your tuning needs.
