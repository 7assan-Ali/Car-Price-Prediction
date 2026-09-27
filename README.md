# Car Price Prediction

Regression pipeline for estimating used-car prices from tabular vehicle attributes.

## Models
Linear Regression · Random Forest Regressor · Gradient Boosting Regressor

## Metrics
MAE · RMSE · R²

## Data
Place a compatible CSV at `data/raw/dataset.csv`. The target should be named `price`, `selling_price`, `sellingprice`, or `msrp`.

No performance numbers are hard-coded; run the training script on your chosen dataset.

## Run
```bash
python -m venv .venv
.venv\\Scripts\\Activate.ps1
pip install -r requirements.txt
python src/train.py
```

## Methodology
Numeric features are imputed and scaled; categorical features are imputed and one-hot encoded inside a Pipeline. The split occurs before fitting preprocessing.

## Author
Hassan Ali — Computer Science student focused on Machine Learning and AI Engineering.
