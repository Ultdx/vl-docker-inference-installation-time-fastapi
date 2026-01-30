# Installation Time Prediction (ML Technical Demo)

This repository presents a machine learning technical demo focused on predicting installation time for VeryGames customer servers.

The colab project is designed to showcase data preprocessing, feature engineering and model training practices rather than maximizing raw predictive performance. 

---

## Dataset

The cleaned dataset is hosted on **noldo.fr**:

https://noldo.fr/dev/ml/vl-model-installation-time/installation_data_clean_v2.csv  
https://noldo.fr

It contains installation durations along with software, version, and add-on metadata.

---

## Data Preparation

Data exploration and cleaning are documented in:

```
colab/01_exploration_and_cleaning.ipynb
````

Key steps:
- Conversion of installation time from milliseconds to seconds
- Handling of missing categorical values
- Outlier detection using IQR
- Removal of non-representative software
- Generation of a clean, reusable dataset for modeling

---

## Modeling Approach

```
colab/02_training_and_evaluation.ipynb
````

The training script demonstrates:

- Feature engineering with categorical identities (`software`, `version`, `software_addon`)
- Preprocessing using `ColumnTransformer`
- One-hot encoding and feature scaling
- Interaction features on binary variables
- Incremental learning with `SGDRegressor` (`partial_fit`)
- Metric tracking across training epochs (R², RMSE, MAE, MSE)
- Model persistence with `joblib`

---

## Results

The model converges smoothly without overfitting, reaching a test R² of ~0.50.

This reflects:
- Meaningful signal in the data
- Expected limits of a linear model on a heterogeneous system

---

## Purpose

This project demonstrates:
- Practical data cleaning and validation
- End-to-end ML pipeline construction
- Thoughtful evaluation and interpretation of results

---

## Demo

You can try the inference, hosted on my VPS: 

```sh
curl -X POST "http://noldo.fr:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "software": "fs25",
    "version": "v-73986",
    "software_addon": "",
    "is_active_int": 1,
    "has_addon": 0
  }'
````
