# Technical demonstration of training a machine learning model to predict installation time for VeryGames customer servers.
# The main goal is educational: exploring feature engineering, preprocessing and online learning with SGD. 
# Predictive performance could be improved by using other algorithms / neural nets.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import OneHotEncoder, StandardScaler, PolynomialFeatures
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score
)

import joblib


DATA_URL = (
    "https://noldo.fr/dev/ml/vl-model-installation-time/"
    "installation_data_clean_v2.csv"
)
RANDOM_STATE = 17
N_EPOCHS = 200
MODEL_PATH = "sgd_installation_time_model.joblib"


def load_and_prepare_data(url: str) -> pd.DataFrame:
    """Load dataset and perform minimal feature engineering."""
    df = pd.read_csv(url)

    df["has_addon"] = (df["software_addon"] != "not_defined").astype(int)
    df["is_active_int"] = df["is_active"].astype(int)

    return df


def build_preprocessor() -> ColumnTransformer:
    """Create preprocessing pipeline."""
    return ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(
                    drop="first",
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
                ["software", "version"],
            ),
            (
                "binary",
                Pipeline(
                    steps=[
                        (
                            "interactions",
                            PolynomialFeatures(
                                interaction_only=True,
                                include_bias=False,
                            ),
                        ),
                        ("scaler", StandardScaler()),
                    ]
                ),
                ["is_active_int", "has_addon"],
            ),
        ]
    )


def build_model() -> SGDRegressor:
    """Create SGD regressor with explicit hyperparameters."""
    #Should experiment with HistGradientBoostingRegressor or XGBoost for better result
    return SGDRegressor(
        loss="squared_error",
        penalty="elasticnet",
        alpha=0.01,
        l1_ratio=0.15,
        learning_rate="constant",
        eta0=0.00005,
        max_iter=1,   # required for manual partial_fit loop
        tol=None,
        random_state=RANDOM_STATE,
    )


def train_model(
    model: SGDRegressor,
    X_train: np.ndarray,
    y_train: pd.Series,
    X_test: np.ndarray,
    y_test: pd.Series,
):
    """Train model iteratively and track metrics."""
    history = {
        "train_r2": [],
        "test_r2": [],
        "train_rmse": [],
        "test_rmse": [],
        "train_mae": [],
        "test_mae": [],
        "train_mse": [],
        "test_mse": [],
    }

    for epoch in range(N_EPOCHS):
        model.partial_fit(X_train, y_train)

        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        history["train_r2"].append(r2_score(y_train, y_train_pred))
        history["test_r2"].append(r2_score(y_test, y_test_pred))

        history["train_rmse"].append(
            np.sqrt(mean_squared_error(y_train, y_train_pred))
        )
        history["test_rmse"].append(
            np.sqrt(mean_squared_error(y_test, y_test_pred))
        )

        history["train_mae"].append(
            mean_absolute_error(y_train, y_train_pred)
        )
        history["test_mae"].append(
            mean_absolute_error(y_test, y_test_pred)
        )

        history["train_mse"].append(
            mean_squared_error(y_train, y_train_pred)
        )
        history["test_mse"].append(
            mean_squared_error(y_test, y_test_pred)
        )

        if (epoch + 1) % 50 == 0:
            print(
                f"Epoch {epoch+1}/{N_EPOCHS} | "
                f"Train R2: {history['train_r2'][-1]:.3f} | "
                f"Test R2: {history['test_r2'][-1]:.3f}"
            )

    return history


def plot_training_history(history: dict):
    """Plot training metrics."""
    epochs = range(1, N_EPOCHS + 1)

    plt.figure(figsize=(15, 10))

    plt.subplot(2, 2, 1)
    plt.plot(epochs, history["train_r2"], label="Train R²")
    plt.plot(epochs, history["test_r2"], label="Test R²")
    plt.title("R² over epochs")
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 2, 2)
    plt.plot(epochs, history["train_rmse"], label="Train RMSE")
    plt.plot(epochs, history["test_rmse"], label="Test RMSE")
    plt.title("RMSE over epochs")
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 2, 3)
    plt.plot(epochs, history["train_mae"], label="Train MAE")
    plt.plot(epochs, history["test_mae"], label="Test MAE")
    plt.title("MAE over epochs")
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 2, 4)
    plt.plot(epochs, history["train_mse"], label="Train MSE")
    plt.plot(epochs, history["test_mse"], label="Test MSE")
    plt.title("MSE (loss) over epochs")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()


def run_example_predictions(model, preprocessor):
    """Run a few deterministic prediction examples."""
    examples = [
        {
            "software": "fs25",
            "version": "v-73986",
            "is_active_int": 1,
            "has_addon": 0,
            "label": "fs25 active, no add-on",
        },
        {
            "software": "fs25",
            "version": "v-73986",
            "is_active_int": 0,
            "has_addon": 0,
            "label": "fs25 inactive, no add-on",
        },
        {
            "software": "minecraft-forge",
            "version": "v-25454",
            "is_active_int": 1,
            "has_addon": 1,
            "label": "minecraft-forge active, with add-on",
        },
    ]

    for ex in examples:
        df_ex = pd.DataFrame([ex]).drop(columns="label")
        X_ex = preprocessor.transform(df_ex)
        pred = model.predict(X_ex)[0]
        print(f"{ex['label']} → {pred:.1f} seconds")


def main():
    df = load_and_prepare_data(DATA_URL)

    X = df[["software", "version", "is_active_int", "has_addon"]]
    y = df["installation_time_seconds"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.1,
        random_state=RANDOM_STATE,
    )

    preprocessor = build_preprocessor()
    X_train_p = preprocessor.fit_transform(X_train)
    X_test_p = preprocessor.transform(X_test)

    model = build_model()
    history = train_model(
        model, X_train_p, y_train, X_test_p, y_test
    )

    plot_training_history(history)

    joblib.dump(
        {"model": model, "preprocessor": preprocessor},
        MODEL_PATH,
    )

    print(f"Model saved to {MODEL_PATH}")

    run_example_predictions(model, preprocessor)


if __name__ == "__main__":
    main()
