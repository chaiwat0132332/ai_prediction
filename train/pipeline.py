# ==========================================================
# Training Pipeline (Production-grade)
# รองรับ:
# - Linear Regression
# - LSTM
# - Learning curve
# - Time-series safe split
# ==========================================================

import numpy as np
import pandas as pd

from sklearn.metrics import r2_score, mean_squared_error

from src.models.factory import create_model


# ==========================================================
# Create sliding windows
# ==========================================================

def create_windows(series, lag):

    X = []
    y = []

    for i in range(len(series) - lag):

        X.append(series[i:i+lag])
        y.append(series[i+lag])

    return np.array(X), np.array(y)


# ==========================================================
# Time-series split (NO SHUFFLE)
# ==========================================================

def time_series_split(series, lag, test_ratio=0.2):

    split_index = int(len(series) * (1 - test_ratio))

    train_series = series[:split_index]
    test_series = series[split_index - lag:]

    return train_series, test_series


# ==========================================================
# Main training function
# ==========================================================

def run_training(
    df,
    target_col,
    model_type,
    lag,
    hidden_size=None,
    num_layers=None,
    dropout=None,
    epochs=None
):

    # ======================================================
    # Prepare data
    # ======================================================

    series = df[target_col].values.astype(np.float32)

    train_series, test_series = time_series_split(
        series,
        lag=lag,
        test_ratio=0.2
    )

    # ======================================================
    # Create model
    # ======================================================

    model = create_model(
        model_type,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
        epochs=epochs
    )

    # ======================================================
    # Train model
    # ======================================================

    learning_curve = model.fit(
        train_series,
        lag=lag
    )

    # ======================================================
    # Evaluate on test set
    # ======================================================

    X_test, y_test = create_windows(test_series, lag)

    test_pred = []

    for x in X_test:

        pred = model.predict(x.reshape(1, -1))[0]

        test_pred.append(pred)

    test_pred = np.array(test_pred)

    # ======================================================
    # Metrics
    # ======================================================

    r2 = r2_score(y_test, test_pred)

    mse = mean_squared_error(y_test, test_pred)

    # ======================================================
    # Artifact
    # ======================================================

    artifact = {

        "model": model,

        "config": {

            "model_type": model_type,
            "lag": lag,
            "hidden_size": hidden_size,
            "num_layers": num_layers,
            "dropout": dropout,
            "epochs": epochs,

            "train_length": len(train_series)

        },

        "learning_curve": learning_curve,

        "metrics": {

            "r2": float(r2),
            "mse": float(mse)

        },

        "test_true": y_test.tolist(),

        "test_pred": test_pred.tolist()

    }

    return artifact


# ==========================================================
# Forecast helper
# ==========================================================

def forecast_future(
    artifact,
    series,
    steps
):

    model = artifact["model"]

    lag = artifact["config"]["lag"]

    last_window = series[-lag:]

    future = model.forecast(
        last_window,
        steps
    )

    return future