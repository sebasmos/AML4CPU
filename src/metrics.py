import time
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
import numpy as np

def calculate_metrics(y_test, y_pred):
    """
    Calculate various regression metrics given true and predicted values.

    Parameters:
    - y_test (array-like): True values.
    - y_pred (array-like): Predicted values.

    Returns:
    - metrics (dict): Dictionary containing calculated metrics.
    """
    print(y_test.shape, y_pred.shape)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred) # / max(y_test)  # Normalize MAE by maximum value of y_test
    mape = mean_absolute_percentage_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    smape = symmetric_mean_absolute_percentage_error(y_test, y_pred)
    y_naive = np.roll(y_test, 1)
    mase = mean_absolute_scaled_error(y_test, y_pred, y_naive)

    metrics = {
        # 'mse': mse,
        'mae': mae,
        # 'mape': mape,
        'rmse': rmse,
        'r2': r2,
        'smape': smape,
        'mase': mase,
    }

    return metrics

def symmetric_mean_absolute_percentage_error(y_true, y_pred):
    """
    Calculate the Symmetric Mean Absolute Percentage Error (SMAPE).

    Parameters:
    - y_true (array-like): True values.
    - y_pred (array-like): Predicted values.

    Returns:
    - smape (float): Symmetric Mean Absolute Percentage Error.
    """
    return 200 * np.mean(np.abs(y_pred - y_true) / (np.abs(y_pred) + np.abs(y_true)))

def mean_absolute_scaled_error(y_true, y_pred, y_naive):
    """
    Calculate the Mean Absolute Scaled Error (MASE).

    Parameters:
    - y_true (array-like): True values.
    - y_pred (array-like): Predicted values.
    - y_naive (array-like): Naive forecast values.

    Returns:
    - mase (float): Mean Absolute Scaled Error.
    """
    mae = np.mean(np.abs(y_true - y_pred))
    mae_naive = np.mean(np.abs(y_true - y_naive))

    mase = mae / mae_naive

    return mase
