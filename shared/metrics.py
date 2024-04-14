import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
from typing import Dict
import numpy as np

from sktime.performance_metrics.forecasting import MeanAbsoluteScaledError

def evaluate_forecast(predictions: pd.Series, actuals: pd.Series, train = None) -> Dict[str, float]:
    """
    Evaluate time series forecast using various metrics.

    Parameters:
    - predictions (pd.Series): The predicted values.
    - actuals (pd.Series): The actual values.

    Returns:
    - Dict[str, float]: A dictionary containing various evaluation metrics.
    """
    metrics = {}

    # MSE
    metrics["MSE"] = mean_squared_error(actuals, predictions)

    # RMSE
    metrics["RMSE"] = np.sqrt(metrics["MSE"])

    # MAE
    metrics["MAE"] = mean_absolute_error(actuals, predictions)

    # MAPE
    metrics["MAPE"] = np.mean(np.abs((actuals - predictions) / actuals)) * 100

    # SMAPE
    metrics["SMAPE"] = round(
        np.mean(
            np.abs(predictions - actuals)
            / ((np.abs(predictions) + np.abs(actuals)) / 2)
        )
        * 100,
        2,
    )

    # MASE
    mase = MeanAbsoluteScaledError()

    metrics["MASE"] = mase(np.array(predictions), np.array(actuals), y_train=np.array(train))

    
    return metrics


def print_evaluation_metrics(predictions: pd.Series, actuals: pd.Series, train = None):
    """
    Print the evaluation metrics in a formatted manner.

    Parameters:
    - predictions (pd.Series): The predicted values.
    - actuals (pd.Series): The actual values.
    """
    metrics = evaluate_forecast(predictions, actuals, train)
    print("Evaluation Metrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.2f}")
