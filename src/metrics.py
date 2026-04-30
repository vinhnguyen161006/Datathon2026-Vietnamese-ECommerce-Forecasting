import numpy as np
import pandas as pd


def evaluate(y_true: np.ndarray | pd.Series,
             y_pred: np.ndarray | pd.Series) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    mae  = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    return {"MAE": mae, "RMSE": rmse, "R2": r2}


def print_metrics(metrics: dict[str, float], label: str = "") -> None:
    prefix = f"[{label}] " if label else ""
    print(f"{prefix}MAE={metrics['MAE']:,.0f}  RMSE={metrics['RMSE']:,.0f}  R²={metrics['R2']:.4f}")
