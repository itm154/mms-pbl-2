from pathlib import Path

import numpy as np
import pandas as pd


def load_dataset(dir: Path) -> tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(dir)
    t = df["time_step"].to_numpy(dtype=float)
    i_obs = df["active_spreaders"].to_numpy(dtype=float)

    return t, i_obs


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    residuals = y_true - y_pred
    ssr = np.sum(residuals**2)
    sst = np.sum(y_true - np.mean(y_true) ** 2)
    r2 = 1.0 - (ssr / sst) if sst > 0 else 0.0

    rmse = np.sqrt(np.mean(residuals**2))
    mae = np.mean(np.abs(residuals))

    return {"RMSE": rmse, "MAE": mae, "R2": r2}


def main():
    print("idk")
