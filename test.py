from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error

from model import seirp_ode


def load_dataset(dir):
    df = pd.read_csv(dir)
    t = df["time_step"].to_numpy(dtype=float)
    i_obs = df["active_spreaders"].to_numpy(dtype=float)

    return t, i_obs


def simulate_i(t_eval, params, y0):
    beta, kappa, xi, omega, rho, u, epsilon, gamma = params
    sol = solve_ivp(
        fun=lambda t, y: seirp_ode(
            t, y, beta, kappa, xi, omega, rho, u, epsilon, gamma
        ),
        t_span=(float(t_eval[0]), float(t_eval[-1])),
        y0=y0,
        t_eval=t_eval,
        method="RK45",
        rtol=1e-6,
        atol=1e-9,
    )
    if not sol.success:
        return np.full_like(t_eval, np.nan, dtype=float)
    return sol.y[2]  # I only


def main():
    csv_dir = Path("data/sydneysiege.csv")
    t_hours, i_true = load_dataset(csv_dir)

    t_days = t_hours / 24.0
    split_idx = int(0.25 * len(t_days))
    t_train, y_train = t_days[:split_idx], i_true[:split_idx]
    t_test, y_test = t_days[split_idx:], i_true[split_idx:]

    N = np.max(i_true)
    i0 = float(y_train[0])
    y0_train = [N - i0, 0.0, i0, 0.0, 0.0]  # [S, E, I, R, P]
    # Parameter: [beta, kappa, xi, omega, rho, u, epsilon, gamma]
    p0 = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    lower_bounds = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    upper_bounds = [100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 1.0, 100.0]

    def residuals_fn(params):
        i_pred = simulate_i(t_train, params, y0_train)
        if np.any(~np.isfinite(i_pred)):
            return np.full_like(y_train, 1e6)
        return i_pred - y_train

    res = least_squares(
        residuals_fn,
        x0=p0,
        bounds=(lower_bounds, upper_bounds),
        loss="soft_l1",
    )

    popt = res.x
    beta, kappa, xi, omega, rho, u, epsilon, gamma = popt
    sol_train_full = solve_ivp(
        fun=lambda t, y: seirp_ode(
            t, y, beta, kappa, xi, omega, rho, u, epsilon, gamma
        ),
        t_span=(float(t_train[0]), float(t_train[-1])),
        y0=y0_train,
        t_eval=t_train,
    )
    y0_test = sol_train_full.y[:, -1]
    i_test_pred = simulate_i(t_test, popt, y0_test)

    # Plot
    plt.figure(figsize=(9, 4))
    plt.scatter(t_hours[:split_idx], y_train, s=14, label="Train")
    plt.scatter(t_hours[split_idx:], y_test, s=14, color="red", label="Test")
    plt.plot(t_hours[:split_idx], sol_train_full.y[2], color="blue", label="Train fit")
    plt.plot(
        t_hours[split_idx:],
        i_test_pred,
        color="darkorange",
        linestyle="--",
        label="Test forecast",
    )
    plt.axvline(
        t_hours[split_idx], color="gray", linestyle=":", label="Train/Test split"
    )
    plt.title("Active Spreaders Forecast")
    plt.xlabel("Time (hours)")
    plt.ylabel("Active spreaders (I)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("test")

    # Metrics
    err = np.abs(y_test - i_test_pred)

    rmse_all = root_mean_squared_error(y_test, i_test_pred)
    mae_all = mean_absolute_error(y_test, i_test_pred)
    r2_all = r2_score(y_test, i_test_pred)

    q25, q75 = np.percentile(err, [25, 75])
    iqr = q75 - q25
    outlier_mask = err <= (q75 + 1.5 * iqr)

    rmse_clean = root_mean_squared_error(
        y_test[outlier_mask], i_test_pred[outlier_mask]
    )
    mae_clean = mean_absolute_error(y_test[outlier_mask], i_test_pred[outlier_mask])
    r2_clean = r2_score(y_test[outlier_mask], i_test_pred[outlier_mask])

    print(f"With outlier RMSE: {rmse_all:.2f}, MAE: {mae_all:.2f}, R2: {r2_all:.2f}")
    print(
        f"Without outlier RMSE: {rmse_clean:.2f}, MAE: {mae_clean:.2f}, R2: {r2_clean:.2f}"
    )
    print(f"Predicted parameters: {popt}")


if __name__ == "__main__":
    main()
