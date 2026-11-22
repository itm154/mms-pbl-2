import re
import tomllib
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter
from scipy.integrate import solve_ivp

from model import calculate_r0, seirp_ode

# Default values
N = 628600
DAYS = 90
I0 = 50
E0 = 0
P0 = 0
R_INITIAL = 0
S0 = N - (E0 + I0 + P0 + R_INITIAL)

# Output directory
OUTPUT_DIR = Path("plots")


def sanitize_filename(name: str) -> str:
    """Sanitize string for safe filenames across platforms."""
    return re.sub(r'[\\/*?:"<>|]', "", name).strip()


def load_scenarios(filepath="scenarios.toml"):
    """Load epidemiological and intervention scenario presets from TOML."""
    scenario_path = Path(__file__).with_name(filepath)
    with scenario_path.open("rb") as file:
        data = tomllib.load(file)
    epi = data.get("epidemiological", {})
    interventions = data.get("interventions", {})
    return epi, interventions


def plot_scenario(epi_name: str, int_name: str, epi_params: dict, int_params: dict):
    """Simulate and plot a single epidemiological x intervention scenario combination with metrics table"""
    y0 = [S0, E0, I0, R_INITIAL, P0]
    t_span = (0, DAYS)
    t_eval = np.linspace(0, DAYS, DAYS + 1)

    beta = epi_params["beta"]
    kappa = epi_params["kappa"]
    xi = epi_params["xi"]
    omega = epi_params["omega"]
    rho = epi_params["rho"]

    u = int_params["u"]
    epsilon = int_params["epsilon"]
    gamma = int_params["gamma"]

    r0_val = calculate_r0(beta, kappa, xi, omega, u, epsilon, gamma)
    r0_text = f"{r0_val:.2f}" if (r0_val is not None and np.isfinite(r0_val)) else "N/A"

    sol = solve_ivp(
        fun=seirp_ode,
        y0=y0,
        t_span=t_span,
        t_eval=t_eval,
        args=(beta, kappa, xi, omega, rho, u, epsilon, gamma),
        method="RK45",
        rtol=1e-6,
        atol=1e-8,
    )

    if not sol.success:
        raise RuntimeError(
            f"Simulation failed for {epi_name} & {int_name}: {sol.message}"
        )

    s, e, i, r, p = sol.y
    t = sol.t

    compartments = {
        "Susceptible (S)": (s, "#1f77b4"),
        "Exposed (E)": (e, "#ff7f0e"),
        "Active Spreaders (I)": (i, "#d62728"),
        "Corrected (R)": (r, "#2ca02c"),
        "Prebunked (P)": (p, "#9467bd"),
    }

    fig, (ax, ax_tab) = plt.subplots(
        2,
        1,
        figsize=(13, 7.5),
        gridspec_kw={"height_ratios": [3.8, 1.2]},
    )

    # Plot population curves
    for name, (vals, color) in compartments.items():
        ax.plot(t, vals, label=name, color=color, lw=2)

    ax.set_title(
        f"{epi_name} & {int_name} ($R_0 = {r0_text}$)",
        fontsize=13,
        fontweight="bold",
    )
    ax.set_xlabel("Day", fontsize=11)
    ax.set_ylabel("Population", fontsize=11)
    ax.set_xlim(0, DAYS)
    ax.set_ylim(0, N * 1.05)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda val, _: f"{int(val):,}"))
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(loc="best", framealpha=0.9)

    # Prepare table metrics matching webui.py
    col_labels = list(compartments.keys())
    row_labels = ["Maximum", "Final"]

    max_row = []
    final_row = []
    for vals, _ in compartments.values():
        peak_idx = int(np.argmax(vals))
        peak_val = vals[peak_idx]
        peak_day = t[peak_idx]
        max_row.append(f"{peak_val:,.0f}\n(Day {peak_day:.0f})")
        final_row.append(f"{vals[-1]:,.0f}")

    cell_text = [max_row, final_row]

    # Render table underneath plot
    ax_tab.axis("off")
    table = ax_tab.table(
        cellText=cell_text,
        rowLabels=row_labels,
        colLabels=col_labels,
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 3)

    # Header and row label styling
    for (row_idx, col_idx), cell in table.get_celld().items():
        if row_idx == 0 or col_idx == -1:
            cell.set_facecolor("#f0f2f6")
            cell.set_text_props(weight="bold")

    plt.tight_layout()

    filename = f"{sanitize_filename(epi_name)} - {sanitize_filename(int_name)}.png"
    output_path = OUTPUT_DIR / filename
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    epi_scenarios, int_scenarios = load_scenarios()

    total = len(epi_scenarios) * len(int_scenarios)
    print(f"Generating {total} scenario combination plots with summary tables...")

    count = 1
    for epi_name, epi_params in epi_scenarios.items():
        for int_name, int_params in int_scenarios.items():
            plot_scenario(epi_name, int_name, epi_params, int_params)
            print(f"[{count:02d}/{total}] Saved: {epi_name} - {int_name}.png")
            count += 1

    print(f"\nAll plots saved to {OUTPUT_DIR.resolve()}/")


if __name__ == "__main__":
    main()
