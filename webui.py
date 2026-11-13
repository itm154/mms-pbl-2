import tomllib
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from scipy.integrate import solve_ivp

from model import calculate_r0, seirp_ode

st.set_page_config(
    page_title="SEIRP Misinformation Model",
    layout="wide",
)


def load_scenarios(filepath="scenarios.toml"):
    """Load epidemiological and intervention scenario presets from TOML."""
    scenario_path = Path(__file__).with_name(filepath)
    with scenario_path.open("rb") as file:
        data = tomllib.load(file)
    epi = data.get("epidemiological", {})
    interventions = data.get("interventions", {})
    return epi, interventions


def apply_epi_preset():
    """Copy the selected epidemiological preset's values into widget state."""
    selected = st.session_state["epi_choice"]

    if selected != "Custom" and selected in EPIDEMIOLOGICAL_SCENARIOS:
        for key, value in EPIDEMIOLOGICAL_SCENARIOS[selected].items():
            st.session_state[key] = value


def apply_intervention_preset():
    """Copy the selected intervention preset's values into widget state."""
    selected = st.session_state["intervention_choice"]

    if selected != "Custom" and selected in INTERVENTION_SCENARIOS:
        for key, value in INTERVENTION_SCENARIOS[selected].items():
            st.session_state[key] = value


# ---------------------------------------------------------------------
# Initialise session state before creating widgets.
# ---------------------------------------------------------------------
EPIDEMIOLOGICAL_SCENARIOS, INTERVENTION_SCENARIOS = load_scenarios()

default_epi = (
    "Baseline"
    if "Baseline" in EPIDEMIOLOGICAL_SCENARIOS
    else next(iter(EPIDEMIOLOGICAL_SCENARIOS), "Custom")
)
default_intervention = (
    "Baseline"
    if "Baseline" in INTERVENTION_SCENARIOS
    else next(iter(INTERVENTION_SCENARIOS), "Custom")
)

defaults = {
    "epi_choice": default_epi,
    "intervention_choice": default_intervention,
    "N": 628000,
    "days": 90,
    "I0": 50,
    "E0": 0,
    "P0": 0,
    "R_initial": 0,
    **EPIDEMIOLOGICAL_SCENARIOS.get(default_epi, {}),
    **INTERVENTION_SCENARIOS.get(default_intervention, {}),
}

for key, value in defaults.items():
    st.session_state.setdefault(key, value)


# ---------------------------------------------------------------------
# Page heading.
# ---------------------------------------------------------------------
st.title("SEIRP Misinformation Model Simulation")
st.caption(
    "Explore how misinformation spreads and how prebunking and "
    "fact-checking interventions affect the population."
)


# ---------------------------------------------------------------------
# Sidebar: scenario selection and general settings.
# ---------------------------------------------------------------------
st.sidebar.header("Scenario Selection")

st.sidebar.selectbox(
    "Epidemiological scenario",
    options=list(EPIDEMIOLOGICAL_SCENARIOS) + ["Custom"],
    key="epi_choice",
    on_change=apply_epi_preset,
)

st.sidebar.selectbox(
    "Intervention scenario",
    options=list(INTERVENTION_SCENARIOS) + ["Custom"],
    key="intervention_choice",
    on_change=apply_intervention_preset,
)

st.sidebar.header("Simulation Settings")

N = st.sidebar.number_input(
    "Total population ($N$)",
    min_value=100,
    step=1000,
    key="N",
)

days = st.sidebar.number_input(
    "Simulation duration ($t$, days)",
    min_value=1,
    max_value=365,
    step=1,
    key="days",
)


# ---------------------------------------------------------------------
# Sidebar: model parameters.
# ---------------------------------------------------------------------
st.sidebar.header("Epidemiological Parameters")

beta = st.sidebar.slider(
    r"Misinformation transmission rate ($\beta$)",
    min_value=0.00,
    max_value=2.00,
    step=0.01,
    key="beta",
)

kappa = st.sidebar.slider(
    r"Exposed $\rightarrow$ active spreader rate ($\kappa$)",
    min_value=0.00,
    max_value=3.00,
    step=0.01,
    key="kappa",
)

xi = st.sidebar.slider(
    r"Exposed self-correction rate ($\xi$)",
    min_value=0.00,
    max_value=1.00,
    step=0.01,
    key="xi",
)

omega = st.sidebar.slider(
    r"Prebunking immunity loss rate ($\omega$)",
    min_value=0.00,
    max_value=1.00,
    step=0.01,
    key="omega",
)

rho = st.sidebar.slider(
    r"Re-exposure susceptibility ($\rho$)",
    min_value=0.00,
    max_value=1.00,
    step=0.01,
    key="rho",
)


# ---------------------------------------------------------------------
# Sidebar: interventions.
# ---------------------------------------------------------------------
st.sidebar.header("Interventions")

u = st.sidebar.slider(
    r"Prebunking campaign rate ($u$)",
    min_value=0.00,
    max_value=1.00,
    step=0.01,
    key="u",
)

epsilon = st.sidebar.slider(
    r"Prebunking effectiveness ($\epsilon$)",
    min_value=0.00,
    max_value=1.00,
    step=0.01,
    key="epsilon",
)

gamma = st.sidebar.slider(
    r"Fact-checking correction rate ($\gamma$)",
    min_value=0.01,
    max_value=1.0,
    step=0.01,
    key="gamma",
)


# ---------------------------------------------------------------------
# Sidebar: Initial conditions.
# ---------------------------------------------------------------------
st.sidebar.header("Initial Conditions")

I0 = st.sidebar.number_input(
    r"Active spreaders ($I_0$)",
    min_value=0,
    max_value=N,
    step=1,
    key="I0",
)

E0 = st.sidebar.number_input(
    r"Exposed ($E_0$)",
    min_value=0,
    max_value=N,
    step=1,
    key="E0",
)

P0 = st.sidebar.number_input(
    r"Prebunked ($P_0$)",
    min_value=0,
    max_value=N,
    step=1,
    key="P0",
)

R_initial = st.sidebar.number_input(
    "Corrected ($R_{init}$)",
    min_value=0,
    max_value=N,
    step=1,
    key="R_initial",
)

S0 = N - (E0 + I0 + P0 + R_initial)

if S0 < 0:
    st.sidebar.error(
        "The initial compartment values exceed the total population. "
        r"Reduce $E_0$, $I_0$, $P_0$ or the initial corrected population."
    )
    st.stop()

st.sidebar.metric(
    r"Calculated susceptible population ($S_0$)",
    f"{S0:,}",
)


# ---------------------------------------------------------------------
# Calculate reproduction threshold and display interpretation.
# ---------------------------------------------------------------------
r0_value = calculate_r0(
    beta,
    kappa,
    xi,
    omega,
    u,
    epsilon,
    gamma,
)

st.subheader("Reproduction Number")

if r0_value is None or not np.isfinite(r0_value):
    st.metric(r"$R_0$", "N/A")
    st.warning("Reproduction threshold is undefined.")
    st.caption(
        r"Check that $\kappa + \xi$, $u + \omega$, and $\gamma$ are greater than zero."
    )

else:
    st.metric(r"$R_0$", f"{r0_value:.2f}")

    if np.isclose(r0_value, 1.0, atol=1e-9):
        st.warning("At Threshold")
        st.caption(
            "The reproduction threshold is approximately 1. "
            "The system is at the critical boundary between decline "
            "and potential growth."
        )

    elif r0_value < 1.0:
        st.success("Contained")
        st.caption(
            r"$R_0$ below 1 indicates that misinformation spread is expected "
            "to decline near the assumed disease-free equilibrium."
        )

    else:
        st.error("Outbreak Potential")
        st.caption(
            r"$R_0$ above 1 indicates that misinformation has the potential "
            "to spread and grow. It does not guarantee an outbreak."
        )

st.caption(
    "This reproduction threshold is calculated under the assumed "
    "disease-free equilibrium with R* = 0. It is a threshold indicator, "
    "not a guarantee of the exact simulated trajectory."
)


# ---------------------------------------------------------------------
# Solve the SEIRP ODE system.
# ---------------------------------------------------------------------
y0 = [S0, E0, I0, R_initial, P0]
t_span = (0, days)
t_eval = np.linspace(0, days, days + 1)

solution = solve_ivp(
    fun=seirp_ode,
    y0=y0,
    t_span=t_span,
    t_eval=t_eval,
    args=(beta, kappa, xi, omega, rho, u, epsilon, gamma),
    method="RK45",
    rtol=1e-6,
    atol=1e-8,
)

if not solution.success:
    st.error(f"The numerical solver could not complete: {solution.message}")
    st.stop()

S, E, I, R, P = solution.y


# ---------------------------------------------------------------------
# Peak population and peak day.
# ---------------------------------------------------------------------
compartments = {
    "Susceptible (S)": S,
    "Exposed (E)": E,
    "Active Spreaders (I)": I,
    "Corrected (R)": R,
    "Prebunked (P)": P,
}

st.subheader("Maximum Population and Peak Day")
peak_columns = st.columns(5)

for column, (name, values) in zip(peak_columns, compartments.items()):
    peak_index = np.argmax(values)
    column.metric(
        f"{name} maximum",
        f"{values[peak_index]:,.0f}",
        delta=f"Day {solution.t[peak_index]:.0f}",
        delta_color="off",
    )


# ---------------------------------------------------------------------
# Final population.
# ---------------------------------------------------------------------
st.subheader("Final Population")
final_columns = st.columns(5)

for column, (name, values) in zip(final_columns, compartments.items()):
    column.metric(
        name,
        f"{values[-1]:,.0f}",
    )


# ---------------------------------------------------------------------
# Plot all five compartments.
# ---------------------------------------------------------------------
st.subheader("SEIRP Population Trends")

chart_data = pd.DataFrame(
    compartments,
    index=solution.t,
)

chart_data.index.name = "Day"

st.line_chart(
    chart_data,
    x_label="Day",
    y_label="Population",
    width="stretch",
)

with st.expander("View simulation data"):
    st.dataframe(
        chart_data.round(4),
        width="stretch",
    )


# ---------------------------------------------------------------------
# Model assumptions.
# ---------------------------------------------------------------------
st.caption(
    "Model assumptions: closed population; no births or deaths; "
    "prebunking effectiveness reduces exposure among the P compartment; "
    r"corrected individuals may be re-exposed according to $\rho$."
)
