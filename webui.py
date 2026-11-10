import tomllib

import numpy as np
import pandas as pd
import streamlit as st
from scipy.integrate import solve_ivp

from model import seirp_ode


def load_scenarios(filepath: str = "scenarios.toml"):
    with open(filepath, "rb") as f:
        data = tomllib.load(f)
    return data["scenarios"]


SCENARIOS = load_scenarios()


def apply_preset():
    selected = st.session_state["scenario_choice"]
    if selected != "Custom":
        preset_values = SCENARIOS[selected]
        for key, val in preset_values.items():
            st.session_state[key] = val


if "beta" not in st.session_state:
    for key, val in SCENARIOS["Baseline"].items():
        st.session_state[key] = val


st.set_page_config(page_title="SEIRP Misinformation Model", layout="wide")
st.title("SEIRP Misinformation Model Simulation")


st.sidebar.header("Scenario Selection")
st.sidebar.selectbox(
    "Preset Scenario:",
    options=list(SCENARIOS.keys()) + ["Custom"],
    key="scenario_choice",
    on_change=apply_preset,
)

# Epidimiological Parameters
st.sidebar.header("Simulation Settings")
N = st.sidebar.number_input(
    "Population ($N$)",
    min_value=100,
    max_value=1_000_000,
    step=100,
    key="N",
    value=5000,
)
days = st.sidebar.number_input(
    "Simulation Duration in Days ($t$)",
    min_value=1,
    max_value=365,
    step=1,
    key="days",
    value=30,
)

# Epidemiological Parameters
st.sidebar.header("Epidemiological Parameters")
beta = st.sidebar.slider(
    r"Misinformation Transmission Rate ($\beta$)",
    min_value=0.00,
    max_value=2.00,
    step=0.01,
    key="beta",
)
kappa = st.sidebar.slider(
    r"Exposed → Active Spreader Rate ($\kappa$)",
    min_value=0.00,
    max_value=1.00,
    step=0.01,
    key="kappa",
)
xi = st.sidebar.slider(
    r"Exposed Self-Correction Rate ($\xi$)",
    min_value=0.00,
    max_value=1.00,
    step=0.01,
    key="xi",
)
omega = st.sidebar.slider(
    r"Prebunking Immunity Loss Rate ($\omega$)",
    min_value=0.00,
    max_value=1.00,
    step=0.01,
    key="omega",
)

# Interventions
st.sidebar.header("Interventions")
u = st.sidebar.slider(
    r"Prebunking Rate ($u$)",
    min_value=0.00,
    max_value=1.00,
    step=0.01,
    key="u",
)
epsilon = st.sidebar.slider(
    r"Prebunking Effectiveness ($\epsilon$)",
    min_value=0.00,
    max_value=1.00,
    step=0.01,
    key="epsilon",
)
gamma = st.sidebar.slider(
    r"Fact-checking Correction Rate ($\gamma$)",
    min_value=0.01,
    max_value=1.00,
    step=0.01,
    key="gamma",
)
rho = st.sidebar.slider(
    r"Re-exposure Susceptibility ($\rho$)",
    min_value=0.00,
    max_value=1.00,
    step=0.01,
    key="rho",
)

# Initial Conditions
st.sidebar.header("Initial Conditions")
I0 = st.sidebar.number_input(
    r"Active Spreaders ($I_0$)", min_value=0, max_value=int(N), value=100, step=1
)
E0 = st.sidebar.number_input(
    r"Exposed ($E_0$)", min_value=0, max_value=int(N), value=50, step=1
)
P0 = st.sidebar.number_input(
    r"Prebunked ($P_0$)", min_value=0, max_value=int(N), value=10, step=1
)
R_init = st.sidebar.number_input(
    r"Corrected ($R_{init}$)", min_value=0, max_value=int(N), value=0, step=1
)

# Calculated initial susceptible population
S0 = int(N - (E0 + I0 + P0 + R_init))

if S0 < 0:
    st.sidebar.error("Initial conditions compartments exceeds total population $N$")
    st.stop()

st.sidebar.number_input(
    r"Susceptible ($S_0$)", value=S0, disabled=True, key="S0_display"
)


y0 = [S0, E0, I0, R_init, P0]
t_span = (0, days)
t_eval = np.linspace(0, days, days)

sol = solve_ivp(
    fun=seirp_ode,
    y0=y0,
    t_span=t_span,
    t_eval=t_eval,
    args=(beta, kappa, xi, omega, u, gamma, epsilon, rho),
    method="RK45",
)

S, E, I, R, P = sol.y

R_0 = (beta * kappa / (gamma * (kappa + xi))) * (
    (omega / (u + omega)) + (1.0 - epsilon) * (u / (u + omega))
)

s_max = int(np.max(S))
e_max = int(np.max(E))
i_max = int(np.max(I))
r_max = int(np.max(R))
p_max = int(np.max(P))

r_final = int(R[-1])
s_final = int(S[-1])
e_final = int(E[-1])
i_final = int(I[-1])
r_final = int(R[-1])
p_final = int(P[-1])

s_peak = np.argmax(S)
s_peak_day = sol.t[s_peak]
e_peak = np.argmax(E)
e_peak_day = sol.t[e_peak]
i_peak = np.argmax(I)
i_peak_day = sol.t[i_peak]
r_peak = np.argmax(R)
r_peak_day = sol.t[r_peak]
p_peak = np.argmax(P)
p_peak_day = sol.t[p_peak]

st.metric(
    "Reproduction rate ($R_0$)",
    f"{R_0:.2f}",
    delta=("Contained" if R_0 <= 1.0 else "Outbreak"),
    delta_arrow="off",
    delta_color=("green" if R_0 <= 1.0 else "red"),
)

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric(
    "$S_{max}$",
    f"{s_max}",
    delta=f"Day: {s_peak_day:.2f}",
    delta_arrow="off",
    delta_color="off",
)
c2.metric(
    "$E_{max}$",
    f"{e_max}",
    delta=f"Day: {e_peak_day:.2f}",
    delta_arrow="off",
    delta_color="off",
)
c3.metric(
    "$I_{max}$",
    f"{i_max}",
    delta=f"Day: {i_peak_day:.2f}",
    delta_arrow="off",
    delta_color="off",
)
c4.metric(
    "$R_{max}$",
    f"{r_max}",
    delta=f"Day: {r_peak_day:.2f}",
    delta_arrow="off",
    delta_color="off",
)
c5.metric(
    "$P_{max}$",
    f"{p_max}",
    delta=f"Day: {p_peak_day:.2f}",
    delta_arrow="off",
    delta_color="off",
)

c1.metric("$S_{final}$", f"{s_final}")
c2.metric("$E_{final}$", f"{e_final}")
c3.metric("$I_{final}$", f"{i_final}")
c4.metric("$R_{final}$", f"{r_final}")
c5.metric("$P_{final}$", f"{p_final}")

chart_data = pd.DataFrame(
    {
        "Susceptible (S)": S,
        "Exposed (E)": E,
        "Active Spreaders (I)": I,
        "Corrected (R)": R,
        "Prebunked (P)": P,
    },
    index=np.round(sol.t, 2),
)
chart_data.index.name = "Day"

st.line_chart(
    chart_data,
    color=["#2b5c8f", "#f39c12", "#c0392b", "#27ae60", "#8e44ad"],
    width="stretch",
)
