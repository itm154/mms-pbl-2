import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp

import streamlit as st

st.set_page_config(page_title="SEIRP Misinformation Model", layout="wide")
st.title("SEIRP Misinformation Model Simulation")

# Epidimiological Parameters
st.sidebar.header("Epidimiological Parameters")
N = st.sidebar.number_input("Population ($N$)", min_value=100, value=5000)
days = st.sidebar.number_input("Days ($t$)", min_value=1, value=30)
beta = st.sidebar.slider(
    r"Misinformation Transmission Rate ($\beta$)", 0.00, 1.50, 0.50, 0.01
)
kappa = st.sidebar.slider(
    r"Exposed → Active Spreader Rate ($\kappa$)", 0.01, 1.00, 0.25, 0.01
)
xi = st.sidebar.slider(r"Exposed Self-Correction Rate ($\xi$)", 0.01, 1.00, 0.15, 0.01)
omega = st.sidebar.slider(
    r"Prebunking Immunity Loss Rate ($\omega$)", 0.01, 0.50, 0.04, 0.01
)

# Interventions
st.sidebar.header("Interventions")
u = st.sidebar.slider("Prebunking Campaign Rate ($u$)", 0.00, 0.50, 0.08, 0.01)
epsilon = st.sidebar.slider(r"Prebunking Efficacy ($\epsilon$)", 0.00, 1.00, 0.70, 0.01)
gamma = st.sidebar.slider(r"Correction Rate ($\gamma$)", 0.01, 1.00, 0.20, 0.01)
rho = st.sidebar.slider(r"Reexposure Susceptibility ($\rho$)", 0.00, 1.00, 0.20, 0.01)

st.sidebar.header("Initial conditions")
I0 = st.sidebar.number_input("Active spreaders ($I_0$)", 0, value=100)
E0 = st.sidebar.number_input("Exposed ($E_0$)", 0, value=50)
P0 = st.sidebar.number_input("Prebunked ($P_0$)", 0, value=10)
R_init = st.sidebar.number_input("Corrected ($R_{init}$)", 0, value=0)
S0 = N - (E0 + I0 + P0 + R_init)
st.sidebar.number_input("Susceptible ($S_0$)", S0, disabled=True)


def seirp_ode(t, y, beta, kappa, xi, omega, epsilon, rho):
    S, E, I, R, P = y[0], y[1], y[2], y[3], y[4]
    N = S + E + I + R + P

    # Individual rate of exposure
    lam = (beta * I) / N if N > 0 else 0.0

    # Susceptible population
    dS_dt = -lam * S - u * S + omega * P

    # Exposed population
    dE_dt = lam * S + (1.0 - epsilon) * lam * P + rho * lam * R - kappa * E - xi * E

    # Active spreaders
    dI_dt = kappa * E - gamma * I

    # Corrected population
    dR_dt = xi * E + gamma * I - rho * lam * R

    # Prebunked population
    dP_dt = u * S - (1.0 - epsilon) * lam * P - omega * P

    return [dS_dt, dE_dt, dI_dt, dR_dt, dP_dt]


y0 = [S0, E0, I0, R_init, P0]
t_span = (0, days)
t_eval = np.linspace(0, days, days)

sol = solve_ivp(
    fun=seirp_ode,
    y0=y0,
    t_span=t_span,
    t_eval=t_eval,
    args=(beta, kappa, xi, omega, epsilon, rho),
    method="RK45",
)

S, E, I, R, P = sol.y

R_0 = (beta * kappa / (gamma * (kappa * xi))) * (
    (omega / (u + omega)) + (1.0 - epsilon) * (u / (u + omega))
)
peak_spreaders = int(np.max(I))

col1, col2 = st.columns(2)
col1.metric("Reproduction Threshold ($R_0$)", f"{R_0:.2f}")
col2.metric("Peak Active Spreaders", f"{peak_spreaders:,}")

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
    width=850,
    height=500,
)
