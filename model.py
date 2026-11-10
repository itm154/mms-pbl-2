def seirp_ode(t, y, beta, kappa, xi, omega, u, gamma, epsilon, rho):
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


# TODO: Plot each scenario in toml file
# # Parameters
# N = 10000
# beta = 0.50
# kappa = 0.25
# xi = 0.15
# omega = 0.05
# epsilon = 0.70
# rho = 0.20
# u = 0.20
# gamma = 0.20  # Fact-checking correction rate
#
# # Initial conditions
# I0 = 10  # Initial active spreaders
# E0 = 20  # Initial exposed
# P0 = 500  # Initial prebunked
# R_init = 0  # Initial corrected
# S0 = N - (E0 + I0 + P0 + R_init)  # Initial susceptible
#
# y0 = [S0, E0, I0, R_init, P0]
# t_span = (0, 30)
# t_eval = np.linspace(t_span[0], t_span[1], 600)
#
# sol = solve_ivp(
#     fun=seirp_ode,
#     y0=y0,
#     t_span=t_span,
#     t_eval=t_eval,
#     args=(beta, kappa, xi, omega, u, gamma, epsilon, rho),
#     method="RK45",
# )
#
# S, E, I, R, P = sol.y
#
# R_0 = (beta * kappa / (gamma * (kappa + xi))) * (
#     (omega / (u + omega)) + (1.0 - epsilon) * (u / (u + omega))
# )
