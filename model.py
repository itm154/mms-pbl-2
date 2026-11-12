def seirp_ode(t, y, beta, kappa, xi, omega, u, gamma, epsilon, rho):
    """
    SEIRP misinformation model.

    Compartments:
        S: Susceptible
        E: Exposed
        I: Active spreaders
        R: Corrected
        P: Prebunked
    """
    S, E, I, R, P = y
    N = S + E + I + R + P

    # Individual rate of exposure
    lam = (beta * I / N) if N > 0 else 0.0

    dS_dt = -lam * S - u * S + omega * P
    dE_dt = lam * S + (1.0 - epsilon) * lam * P + rho * lam * R - kappa * E - xi * E
    dI_dt = kappa * E - gamma * I
    dR_dt = xi * E + gamma * I - rho * lam * R
    dP_dt = u * S - (1.0 - epsilon) * lam * P - omega * P

    return [dS_dt, dE_dt, dI_dt, dR_dt, dP_dt]


def calculate_r0(beta, kappa, xi, omega, u, gamma, epsilon):
    """
    Calculate the project's reproduction threshold under the
    disease-free equilibrium, assuming R* = 0.

    This expression is undefined when:
        kappa + xi = 0
        u + omega = 0
        gamma = 0
    """
    if (kappa + xi) <= 0 or (u + omega) <= 0 or gamma <= 0:
        return None

    susceptible_fraction = omega / (u + omega)
    prebunked_fraction = u / (u + omega)

    return (
        beta
        * kappa
        / (gamma * (kappa + xi))
        * (susceptible_fraction + (1.0 - epsilon) * prebunked_fraction)
    )
