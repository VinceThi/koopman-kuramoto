import numpy as np


def nu_function(X):
    if X > 1 or X < -1:
        Gamma = np.sqrt(X**2 - 1)
        ratio = (X + Gamma)/(X - Gamma)
        return np.log(ratio)/(2*Gamma)
    elif X == 1:
        return 1
    elif 0 <= X < 1:    # Including 0 is a choice, it could have been included
        angle = np.angle(np.abs(X) + 1j*np.sqrt(1 - X**2))     # np.angle uses np.arctan2, we use the principal value
        return angle/np.sqrt(1 - X**2)
    elif -1 < X < 0:
        angle = np.angle(np.abs(X) + 1j*np.sqrt(1 - X**2))     # np.angle uses np.arctan2, we use the principal value
        return -angle/np.sqrt(1 - X**2)
    elif X == -1:
        return -1
    else:
        raise ValueError("X must be a real number.")


def nu_derivative(X):
    if X > 1 or X < -1:
        return (1 - nu_function(X)*X)/(X**2 - 1)
    elif X == 1 or X == -1:
        return -1/3
    elif 0 <= X < 1:
        angle = np.angle(np.abs(X) + 1j*np.sqrt(1 - X**2))     # np.angle uses np.arctan2, we use the principal value
        return (angle*X - np.sqrt(1 - X**2))/(1 - X**2)**(3/2)
    elif -1 < X < 0:
        angle = np.angle(np.abs(X) + 1j*np.sqrt(1 - X**2))     # np.angle uses np.arctan2, we use the principal value
        return (-angle*X - np.sqrt(1 - X**2))/(1 - X**2)**(3/2)
    else:
        raise ValueError(f"in function nu_derivative, X must be a real number. In this case, X = {X}")


def disk_automorphism(U, V, z):
    # return (np.conjugate(U)*z - V) / (-np.conjugate(V)*z + U)
    return (U*z + V)/(np.conjugate(V)*z + np.conjugate(U))


# For the non-autonomous model (but this was a mistake, we in fact have the same determining equations!)
def determining_equations_disk_automorphism(t, state, hattheta, current_index, omega, coupling):
    R, Phi, Y = state
    assert R**2 - Y**2 + 1 > 0
    p1 = coupling/2*np.sum(np.exp(1j*hattheta[current_index, :]))
    rho1, phi1 = np.abs(p1), np.angle(p1)
    dRdt = -2*rho1*Y*np.sin(Phi - phi1)
    dPhidt = omega - 2*rho1*Y/R*np.cos(Phi - phi1)
    dYdt = -2*rho1*R*np.sin(Phi - phi1)
    return np.array([dRdt, dPhidt, dYdt])


# def determining_equations_disk_automorphism(t, state, W, omega, coupling):
#     N = len(W[0])
#     theta = state[:N]
#     R, Phi, Y = state[-3], state[-2], state[-1]
#     assert R**2 - Y**2 + 1 > 0
#     p1 = coupling/2*np.sum(np.exp(1j*theta))
#     rho1, phi1 = np.abs(p1), np.angle(p1)
#     dthetadt = omega + coupling*(np.cos(theta)*(W@np.sin(theta)) - np.sin(theta)*(W@np.cos(theta)))
#     dRdt = -2*rho1*Y*np.sin(Phi - phi1)
#     dPhidt = omega - 2*rho1*Y/R*np.cos(Phi - phi1)
#     dYdt = -2*rho1*R*np.sin(Phi - phi1)
#     return np.concatenate([dthetadt, np.array([dRdt, dPhidt, dYdt])])


def determining_equations_real_disk_automorphism(t, state, theta, current_index, omega, coupling):
    R, Phi, Y = state
    p0 = len(theta[0, :])*coupling/2
    p1 = coupling/2*np.sum(np.exp(1j*theta[current_index, :]))
    p2 = coupling/2*np.sum(np.exp(2*1j*theta[current_index, :]))
    rho1, phi1 = np.abs(p1), np.angle(p1)
    rho2, phi2 = np.abs(p2), np.angle(p2)
    chi1 = 2*rho1*np.sin(Phi - phi1)
    chi2 = p0 - rho2*np.cos(2*Phi - phi2)
    assert R**2 - Y**2 + 1 > 0
    X = np.sqrt(R**2 - Y**2 + 1)
    mu = nu_derivative(X)*(chi1*Y*R + chi2*R**2)
    dRdt = (chi2 - mu)*R
    dPhidt = omega + rho2*np.sin(2*Phi - phi2)
    dYdt = -mu*Y - chi1*R
    return np.array([dRdt, dPhidt, dYdt])


def determining_equations_real_disk_automorphism_kuramoto(t, state, W, omega, coupling):
    N = len(W[0])
    theta = state[:N]
    R, Phi, Y = state[-3], state[-2], state[-1]
    dthetadt = omega + coupling*(np.cos(theta)*(W@np.sin(theta)) - np.sin(theta)*(W@np.cos(theta)))
    p0 = N*coupling/2
    p1 = coupling/2*np.sum(np.exp(1j*theta))
    p2 = coupling/2*np.sum(np.exp(2*1j*theta))
    rho1, phi1 = np.abs(p1), np.angle(p1)
    rho2, phi2 = np.abs(p2), np.angle(p2)
    chi1 = 2*rho1*np.sin(Phi - phi1)
    chi2 = p0 - rho2*np.cos(2*Phi - phi2)
    assert R**2 - Y**2 + 1 > 0
    X = np.sqrt(R**2 - Y**2 + 1)
    mu = nu_derivative(X)*(chi1*Y*R + chi2*R**2)
    dRdt = (chi2 - mu)*R
    dPhidt = omega + rho2*np.sin(2*Phi - phi2)
    dYdt = -mu*Y - chi1*R
    return np.concatenate([dthetadt, np.array([dRdt, dPhidt, dYdt])])


def determining_equations_disk_automorphism_bounded(t, state, theta, current_index, omega, coupling):
    rho, Psi, phi = state
    p0 = len(theta[0, :])*coupling/2
    p1 = coupling/2*np.sum(np.exp(1j*theta[current_index, :]))
    p2 = coupling/2*np.sum(np.exp(2*1j*theta[current_index, :]))
    rho1, phi1 = np.abs(p1), np.angle(p1)
    rho2, phi2 = np.abs(p2), np.angle(p2)
    chi1 = 2*rho1*np.sin(Psi - phi/2 - phi1)
    chi2 = p0 - rho2*np.cos(2*Psi - phi - phi2)
    X = np.cos(phi/2)/np.sqrt(1 - rho**2)
    assert X**2 >= 0
    mu = nu_derivative(X)*rho/(1 - rho**2)*(np.sin(phi/2)*chi1 + rho*chi2)
    drhodt = (chi2 - mu)*rho*(1 - rho**2)
    dphidt = -2*np.tan(phi/2)*((1 - rho**2)*mu + rho**2*chi2) - 2*rho*chi1/np.cos(phi/2)
    dPsidt = omega + rho2*np.sin(2*Psi - phi - phi2) + dphidt/2
    return np.array([drhodt, dPsidt, dphidt])


# def determining_equations_disk_automorphism(t, state, theta, current_index, omega, coupling):
#     Z, phi = state
#     Zbar = np.conjugate(Z)
#     p0 = len(theta[0, :])*coupling/2
#     p1 = coupling/2*np.sum(np.exp(1j*theta[current_index, :]))
#     p2 = coupling/2*np.sum(np.exp(2*1j*theta[current_index, :]))
#     e = np.exp(1j*phi/2)
#     ebar = np.conjugate(e)
#     X = e*np.cos(phi/2)
#     nu = nu_function(X)
#     f = 1j*(e**2)*(1 - nu*X)/(2*nu*(X**2 - 1))
#     dphidt = 2*(np.conjugate(p1)*ebar*Z - p1*e*Zbar)/(ebar*f - e*np.conjugate(f))
#     mu = f*dphidt
#     dZdt = (1j*omega + p0 - mu)*Z - p2*(e**2)*Zbar
#     return np.array([dZdt, dphidt])


# def determining_equations_disk_automorphism_bounded(t, state, theta, current_index, omega, coupling):
#     rho, Psi, phi = state
#     p0 = len(theta[0, :])*coupling/2
#     p1 = coupling/2*np.sum(np.exp(1j*theta[current_index, :]))
#     p2 = coupling/2*np.sum(np.exp(2*1j*theta[current_index, :]))
#     rho1, phi1 = np.abs(p1), np.angle(p1)
#     rho2, phi2 = np.abs(p2), np.angle(p2)
#     chi1 = 2*rho1*np.sin(Psi - phi/2 - phi1)
#     chi2 = p0 - rho2*np.cos(2*Psi - phi - phi2)
#     nu = nu_function_bounded(rho, phi)
#     mu = ((1 + nu*np.cos(phi/2)/np.sqrt(1 - rho**2))/(rho**2 + np.cos(phi/2)**2 - 1))*(chi1*rho*np.sin(phi/2) + chi2*rho**2)
#     drhodt = (chi2 - mu)*rho*(1 - rho**2)
#     dphidt = 2*mu*np.tan(phi/2) + 2*rho*chi1/np.cos(phi/2) - 2*(rho**2*np.sin(phi/2))/(np.sqrt(1 - rho**2))*(chi2 - mu)
#     dPsidt = omega + rho2*np.sin(2*Psi - phi - phi2) + dphidt/2
#     return np.array([drhodt, dPsidt, dphidt])


# def determining_equations_real_disk_automorphism(t, state, timesteps, theta, current_index, omega, coupling):
#     R, Phi, Y = state
# 
#     # Find the index corresponding to the current time step
#     index = np.searchsorted(timesteps, t) - 1
#     if index < 0:
#         index = 0
#     elif index >= len(theta[0, :]):
#         index = len(theta[0, :]) - 1
# 
#     # Get the value of the non-autonomous parts at the current time step
#     theta_t = theta[index, :]
# 
#     p0 = len(theta[0, :])*coupling/2
#     p1 = coupling/2*np.sum(np.exp(1j*theta_t))
#     p2 = coupling/2*np.sum(np.exp(2*1j*theta_t))
#     rho1, phi1 = np.abs(p1), np.angle(p1)
#     rho2, phi2 = np.abs(p2), np.angle(p2)
#     chi1 = 2*rho1*np.sin(Phi - phi1)
#     chi2 = p0 - rho2*np.cos(2*Phi - phi2)
#     mu = (1 - nu_function(R, Y)*np.sqrt(R**2 - Y**2 + 1))/(R**2 - Y**2)*(chi1*Y*R + chi2*R**2)
#     dRdt = (chi2 - mu)*R
#     dPhidt = omega + rho2*np.sin(2*Phi - phi2)
#     dYdt = -mu*Y - chi1*R
#     return np.array([dRdt, dPhidt, dYdt])
# 
# 