import numpy as np


def nu_function(R, Y):
    X = np.sqrt(R**2 - Y**2 + 1)
    Gamma = np.sqrt(X**2 - 1)
    ratio = (X + Gamma) / (X - Gamma)
    return np.log(ratio) / (2 * Gamma)


def determining_equations_real_disk_automorphism(t, state, theta, current_index, omega, coupling):
    R, Phi, Y = state
    p0 = len(theta[0, :])*coupling/2
    p1 = coupling/2*np.sum(np.exp(1j*theta[current_index, :]))
    p2 = coupling/2*np.sum(np.exp(2*1j*theta[current_index, :]))
    rho1, phi1 = np.abs(p1), np.angle(p1)
    rho2, phi2 = np.abs(p2), np.angle(p2)
    chi1 = 2*rho1*np.sin(Phi - phi1)
    chi2 = p0 - rho2*np.cos(2*Phi - phi2)
    mu = (1 - nu_function(R, Y)*np.sqrt(R**2 - Y**2 + 1))/(R**2 - Y**2)*(chi1*Y*R + chi2*R**2)
    dRdt = (chi2 - mu)*R
    dPhidt = omega + rho2*np.sin(2*Phi - phi2)
    dYdt = -mu*Y - chi1*R
    return np.array([dRdt, dPhidt, dYdt])


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