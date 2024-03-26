import numpy as np
from dynamics.watanabe_strogatz import ws_transformation, Z_dot


# def infinitesimal_condition_symmetry_kuramoto(t, state, w, coupling, omega):
#     Z, phi = state
#     F_bar = coupling / (2 * 1j) * np.sum(ws_transformation(Z, phi, w))
#     G = omega
#     F = np.conjugate(F_bar)
#
#     dphidt = F*Z + 2*F_bar*np.conjugate(Z) + (np.exp(1j*phi) - 1)*F_bar/Z + F*Z*np.exp(1j*phi)
#     return np.array([Z_dot(Z, F, G, F_bar), np.real(dphidt)])


def rfun(Z, Zbar, phi):
    x = 1 + np.exp(-1j*phi)
    y = np.sqrt((1 - np.exp(-1j*phi))**2 + 4*np.exp(-1j*phi)*Z*Zbar)
    return np.log((x + y)/(x - y))/y


def infinitesimal_condition_symmetry_kuramoto(t, state, w, coupling, omega):
    Z, Zbar, phi = state
    r = rfun(Z, Zbar, phi)
    p1 = coupling/2*np.sum(ws_transformation(Z, phi, w))
    pm1 = np.conjugate(p1)
    zeta = np.exp(-1j*phi)
    zetabar = np.exp(1j*phi)
    gamma = (1-zeta)**2 + 4*zeta*Z*Zbar
    alpha = (1 - zetabar - r*zeta + r - 2*r*Z*Zbar)/gamma
    beta = ((1 + zeta)/(1 - Z*Zbar) - 2*r*zeta)/gamma

    A = np.array([[r + alpha*(zeta-1) + beta*Z*Zbar, - beta*Zbar**2, beta*(1 - zeta)*Zbar],
                  [-beta*zetabar*Z**2, r + alpha*(zeta - 1) + beta*zetabar*Z*Zbar, beta*(1 - zeta)*Z],
                  [-(r*zetabar + alpha + beta*zetabar*Z*Zbar)*Z, (beta*zetabar*Z*Zbar - alpha)*Zbar, r+2*beta*Z*Zbar]])
    b = np.array([1j*omega*Z + (1-zetabar)*p1, -1j*omega*Zbar + (1-zeta)*pm1, 2*pm1*Z*zeta - 2*p1*Zbar])
    return (1-Z*Zbar)*zeta*(A@b)


def infinitesimal_condition_symmetry_kuramoto_simplified(t, state, w, coupling, omega):
    Z, phi = state
    F_bar = coupling / (2*1j) * np.sum(ws_transformation(Z, phi, w))
    G = omega
    F = np.conjugate(F_bar)

    dphidt = F*Z + 2*F_bar*np.conjugate(Z) + (np.exp(1j*phi) - 1)*F_bar/Z + F*Z*np.exp(1j*phi)
    return np.array([Z_dot(Z, F, G, F_bar), np.real(dphidt)])