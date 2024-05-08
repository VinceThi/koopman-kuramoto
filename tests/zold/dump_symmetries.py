def infinitesimal_condition_symmetry_kuramoto_simplified(t, state, w, coupling, omega):
    Z, phi = state
    F_bar = coupling / (2*1j) * np.sum(ws_transformation(Z, phi, w))
    G = omega
    F = np.conjugate(F_bar)

    dphidt = F*Z + 2*F_bar*np.conjugate(Z) + (np.exp(1j*phi) - 1)*F_bar/Z + F*Z*np.exp(1j*phi)
    return np.array([Z_dot(Z, F, G, F_bar), np.real(dphidt)])


def watanabe_strogatz_generator_on_w(w, Z, phi):
    Zbar = np.conjugate(Z)
    zeta = np.exp(-1j*phi)
    return rfun(Z, Zbar, phi)*(zeta*Z + (1-zeta)*w - Zbar*w**2)