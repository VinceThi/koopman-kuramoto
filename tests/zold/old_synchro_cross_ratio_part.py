
# # @njit(fastmath=True)
# def integrate_dopri45_non_autonomous(t0, t1, dt, dynamics, init_cond, non_autonomous_term, *args):
#     """ For nonautonomous dynamics where the explicit time-dependence is unknown, but the associated
#         time series (non_autonomous_term) is known. """
#     f = dynamics
#     tvec = np.arange(t0, t1, dt)
#     sol = [init_cond]
#     for i, t in enumerate(tvec[0:-1]):
#         k1 = f(t, sol[i], non_autonomous_term, i, *args)
#         k2 = f(t + 1./5*dt, sol[i] + dt*(1./5*k1), non_autonomous_term, i, *args)
#         k3 = f(t + 3./10*dt, sol[i] + dt*(3./40*k1 + 9./40*k2), non_autonomous_term, i, *args)
#         k4 = f(t + 4./5*dt, sol[i] + dt*(44./45*k1 - 56./15*k2 + 32./9*k3), non_autonomous_term, i, *args)
#         k5 = f(t + 8./9*dt, sol[i] + dt*(19372./6561*k1 - 25360./2187*k2 + 64448./6561*k3 - 212./729*k4),
#                non_autonomous_term, i, *args)
#         k6 = f(t + dt, sol[i] + dt*(9017./3168*k1 - 355./33*k2 + 46732./5247*k3 + 49./176*k4 - 5103./18656*k5),
#                non_autonomous_term, i, *args)
#         v5 = 35./384*k1 + 500./1113*k3 + 125./192*k4 - 2187./6784*k5 + 11./84*k6
#         # k7 = f(t + dt, sol[i] + dt*v5, *args)
#         # v4 = 5179./57600*k1 + 7571./16695*k3 + 393./640*k4 \
#         #     - 92097./339200*k5 + 187./2100*k6 + 1./40*k7
#         sol.append(sol[i] + dt*v5)
#     return sol
#
# # @njit(fastmath=True)
# def ws_transformation(Z, phi, w):
#     return (np.exp(1j*phi)*w + Z)/(1 + np.exp(1j*phi)*np.conjugate(Z)*w)
#
# # @njit(fastmath=True)
# def ws_equations_kooku1_fig3(t, state, theta, current_index, w, calA_sources, calA_row_periphery, omega):
#     """
#     This is a hardcoded function for the example of Fig. 3 in Thibeault et al., Kuramoto meets Koopman, 2025
#     sources_input = np.sum(*z_sources)
#     """
#     zt = np.exp(1j*theta)
#     sources_input = np.sum(calA_sources*zt[current_index, :3])
#     Z = state[0]
#     phi = state[1]
#     F = np.sum(calA_row_periphery*ws_transformation(Z, phi, w)) + sources_input
#     G = omega
#     F_bar = np.conjugate(F)
#     dotZ = F + 1j*G*Z - F_bar*Z**2
#     dotphi = G - 1j*F*np.conjugate(Z) + 1j*F_bar*Z
#     return np.array([dotZ, dotphi], dtype=np.complex128)