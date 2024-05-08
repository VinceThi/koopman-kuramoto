import numpy as np


def rfun(Z, Zbar, phi):
    zeta = np.exp(-1j*phi)
    x = 1 + zeta
    y = np.sqrt((1 - zeta)**2 + 4*zeta*Z*Zbar)
    ratio = (x + y)/(x - y)
    module = np.abs(ratio)
    argument = np.angle(ratio)
    return (np.log(module) + 1j*argument)/y
    # return np.log((x + y)/(x - y))/y


def infinitesimal_condition_symmetry_kuramoto(t, state, p1, current_index, omega):
    Z, Zbar, phi = state
    r = rfun(Z, Zbar, phi)
    normZ2 = Z*Zbar
    zeta = np.exp(-1j*phi)
    zetabar = np.exp(1j*phi)
    gamma = (1 - zeta)**2 + 4*zeta*Z*Zbar
    alpha = (1 - zetabar - r*zeta + r - 2*r*Z*Zbar)/gamma
    beta = ((1 + zeta)/(1 - Z*Zbar) - 2*r*zeta)/gamma
    k = current_index
    pm1 = np.conjugate(p1[k])

    A = np.array([[r + alpha*(zeta-1) + beta*normZ2, - beta*Zbar**2, beta*(1 - zeta)*Zbar],
                  [-beta*zetabar*Z**2, r + alpha*(zeta - 1) + beta*zetabar*normZ2, beta*(1 - zeta)*Z],
                  [-(r*zetabar + alpha + beta*zetabar*normZ2)*Z, (beta*zetabar*normZ2 - alpha)*Zbar, r+2*beta*normZ2]])
    b = np.array([1j*omega*Z + (1-zetabar)*p1[k], -1j*omega*Zbar + (1-zeta)*pm1, 2*pm1*Z*zeta - 2*p1[k]*Zbar])
    dZdt, dZbardt, dzetadt = (1-normZ2)*zeta*(A@b)
    return np.array([dZdt, dZbardt, np.angle(dzetadt)])


def infinitesimal_condition_symmetry_kuramoto_2(t, state, p1, current_index, omega):
    X, Y, phi = state
    Z = X + 1j*Y
    Zbar = X - 1j*Y
    normZ2 = X**2 + Y**2
    r = rfun(Z, Zbar, phi)
    zeta = np.exp(-1j*phi)
    zetabar = np.exp(1j*phi)
    gamma = (1 - zeta)**2 + 4*zeta*normZ2
    alpha = (1 - zetabar - r*zeta + r - 2*r*normZ2)/gamma
    beta = ((1 + zeta)/(1 - normZ2) - 2*r*zeta)/gamma
    k = current_index
    pm1 = np.conjugate(p1[k])

    A = np.array([[1j*beta*(1 - zetabar)*X*Y + r + alpha*(zeta - 1) + beta*(1 + zetabar)*Y**2,
                   beta*X*(1j*X*(zetabar-1)-Y*(zetabar+1)),
                   2*beta*(1 - zeta)*X],
                  [beta*Y*(1j*Y*(1 - zetabar) - X*(zetabar + 1)),
                   1j*beta*(zetabar - 1)*X*Y + r + alpha*(zeta - 1) + beta*(1 + zetabar)*X**2,
                   2*beta*(1 - zeta)*Y],
                  [-2*alpha*X - r*zetabar*(X + 1j*Y) - 2*1j*beta*zetabar*Y*normZ2,
                   -2*alpha*Y + 1j*r*zetabar*(X + 1j*Y) + 2*1j*beta*zetabar*X*normZ2,
                   2*r+4*beta*normZ2]])
    b = np.array([-omega*Y + ((1-zetabar)*p1[k] + (1-zeta)*pm1)/2,
                  omega*X + ((1-zetabar)*p1[k] - (1-zeta)*pm1)/(2*1j),
                  (pm1*zeta - p1[k])*X + 1j*(pm1*zeta + p1[k])*Y])
    dXdt, dYdt, dzetadt = (1-normZ2)*zeta*(A@b)
    return np.array([dXdt, dYdt, np.angle(dzetadt)])
