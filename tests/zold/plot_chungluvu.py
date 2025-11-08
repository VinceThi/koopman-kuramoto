# Chung–Lu (cas hétérogène seulement): comparaison empirique λ1 vs prédicteur max{sqrt(kmax), <k^2>/<k>}
# Matplotlib-only; taille un peu plus grande mais raisonnable pour exécution ici.

import numpy as np
import matplotlib.pyplot as plt
from plots.config_rcparams import *
plt.rcParams.update({
    "text.usetex": True,               # Use LaTeX for all text
    "font.family": "serif",            # Use serif fonts
    "font.serif": ["Computer Modern"], # Same as LaTeX default
    "axes.unicode_minus": False        # Allow proper minus signs
})

rng = np.random.default_rng(1234)

def power_iteration_largest_eig(A: np.ndarray, n_iter: int = 18, tol: float = 1e-6) -> float:
    n = A.shape[0]
    v = rng.normal(size=n)
    v /= np.linalg.norm(v)
    last = 0.0
    for _ in range(n_iter):
        w = A @ v
        rq = float(v @ w)
        nw = np.linalg.norm(w)
        if nw == 0:
            return 0.0
        v = w / nw
        if abs(rq - last) < tol * max(1.0, abs(rq)):
            break
        last = rq
    return rq

def chung_lu_prob(kappa: np.ndarray) -> np.ndarray:
    s = float(np.sum(kappa))
    P = np.outer(kappa, kappa) / s
    np.fill_diagonal(P, 0.0)
    np.clip(P, 0.0, 1.0, out=P)
    return P

def sample_adj(P: np.ndarray) -> np.ndarray:
    U = rng.random(P.shape)
    A = (U < P).astype(float)
    A = np.triu(A, 1)
    A = A + A.T
    return A

def simulate_hetero(N: int = 1000, alpha: float = 2.6, kmin: float = 1.0, kmax_trunc: float = 50.0,
                    Ek_target: float = 10.0, trials: int = 1000):
    # Truncated power-law latent degrees
    u = rng.random(N)
    a1 = 1.0 - alpha
    samples = (kmin**a1 + u * (kmax_trunc**a1 - kmin**a1)) ** (1.0 / a1)
    scale = Ek_target / np.mean(samples)
    kappa = samples * scale
    # plt.hist(kappa)
    # plt.yscale('log')
    # plt.show()

    P = chung_lu_prob(kappa)
    lambdas = np.empty(trials, dtype=float)
    for t in range(trials):
        A = sample_adj(P)
        lambdas[t] = power_iteration_largest_eig(A, n_iter=50)

    kmean = float(np.mean(kappa))
    k2mean = float(np.mean(kappa**2))
    kmax = float(np.max(kappa))
    predictor = max(np.sqrt(kmax), k2mean / kmean)
    return kappa, lambdas, predictor, dict(N=N, Ek=kmean, Ek2=k2mean, kmax=kmax)

def plot_hist(lambdas: np.ndarray, predictor: float):
    plt.figure(figsize=(6, 4))
    plt.hist(lambdas, bins=50, color='grey', density=True)
    plt.axvline(predictor, linestyle="--", linewidth=2, color=total_color,
                label=r'max$(\sqrt{\kappa_{\mathrm{max}}}, \langle \kappa^2 \rangle/\langle \kappa\rangle)$')
    plt.xlabel(r"$\lambda_1(A)$")
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.legend(loc=1)
    plt.show()

# ---- Run single heterogeneous case ----
kappa, lambdas, predictor, meta = simulate_hetero(N=1000, alpha=2.5, kmin=1.0, kmax_trunc=50.0,
                                                  Ek_target=10.0, trials=5000)

plot_hist(lambdas, predictor)
