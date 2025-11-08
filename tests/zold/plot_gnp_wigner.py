import numpy as np
import matplotlib.pyplot as plt
from plots.config_rcparams import *
plt.rcParams.update({
    "text.usetex": True,               # Use LaTeX for all text
    "font.family": "serif",            # Use serif fonts
    "font.serif": ["Computer Modern"], # Same as LaTeX default
    "axes.unicode_minus": False        # Allow proper minus signs
})


# --- parameters ---
N = 1000          # nodes
p = 0.2          # edge probability
trials = 10       # # of independent graphs
bins = 120
rng = np.random.default_rng(0)
# ------------------

def sample_eigs_Gnp(N, p, trials, rng):
    """Eigenvalues of centered, 1/sqrt(N p (1-p))–scaled G(N,p) adjacency."""
    eigs_all = []
    one_minus_I = np.ones((N, N)) - np.eye(N)
    scale = np.sqrt(N*p*(1 - p))
    for _ in range(trials):
        U = rng.binomial(1, p, size=(N, N))
        U = np.triu(U, 1)           # upper triangle, no self-loops
        A = U + U.T                 # symmetric adjacency
        X = (A - p * one_minus_I) / scale
        eigs_all.append(np.linalg.eigvalsh(X))  # symmetric eigs
    return np.concatenate(eigs_all)

eigs = sample_eigs_Gnp(N, p, trials, rng)

# Semicircle density
x = np.linspace(-2.2, 2.2, 2000)
rho = np.where(np.abs(x) <= 2, (1/(2*np.pi))*np.sqrt(4 - x**2), 0.0)

plt.figure(figsize=(4,3))
plt.hist(eigs, bins=bins, density=True, alpha=0.75, label="Distribution empirique $\mathcal{G}(N,p)$ (ajusté)", color='grey')
plt.plot(x, rho, linewidth=3, label="Loi du demi-cercle de Wigner", color=total_color)
# plt.title(f"Wigner semicircle for G(N,p) after centering & scaling\nN={N}, p={p}, trials={trials}")
plt.xlabel("Valeurs propres")
# plt.ylabel("Densité spectrale")
plt.legend(loc=1, bbox_to_anchor=(1, 1.05))
plt.yticks([0, 0.1, 0.2, 0.3, 0.4])
plt.tight_layout()
plt.show()

print("Mean eigenvalue (empirical):", eigs.mean())
print("Std eigenvalue (empirical) :", eigs.std())
print("Min/Max eigenvalues        :", eigs.min(), eigs.max())