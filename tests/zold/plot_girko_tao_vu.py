# Circular law for a weighted, directed Erdős–Rényi graph
# Now weights R_ij ~ Uniform[-1,1], independent of Bernoulli(p) adjacency.

import numpy as np
from plots.config_rcparams import *
plt.rcParams.update({
    "text.usetex": True,               # Use LaTeX for all text
    "font.family": "serif",            # Use serif fonts
    "font.serif": ["Computer Modern"], # Same as LaTeX default
    "axes.unicode_minus": False        # Allow proper minus signs
})

rng = np.random.default_rng(42)

def weighted_directed_ER_uniform(N, p, rng):
    B = rng.binomial(1, p, size=(N, N)).astype(float)
    R = rng.uniform(-1, 1, size=(N, N))
    A = B * R
    np.fill_diagonal(A, 0.0)  # optional: remove self-loops
    return A

def scaled_matrix(N, p, rng):
    A = weighted_directed_ER_uniform(N, p, rng)
    # Var(A_ij) = p * Var(U[-1,1]) = p * 1/3 = p/3
    # scale so that Var(entry) = 1/N => divide by sqrt(N * p/3)
    return A / np.sqrt(N * p / 3)

def eigvals_scaled(N, p, rng):
    X = scaled_matrix(N, p, rng)
    return np.linalg.eigvals(X)

# --- 1) Scatter plots for increasing N ---
N = 1000
p = 0.2
vals = eigvals_scaled(N, p, rng)
plt.figure(figsize=(3, 3))
plt.scatter(vals.real, vals.imag, s=8, alpha=0.7, label=f'N={N}', color='grey')
th = np.linspace(0, 2*np.pi, 512)
plt.plot(np.cos(th), np.sin(th), linewidth=3, color=total_color)  # unit circle
plt.gca().set_aspect('equal', 'box')
# plt.title(f'Circular law: weighted directed ER (Uniform[-1,1]), p={p}, N={N}')
plt.xlabel('Re($\lambda$)'); plt.ylabel('Im($\lambda$)')
# plt.legend()
plt.tight_layout()
plt.show()

# # --- 2) Radial CDF vs theory F(r)=r^2 ---
# N_cdf, trials = 160, 8
# radii = np.concatenate([np.abs(eigvals_scaled(N_cdf, p, rng)) for _ in range(trials)])
# r_sorted = np.sort(radii)
# F_emp = np.arange(1, r_sorted.size + 1) / r_sorted.size
# r_grid = np.linspace(0, max(1.0, r_sorted.max()), 400)
# F_theory = np.clip(r_grid**2, 0, 1)
#
# plt.figure(figsize=(7,5))
# plt.plot(r_sorted, F_emp, linewidth=2, label='Empirical CDF (aggregated)')
# plt.plot(r_grid, F_theory, linewidth=2, label=r'Theory $F(r)=r^2$ (uniform disk)')
# plt.xlim(0, max(1.0, r_sorted.max())); plt.ylim(0, 1)
# plt.xlabel('Radius r = |$\lambda$|'); plt.ylabel('CDF')
# plt.title(f'Radial CDF vs circular law, Weighted Directed ER (Uniform[-1,1]): N={N_cdf}, p={p}, trials={trials}')
# plt.legend(); plt.tight_layout(); plt.show()
#
# print("Empirical mean radius (N=160):", radii.mean())
# print("Fraction inside unit disk (N=160):", np.mean(radii <= 1.0))
