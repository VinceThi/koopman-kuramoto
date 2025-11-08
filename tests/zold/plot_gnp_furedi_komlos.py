import matplotlib.pyplot as plt
import numpy as np
from math import sqrt, pi
from scipy.sparse import triu, csr_matrix
from scipy.sparse.linalg import eigsh
from scipy.stats import kstest, norm, shapiro
from plots.config_rcparams import *
plt.rcParams.update({
    "text.usetex": True,               # Use LaTeX for all text
    "font.family": "serif",            # Use serif fonts
    "font.serif": ["Computer Modern"], # Same as LaTeX default
    "axes.unicode_minus": False        # Allow proper minus signs
})

# ------------------- parameters -------------------
N = 1000         # big N is fine here
p = 0.2
trials = 5000
rng = np.random.default_rng(123)
# --------------------------------------------------

def sample_lambda1_Gnp_sparse(N, p, trials, rng):
    vals = np.empty(trials)
    for t in range(trials):
        # sample upper-tri angular in sparse form, then symmetrize
        U = triu((rng.random((N, N)) < p).astype(np.float64), k=1).tocsr()
        A = U + U.T
        # largest eigenvalue via Lanczos
        # (k=1, which='LA' = largest algebraic)
        vals[t] = eigsh(A, k=1, which='LA', return_eigenvectors=False, tol=1e-3)[0]
    return vals

l1 = sample_lambda1_Gnp_sparse(N, p, trials, rng)

mu = (N - 2) * p + 1
sigma2 = 2 * p * (1 - p)
sigma = sqrt(sigma2)

def normal_pdf(x, mu, sigma):
    return (1.0/(sigma*np.sqrt(2*np.pi))) * np.exp(-0.5*((x-mu)/sigma)**2)

xgrid = np.linspace(l1.min() - 2*sigma, l1.max() + 2*sigma, 1200)
plt.figure(figsize=(6, 3))
plt.hist(l1, bins=60, density=True, alpha=0.75, label=r'Distribution empirique $\lambda_1(A)$, $\mathcal{G}(N,p)$', color='grey')
plt.plot(xgrid, normal_pdf(xgrid, mu, sigma), linewidth=3,
         label=rf'$\mathcal{{N}}(\mu^\prime,\sigma^2)$ avec $\mu^\prime=(N-2)p+1$, $\sigma^2=2p(1-p)$', color=total_color)
# plt.title(fr'Leading eigenvalue of $G(N,p)$: $N={N}$, $p={p}$, trials={trials}')
plt.xlabel(r'$\lambda_1(A)$');
#plt.ylabel('Density');
plt.legend(loc=2, bbox_to_anchor=(0, 1.05));
plt.tight_layout()
plt.ylim([-0.05, 1.05])
plt.yticks([0, 0.5, 1])
plt.show()
# plt.ylim([-0.05, 1.05])
# plt.yticks([0, 0.5, 1])
# plt.savefig('chapitre2_furedikomlos_gnp.pdf')
# z = (l1 - mu) / sigma
# xgrid2 = np.linspace(z.min() - 1.0, z.max() + 1.0, 1200)
# std_pdf = (1.0/np.sqrt(2*np.pi)) * np.exp(-0.5 * xgrid2**2)
#
# # plt.figure(figsize=(8,5))
# # plt.hist(z, bins=60, density=True, alpha=0.75, label='Empirical z-scores')
# # plt.plot(xgrid2, std_pdf, linewidth=3, label=r'$\mathcal{N}(0,1)$')
# # plt.title(r'Standardized $\lambda_1$: $z = (\lambda_1 - \mu)/\sigma$')
# # plt.xlabel('z'); plt.ylabel('Density'); plt.legend(); plt.tight_layout(); plt.show()
#
# print("Empirical mean  E[λ1] ≈", l1.mean(), "  vs  μ =", mu)
# print("Empirical std   SD[λ1] ≈", l1.std(ddof=1), " vs  σ =", sigma)
#
# ks_stat, ks_p = kstest(z, 'norm')
# sha_stat, sha_p = shapiro(z)
# print(f"KS test vs N(0,1): stat={ks_stat:.3f}, p={ks_p:.3g}")
# print(f"Shapiro test:      stat={sha_stat:.3f}, p={sha_p:.3g}")
