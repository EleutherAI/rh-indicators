"""Gaussian Process model for exploit rate surfaces.

Fits a 2D GP with binomial likelihood to model P(exploit | checkpoint, prefill)
using Laplace approximation for inference and L-BFGS-B for hyperparameter optimization.

No heavy dependencies (GPflow/TensorFlow) — uses only numpy/scipy.
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
from scipy.special import expit


# ---------------------------------------------------------------------------
# Kernel
# ---------------------------------------------------------------------------

def matern52_kernel(X1, X2, variance, lengthscales):
    """Matérn-5/2 kernel with ARD (separate lengthscale per dimension).

    k(x1, x2) = σ² (1 + √5 r + 5/3 r²) exp(-√5 r)
    where r = ‖(x1 - x2) / ℓ‖₂
    """
    X1s = X1 / lengthscales
    X2s = X2 / lengthscales
    r = cdist(X1s, X2s, metric="euclidean")
    sqrt5r = np.sqrt(5.0) * r
    return variance * (1.0 + sqrt5r + 5.0 / 3.0 * r**2) * np.exp(-sqrt5r)


# ---------------------------------------------------------------------------
# Laplace approximation (Rasmussen & Williams Algorithm 3.1)
# ---------------------------------------------------------------------------

def _laplace_mode(K, k, n, mu, offset=None, g_init=None, max_iter=50, tol=1e-8):
    """Find posterior mode of the zero-mean GP deviation g = f - μ - offset.

    The generative model is:
        g ~ N(0, K)
        f = g + μ + offset
        y_i ~ Binomial(n_i, σ(f_i))

    Args:
        K: (N, N) kernel matrix
        k: (N,) success counts
        n: (N,) trial counts
        mu: scalar or (N,) mean function values
        offset: (N,) known offsets (e.g. KL divergence). Default 0.
        g_init: optional warm-start for g

    Returns:
        g_hat: (N,) posterior mode
        W: (N,) negative Hessian diagonal at mode
        L: (N, N) Cholesky of B = I + W^{1/2} K W^{1/2}
        log_marg_lik: Laplace approximate log marginal likelihood
    """
    N = len(k)
    g = g_init.copy() if g_init is not None else np.zeros(N)
    jitter = 1e-6 * np.eye(N)
    _offset = offset if offset is not None else 0.0

    for _ in range(max_iter):
        f = g + mu + _offset
        p = expit(f)
        grad = k - n * p
        W = np.maximum(n * p * (1.0 - p), 1e-10)

        W_sqrt = np.sqrt(W)
        B = np.eye(N) + np.outer(W_sqrt, W_sqrt) * K + jitter
        L = np.linalg.cholesky(B)

        b = W * g + grad
        Kb = K @ b
        v = np.linalg.solve(L, W_sqrt * Kb)
        a = b - W_sqrt * np.linalg.solve(L.T, v)
        g_new = K @ a

        if np.max(np.abs(g_new - g)) < tol:
            g = g_new
            break
        g = g_new

    # Recompute at final mode
    f = g + mu + _offset
    p = expit(f)
    W = np.maximum(n * p * (1.0 - p), 1e-10)
    W_sqrt = np.sqrt(W)
    B = np.eye(N) + np.outer(W_sqrt, W_sqrt) * K + jitter
    L = np.linalg.cholesky(B)

    # Log marginal likelihood (R&W eq 3.32)
    log_lik = np.sum(k * f - n * np.logaddexp(0, f))
    K_reg = K + jitter
    alpha = np.linalg.solve(K_reg, g)
    log_marg_lik = log_lik - 0.5 * g @ alpha - np.sum(np.log(np.diag(L)))

    return g, W, L, log_marg_lik


def _laplace_mode_constrained(K, k, n, mu, C_i, C_j, C_kl, sigma_is,
                               sigma_slope=0.0,
                               g_init=None, max_iter=100, tol=1e-8):
    """Find posterior mode with binomial likelihood + pairwise KL constraints.

    Uses direct Newton iteration since the pairwise constraints make the
    observation Hessian non-diagonal (can't use the R&W Algorithm 3.1 trick).

    Model:
        g ~ N(0, K)
        f = g + μ
        y_i ~ Binomial(n_i, σ(f_i))               [binomial obs]
        g_{C_i[c]} - g_{C_j[c]} ~ N(-C_kl[c], σ²_c)  [pairwise IS constraints]

    where σ²_c = σ²_IS + (KL_c × sigma_slope)². Large-KL constraints
    get proportionally more slack (logit≈log breaks at high rates).

    Args:
        K: (N, N) kernel matrix
        k: (N,) success counts
        n: (N,) trial counts
        mu: scalar mean function value
        C_i: (M,) indices of pfx=0 cells for each constraint
        C_j: (M,) indices of pfx>0 cells for each constraint
        C_kl: (M,) effective KL values for each constraint (pre-scaled by α)
        sigma_is: base noise std for pairwise constraints. Can be a scalar
                  (applied uniformly) or an (M,) array of per-constraint stds.
        sigma_slope: per-unit-KL noise increase (σ²_c = σ²_base + (KL × slope)²)
        g_init: optional warm-start for g

    Returns:
        g_hat: (N,) posterior mode
        Omega: (N, N) observation model negative Hessian at mode
        K_inv: (N, N) inverse of regularized kernel matrix
        log_marg_lik: Laplace approximate log marginal likelihood
    """
    N = len(k)
    g = g_init.copy() if g_init is not None else np.zeros(N)
    K_reg = K + 1e-6 * np.eye(N)
    K_inv = np.linalg.solve(K_reg, np.eye(N))

    C_i = np.asarray(C_i, dtype=np.intp)
    C_j = np.asarray(C_j, dtype=np.intp)
    C_kl = np.asarray(C_kl, dtype=np.float64)
    M = len(C_i)

    # Per-constraint variance: σ²_c = σ²_base + (KL × slope)²
    # sigma_is can be scalar or per-constraint vector
    sigma_is_arr = np.broadcast_to(np.asarray(sigma_is, dtype=np.float64), (M,))
    sigma_c_sq = sigma_is_arr**2 + (C_kl * sigma_slope)**2  # (M,)
    inv_sigma_c_sq = 1.0 / sigma_c_sq  # (M,)

    # Build constraint Hessian structure (constant across iterations)
    P = np.zeros((N, N))
    for c_idx in range(M):
        ci, cj = C_i[c_idx], C_j[c_idx]
        w_c = inv_sigma_c_sq[c_idx]
        P[ci, ci] += w_c
        P[cj, cj] += w_c
        P[ci, cj] -= w_c
        P[cj, ci] -= w_c

    for _iter in range(max_iter):
        f = g + mu
        p = expit(f)
        w = np.maximum(n * p * (1.0 - p), 1e-10)

        # Binomial gradient
        grad_binom = k - n * p

        # Pairwise constraint gradient (vectorized)
        residuals = g[C_i] - g[C_j] + C_kl
        grad_pair = np.zeros(N)
        np.add.at(grad_pair, C_i, -residuals * inv_sigma_c_sq)
        np.add.at(grad_pair, C_j, residuals * inv_sigma_c_sq)

        # Prior gradient
        grad_prior = -K_inv @ g

        # Newton step
        grad = grad_binom + grad_pair + grad_prior
        Omega = np.diag(w) + P
        H = Omega + K_inv

        try:
            step = np.linalg.solve(H, grad)
        except np.linalg.LinAlgError:
            step = np.linalg.solve(H + 1e-4 * np.eye(N), grad)

        g_new = g + step

        if np.max(np.abs(step)) < tol:
            g = g_new
            break
        g = g_new

    # Recompute at final mode
    f = g + mu
    p = expit(f)
    w = np.maximum(n * p * (1.0 - p), 1e-10)
    Omega = np.diag(w) + P

    residuals = g[C_i] - g[C_j] + C_kl
    log_lik_binom = np.sum(k * f - n * np.logaddexp(0, f))
    log_lik_pair = -0.5 * np.sum(residuals**2 * inv_sigma_c_sq)
    # Normalization of per-constraint Gaussians
    log_lik_pair -= 0.5 * np.sum(np.log(sigma_c_sq))
    log_lik_total = log_lik_binom + log_lik_pair

    # Log marginal likelihood:
    # log p(y|θ) ≈ log_lik_total - 0.5 g^T K^{-1} g - 0.5 log|I + K Ω|
    IKO = np.eye(N) + K_reg @ Omega
    _, logdet_IKO = np.linalg.slogdet(IKO)
    log_marg_lik = log_lik_total - 0.5 * g @ K_inv @ g - 0.5 * logdet_IKO

    return g, Omega, K_inv, log_marg_lik


# ---------------------------------------------------------------------------
# Model class
# ---------------------------------------------------------------------------

class ExploitRateGP:
    """2D GP with binomial likelihood for exploit rate as a function of
    (checkpoint, prefill_tokens).

    f ~ GP(μ₀, Matérn-5/2 with ARD)
    y_i ~ Binomial(n_i, σ(f(x_i)))

    Inference via Laplace approximation; hyperparameters optimized by
    maximizing the Laplace marginal likelihood with L-BFGS-B.
    """

    def __init__(self):
        self.params_ = None
        self._X = None
        self._k = None
        self._n = None
        self._g_hat = None  # posterior mode (zero-mean deviation)
        self._W = None
        self._L = None
        self._K = None
        self._log_marg_lik = None
        self._x0_center = None  # normalization for tanh mean
        self._x0_scale = None

    @staticmethod
    def _tanh_warp(t_norm, gamma):
        """Compute tanh(γ·t)/tanh(γ), reducing to identity as γ→0."""
        if abs(gamma) < 0.01:
            return t_norm
        return np.tanh(gamma * t_norm) / np.tanh(gamma)

    def _mean_function(self, X):
        """Evaluate the mean function at input locations.

        Tanh mean in the first input dimension (log-checkpoint):
            t_norm = (x₀ - center) / scale   →  maps training range to [-1, 1]
            μ(x) = β₀ + β₁ × tanh(γ·t_norm) / tanh(γ)

        When γ≈0 this reduces to β₀ + β₁·t_norm (linear).
        When γ is large the mean saturates at β₀ ± β₁.

        Returns scalar or (N,) array depending on params.
        """
        if self.params_ is None:
            raise RuntimeError("Model not fitted yet")
        b0 = self.params_["mean_intercept"]
        b1 = self.params_["mean_slope"]
        gamma = self.params_.get("mean_gamma", 0.0)
        if b1 == 0.0:
            return b0
        x0 = np.asarray(X)[:, 0]
        t_norm = (x0 - self._x0_center) / self._x0_scale
        return b0 + b1 * self._tanh_warp(t_norm, gamma)

    def fit(self, X, successes, trials, optimize_mean=True):
        """Fit the GP to aggregated count data.

        Args:
            X: (N, D) input features. For exploit rate surface, typically
               D=2 with (log_checkpoint, prefill_tokens).
            successes: (N,) success counts per cell
            trials: (N,) trial counts per cell
            optimize_mean: if True, optimize the linear mean function;
                           otherwise fix at β₀=-3.0, β₁=0

        Returns:
            self
        """
        self._X = np.asarray(X, dtype=np.float64)
        self._k = np.asarray(successes, dtype=np.float64)
        self._n = np.asarray(trials, dtype=np.float64)
        N, D = self._X.shape

        # Compute normalization for tanh mean (maps x₀ to [-1, 1])
        x0 = self._X[:, 0]
        x0_min, x0_max = x0.min(), x0.max()
        self._x0_center = (x0_min + x0_max) / 2.0
        self._x0_scale = max((x0_max - x0_min) / 2.0, 1e-6)
        x0_norm = (x0 - self._x0_center) / self._x0_scale

        # Data-driven initialization
        emp_rates = np.clip(self._k / np.maximum(self._n, 1), 0.001, 0.999)
        emp_logits = np.log(emp_rates / (1.0 - emp_rates))
        # Moderate initial variance — not too small (can't fit) or too large (degenerate)
        log_var0 = np.log(np.clip(np.var(emp_logits), 1.0, 25.0))
        # Lengthscales: half the range of each input dimension
        x_ranges = np.ptp(self._X, axis=0)
        log_ls0 = np.log(np.maximum(x_ranges / 2.0, 0.1))

        # Linear mean initialization in normalized coords:
        # fit regression of empirical logits on t_norm
        if np.ptp(x0_norm) > 0 and optimize_mean:
            from scipy.stats import linregress
            lr = linregress(x0_norm, emp_logits)
            beta0_init = lr.intercept
            beta1_init = lr.slope
        else:
            beta0_init = float(np.median(emp_logits))
            beta1_init = 0.0
        gamma_init = 0.0  # start linear

        # Bounds: prevent degenerate hyperparameters
        var_bounds = (np.log(0.5), np.log(100.0))
        ls_bounds = [(np.log(0.05 * max(r, 0.1)), np.log(10.0 * max(r, 1.0))) for r in x_ranges]
        beta0_bounds = (-15.0, 10.0)
        beta1_bounds = (-10.0, 10.0)
        gamma_bounds = (0.0, 5.0)  # 0=linear, 5=very S-shaped

        if optimize_mean:
            theta0 = np.concatenate([[log_var0], log_ls0, [beta0_init, beta1_init, gamma_init]])
            bounds = [var_bounds] + ls_bounds + [beta0_bounds, beta1_bounds, gamma_bounds]
        else:
            theta0 = np.concatenate([[log_var0], log_ls0])
            bounds = [var_bounds] + ls_bounds

        def neg_lml(theta):
            variance = np.exp(theta[0])
            lengthscales = np.exp(theta[1 : 1 + D])
            if optimize_mean:
                beta0 = theta[1 + D]
                beta1 = theta[2 + D]
                gamma = theta[3 + D]
                warped = self._tanh_warp(x0_norm, gamma)
                mu_vec = beta0 + beta1 * warped
            else:
                mu_vec = beta0_init

            K = matern52_kernel(self._X, self._X, variance, lengthscales)
            K += 1e-6 * variance * np.eye(N)  # jitter scales with variance

            try:
                g_hat, _, _, lml = _laplace_mode(K, self._k, self._n, mu_vec)
                return -lml
            except (np.linalg.LinAlgError, ValueError):
                return 1e10

        result = minimize(
            neg_lml,
            theta0,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 500, "ftol": 1e-8},
        )

        # Unpack optimized parameters
        variance = np.exp(result.x[0])
        lengthscales = np.exp(result.x[1 : 1 + D])
        if optimize_mean:
            beta0 = result.x[1 + D]
            beta1 = result.x[2 + D]
            gamma = result.x[3 + D]
        else:
            beta0 = beta0_init
            beta1 = 0.0
            gamma = 0.0

        self.params_ = {
            "variance": variance,
            "lengthscales": lengthscales,
            "mean_intercept": beta0,
            "mean_slope": beta1,
            "mean_gamma": gamma,
            # Backward compat: mean_constant is the intercept
            "mean_constant": beta0,
        }

        # Final Laplace at optimum
        warped = self._tanh_warp(x0_norm, gamma)
        mu_vec = beta0 + beta1 * warped
        self._K = matern52_kernel(self._X, self._X, variance, lengthscales)
        self._K += 1e-6 * variance * np.eye(N)
        self._g_hat, self._W, self._L, self._log_marg_lik = _laplace_mode(
            self._K, self._k, self._n, mu_vec
        )

        return self

    def predict(self, X_new):
        """Predict exploit rate at new input locations.

        Args:
            X_new: (M, D) new input locations

        Returns:
            dict with keys:
                rate_mean: (M,) sigmoid(f_mean)
                rate_lower: (M,) lower 95% credible bound
                rate_upper: (M,) upper 95% credible bound
                f_mean: (M,) latent GP mean
                f_var: (M,) latent GP variance
        """
        X_new = np.asarray(X_new, dtype=np.float64)
        var = self.params_["variance"]
        ls = self.params_["lengthscales"]
        mu_vec = self._mean_function(X_new)

        k_star = matern52_kernel(X_new, self._X, var, ls)
        k_ss = np.full(len(X_new), var)  # Matérn diag = variance

        # Predictive mean: f* = k*ᵀ K⁻¹ g_hat + μ(x*)
        alpha = np.linalg.solve(self._K, self._g_hat)
        f_mean = k_star @ alpha + mu_vec

        # Predictive variance (R&W eq 3.24)
        W_sqrt = np.sqrt(np.maximum(self._W, 1e-10))
        v = np.linalg.solve(self._L, (W_sqrt[:, None] * k_star.T))
        f_var = np.maximum(k_ss - np.sum(v**2, axis=0), 1e-10)
        f_std = np.sqrt(f_var)

        return {
            "rate_mean": expit(f_mean),
            "rate_lower": expit(f_mean - 1.96 * f_std),
            "rate_upper": expit(f_mean + 1.96 * f_std),
            "f_mean": f_mean,
            "f_var": f_var,
        }

    def predict_surface(self, checkpoints, prefill_levels, log_checkpoint=True):
        """Predict on a grid of (checkpoint, prefill) for surface plotting.

        Args:
            checkpoints: array of checkpoint values
            prefill_levels: array of prefill token counts
            log_checkpoint: if True, log-transform checkpoints (should match training)

        Returns:
            dict with 2D arrays keyed by 'rate_mean', 'rate_lower', etc.,
            plus 'checkpoints' and 'prefill_levels' grid vectors
        """
        ckpts = np.log(checkpoints) if log_checkpoint else np.array(checkpoints)
        pfx = np.array(prefill_levels, dtype=np.float64)
        grid_c, grid_p = np.meshgrid(ckpts, pfx, indexing="ij")
        X_grid = np.column_stack([grid_c.ravel(), grid_p.ravel()])

        preds = self.predict(X_grid)
        n_c, n_p = len(checkpoints), len(prefill_levels)
        result = {
            "checkpoints": np.array(checkpoints),
            "prefill_levels": np.array(prefill_levels),
        }
        for key in ("rate_mean", "rate_lower", "rate_upper", "f_mean", "f_var"):
            result[key] = preds[key].reshape(n_c, n_p)
        return result

    @property
    def lengthscales(self):
        return self.params_["lengthscales"] if self.params_ else None

    @property
    def variance(self):
        return self.params_["variance"] if self.params_ else None

    @property
    def mean_constant(self):
        """Backward compat: returns intercept."""
        return self.params_["mean_intercept"] if self.params_ else None

    @property
    def mean_intercept(self):
        return self.params_["mean_intercept"] if self.params_ else None

    @property
    def mean_slope(self):
        return self.params_["mean_slope"] if self.params_ else None

    @property
    def mean_gamma(self):
        return self.params_.get("mean_gamma", 0.0) if self.params_ else None

    @property
    def log_marginal_likelihood(self):
        return self._log_marg_lik


# ---------------------------------------------------------------------------
# Offset GP: 1D GP with known KL offsets
# ---------------------------------------------------------------------------

class ExploitRateOffsetGP:
    """1D GP with binomial likelihood and known KL offsets.

    Models the true (unprompted) exploit rate as a function of checkpoint only:

        g(ckpt) ~ GP(μ, Matérn-5/2)
        f_i = g(ckpt_i) + KL_i
        k_i ~ Binomial(n_i, σ(f_i))

    Each (checkpoint, prefill) observation contributes to estimating the same
    g(checkpoint). The KL offset captures the importance-sampling relationship:
    in logit space, the observed rate ≈ true rate + KL (exact for small rates).

    At prediction time with offset=0, σ(g(ckpt)) gives the true unprompted rate.
    """

    def __init__(self):
        self.params_ = None
        self._X = None
        self._k = None
        self._n = None
        self._offsets = None
        self._g_hat = None
        self._W = None
        self._L = None
        self._K = None
        self._log_marg_lik = None

    def fit(self, X, successes, trials, offsets):
        """Fit the 1D GP with known KL offsets.

        The effective offset is α * KL, where α is learned. The IS relationship
        log P(obs) ≈ log P(true) + KL holds in log-probability space, but in
        logit space the effective coefficient is < 1 and data-dependent.

        Args:
            X: (N,) or (N, 1) checkpoint values (typically log-transformed)
            successes: (N,) success counts per cell
            trials: (N,) trial counts per cell
            offsets: (N,) known offsets (KL divergence values). KL=0 for prefill=0.

        Returns:
            self
        """
        self._X = np.asarray(X, dtype=np.float64).reshape(-1, 1)
        self._k = np.asarray(successes, dtype=np.float64)
        self._n = np.asarray(trials, dtype=np.float64)
        self._offsets = np.asarray(offsets, dtype=np.float64)
        N = len(self._k)

        # Data-driven initialization
        emp_rates = np.clip(self._k / np.maximum(self._n, 1), 0.001, 0.999)
        emp_logits = np.log(emp_rates / (1.0 - emp_rates))
        # Use prefill=0 cells (offset=0) to estimate mu
        p0_mask = self._offsets == 0
        if p0_mask.any():
            mu0 = float(np.median(emp_logits[p0_mask]))
        else:
            mu0 = float(np.median(emp_logits))
        log_var0 = np.log(np.clip(np.var(emp_logits[p0_mask]) if p0_mask.any() else np.var(emp_logits), 1.0, 25.0))
        x_range = float(np.ptp(self._X))
        log_ls0 = np.log(max(x_range / 2.0, 0.1))
        alpha0 = 0.1  # Start conservative — IS relationship is approximate in logit space

        # Bounds
        var_bounds = (np.log(0.5), np.log(100.0))
        ls_bounds = (np.log(0.05 * max(x_range, 0.1)), np.log(10.0 * max(x_range, 1.0)))
        mu_bounds = (-15.0, 10.0)
        alpha_bounds = (0.0, 2.0)

        theta0 = np.array([log_var0, log_ls0, mu0, alpha0])
        bounds = [var_bounds, ls_bounds, mu_bounds, alpha_bounds]

        def neg_lml(theta):
            variance = np.exp(theta[0])
            lengthscale = np.exp(theta[1])
            mu = theta[2]
            alpha = theta[3]

            K = matern52_kernel(self._X, self._X, variance, np.array([lengthscale]))
            K += 1e-6 * variance * np.eye(N)

            try:
                _, _, _, lml = _laplace_mode(
                    K, self._k, self._n, mu, offset=alpha * self._offsets
                )
                return -lml
            except (np.linalg.LinAlgError, ValueError):
                return 1e10

        result = minimize(
            neg_lml, theta0, method="L-BFGS-B", bounds=bounds,
            options={"maxiter": 500, "ftol": 1e-8},
        )

        variance = np.exp(result.x[0])
        lengthscale = np.exp(result.x[1])
        mu = result.x[2]
        alpha = result.x[3]

        self.params_ = {
            "variance": variance,
            "lengthscale": lengthscale,
            "mean_constant": mu,
            "alpha": alpha,
        }

        self._scaled_offsets = alpha * self._offsets
        self._K = matern52_kernel(self._X, self._X, variance, np.array([lengthscale]))
        self._K += 1e-6 * variance * np.eye(N)
        self._g_hat, self._W, self._L, self._log_marg_lik = _laplace_mode(
            self._K, self._k, self._n, mu, offset=self._scaled_offsets
        )

        return self

    def predict(self, X_new, offset=None):
        """Predict exploit rate at new checkpoint values.

        Args:
            X_new: (M,) or (M, 1) new checkpoint values
            offset: (M,) raw KL offsets for new points (will be scaled by α).
                    Default None → offset=0 → true unprompted rate.

        Returns:
            dict with rate_mean, rate_lower, rate_upper, f_mean, f_var, g_mean
        """
        X_new = np.asarray(X_new, dtype=np.float64).reshape(-1, 1)
        var = self.params_["variance"]
        ls = np.array([self.params_["lengthscale"]])
        mu = self.params_["mean_constant"]
        alpha_kl = self.params_["alpha"]

        k_star = matern52_kernel(X_new, self._X, var, ls)
        k_ss = np.full(len(X_new), var)

        # g(ckpt) prediction — the true logit exploit rate
        solve_alpha = np.linalg.solve(self._K, self._g_hat)
        g_mean = k_star @ solve_alpha + mu

        # Predictive variance (R&W eq 3.24)
        W_sqrt = np.sqrt(np.maximum(self._W, 1e-10))
        v = np.linalg.solve(self._L, (W_sqrt[:, None] * k_star.T))
        f_var = np.maximum(k_ss - np.sum(v**2, axis=0), 1e-10)
        f_std = np.sqrt(f_var)

        # Add scaled offset for observation-space predictions
        if offset is not None:
            f_mean = g_mean + alpha_kl * np.asarray(offset, dtype=np.float64)
        else:
            f_mean = g_mean  # offset=0 → true unprompted rate

        return {
            "rate_mean": expit(f_mean),
            "rate_lower": expit(f_mean - 1.96 * f_std),
            "rate_upper": expit(f_mean + 1.96 * f_std),
            "f_mean": f_mean,
            "f_var": f_var,
            "g_mean": g_mean,
        }

    @property
    def lengthscale(self):
        return self.params_["lengthscale"] if self.params_ else None

    @property
    def variance(self):
        return self.params_["variance"] if self.params_ else None

    @property
    def mean_constant(self):
        return self.params_["mean_constant"] if self.params_ else None

    @property
    def alpha(self):
        return self.params_["alpha"] if self.params_ else None

    @property
    def log_marginal_likelihood(self):
        return self._log_marg_lik


# ---------------------------------------------------------------------------
# Constrained GP: 2D GP + pairwise IS constraints
# ---------------------------------------------------------------------------

class ExploitRateConstrainedGP:
    """2D GP with binomial likelihood + pairwise IS constraints.

    Extends ExploitRateGP with soft constraints linking latent values at
    (ckpt, pfx=0) and (ckpt, pfx) through KL divergence:

        f(ckpt, 0) - f(ckpt, pfx) ~ N(-KL, σ²_IS)

    The binomial observations tell the GP what f(ckpt, pfx) is at each cell.
    The KL constraints propagate that signal down to pfx=0, where direct
    observations are typically 0/n at early checkpoints.

    In logit space for small p (where logit ≈ log), KL is approximately the
    correct additive offset. σ_IS is learned to account for approximation
    error and IS inequality slack.
    """

    def __init__(self):
        self.params_ = None
        self._X = None
        self._k = None
        self._n = None
        self._C_i = None   # pfx=0 indices into X
        self._C_j = None   # pfx>0 indices into X
        self._C_kl = None  # KL values
        self._g_hat = None
        self._K = None
        self._K_inv = None
        self._A = None           # precomputed for predictive variance
        self._alpha_mean = None  # K_inv @ g_hat
        self._log_marg_lik = None
        self._x0_center = None  # normalization for tanh mean
        self._x0_scale = None

    def fit_two_stage(self, X, successes, trials, constraints,
                      base_gp=None, alpha=0.15, sigma_is=1.5, sigma_slope=0.0,
                      optimize_constraints=False, jensen_correction=False):
        """Two-stage fitting: fix kernel from unconstrained GP, add constraints to posterior.

        Stage 1: Fit an unconstrained ExploitRateGP to learn good kernel hyperparameters
                 (variance, lengthscales, mean). Or accept a pre-fit GP.
        Stage 2: Using those fixed kernel params, find the posterior mode under both
                 binomial likelihood AND pairwise KL constraints.

        This avoids the failure mode where jointly optimizing kernel + constraint
        params causes the optimizer to weaken the kernel to accommodate constraints
        (e.g., inflating ls_prefill to ignore the prefill dimension).

        Args:
            X: (N, 2) input features (log_checkpoint, prefill_tokens)
            successes: (N,) success counts per cell
            trials: (N,) trial counts per cell
            constraints: tuple of (C_i, C_j, C_kl) arrays
            base_gp: pre-fit ExploitRateGP to take kernel params from.
                     If None, fits one automatically.
            alpha: KL scaling factor. Effective constraint is
                   f(ckpt,0) - f(ckpt,pfx) ~ N(-α*KL, σ²). Smaller α → weaker
                   constraints. 0.15 works well empirically. Ignored if
                   optimize_constraints=True.
            sigma_is: base noise std for pairwise constraints. Ignored if
                      optimize_constraints=True.
            sigma_slope: per-unit-KL noise increase. σ²_c = σ²_IS + (KL*slope)².
            optimize_constraints: if True, optimize α and σ_IS by maximizing the
                                  constrained LML with fixed kernel params. This is
                                  a cheap 2D optimization (not the full joint opt).
            jensen_correction: if True, use per-cell Jensen gap variance correction.
                              Replaces α*KL with KL - Var(kl)/2 as constraint offset
                              and uses Var(kl) to set per-cell constraint uncertainty.
                              Requires constraints tuple to include kl_var (4th element).
                              When enabled, alpha and sigma_slope are ignored.

        Returns:
            self
        """
        self._X = np.asarray(X, dtype=np.float64)
        self._k = np.asarray(successes, dtype=np.float64)
        self._n = np.asarray(trials, dtype=np.float64)
        N, D = self._X.shape

        # Parse constraints (3-tuple: i, j, kl; or 4-tuple: i, j, kl, kl_var)
        if isinstance(constraints, tuple) and len(constraints) in (3, 4):
            self._C_i = np.asarray(constraints[0])
            self._C_j = np.asarray(constraints[1])
            self._C_kl = np.asarray(constraints[2])
            self._C_kl_var = np.asarray(constraints[3]) if len(constraints) == 4 else np.zeros(len(self._C_i))
        else:
            self._C_i = np.array([c[0] for c in constraints])
            self._C_j = np.array([c[1] for c in constraints])
            self._C_kl = np.array([c[2] for c in constraints])
            self._C_kl_var = np.zeros(len(self._C_i))

        n_constraints = len(self._C_i)

        # Stage 1: get kernel hyperparameters + mean function params
        if base_gp is not None:
            variance = base_gp.params_["variance"]
            lengthscales = base_gp.params_["lengthscales"]
            mean_intercept = base_gp.params_.get("mean_intercept", base_gp.params_.get("mean_constant", -3.0))
            mean_slope = base_gp.params_.get("mean_slope", 0.0)
            mean_gamma = base_gp.params_.get("mean_gamma", 0.0)
            self._x0_center = base_gp._x0_center
            self._x0_scale = base_gp._x0_scale
        else:
            base_gp = ExploitRateGP()
            base_gp.fit(self._X, self._k, self._n)
            variance = base_gp.params_["variance"]
            lengthscales = base_gp.params_["lengthscales"]
            mean_intercept = base_gp.params_["mean_intercept"]
            mean_slope = base_gp.params_["mean_slope"]
            mean_gamma = base_gp.params_.get("mean_gamma", 0.0)
            self._x0_center = base_gp._x0_center
            self._x0_scale = base_gp._x0_scale

        # Compute mean function vector using tanh warp
        x0_norm = (self._X[:, 0] - self._x0_center) / self._x0_scale
        warped = ExploitRateGP._tanh_warp(x0_norm, mean_gamma)
        mu_vec = mean_intercept + mean_slope * warped

        # Stage 2: constrained posterior with fixed kernel
        self._K = matern52_kernel(self._X, self._X, variance, lengthscales)
        self._K += 1e-6 * variance * np.eye(N)

        if jensen_correction and n_constraints > 0:
            # Jensen gap variance correction: replace α*KL with KL - Var/2
            # and use Var to set per-cell constraint uncertainty.
            #
            # From IS theory: log P(exploit) >= E[log P(exploit|p)] - KL(Q||P)
            # Jensen gap ≈ Var(log w) / 2 where log w = log P(p)/Q(p)
            # Since kl_divergence = -log w, Var(kl) = Var(log w)
            #
            # Corrected constraint offset: KL - Var(kl)/2
            # Constraint uncertainty: sigma_IS^2 + Var(kl) (data-driven)
            jensen_gap = self._C_kl_var / 2.0  # second-order correction
            C_kl_effective = np.maximum(self._C_kl - jensen_gap, 0.0)
            # Per-cell sigma: base sigma + data-driven variance contribution
            sigma_is_vec = np.sqrt(sigma_is**2 + self._C_kl_var)

        elif n_constraints > 0 and optimize_constraints:
            # Optimize α and σ_IS by maximizing constrained LML (2D search)
            C_kl_effective = None  # will be set after optimization
            sigma_is_vec = None
            g_warm = None

            def neg_lml_constraint(theta):
                nonlocal g_warm
                a = theta[0]
                s = np.exp(theta[1])  # log-parameterize sigma for positivity
                try:
                    g, _, _, lml = _laplace_mode_constrained(
                        self._K, self._k, self._n, mu_vec,
                        self._C_i, self._C_j, a * self._C_kl, s,
                        sigma_slope=sigma_slope,
                        g_init=g_warm,
                    )
                    g_warm = g
                    return -lml
                except (np.linalg.LinAlgError, ValueError):
                    return 1e10

            from scipy.optimize import minimize as sp_minimize
            res = sp_minimize(
                neg_lml_constraint,
                x0=[0.15, np.log(1.5)],
                method="L-BFGS-B",
                bounds=[(0.01, 1.0), (np.log(0.1), np.log(10.0))],
                options={"maxiter": 100, "ftol": 1e-6},
            )
            alpha = res.x[0]
            sigma_is = np.exp(res.x[1])
            C_kl_effective = alpha * self._C_kl
            sigma_is_vec = None  # use scalar sigma_is
        else:
            C_kl_effective = alpha * self._C_kl if n_constraints > 0 else None
            sigma_is_vec = None

        self.params_ = {
            "variance": variance,
            "lengthscales": lengthscales,
            "mean_intercept": mean_intercept,
            "mean_slope": mean_slope,
            "mean_gamma": mean_gamma,
            "mean_constant": mean_intercept,  # backward compat
            "sigma_is": sigma_is,
            "alpha": alpha,
            "sigma_slope": sigma_slope,
            "n_constraints": n_constraints,
            "two_stage": True,
            "jensen_correction": jensen_correction,
        }

        if n_constraints > 0:
            g_hat, Omega, K_inv, lml = _laplace_mode_constrained(
                self._K, self._k, self._n, mu_vec,
                self._C_i, self._C_j, C_kl_effective,
                sigma_is_vec if sigma_is_vec is not None else sigma_is,
                sigma_slope=0.0 if jensen_correction else sigma_slope,
            )
            self._g_hat = g_hat
            self._K_inv = K_inv
            self._log_marg_lik = lml

            H = Omega + K_inv
            H_inv = np.linalg.inv(H)
            self._A = K_inv - K_inv @ H_inv @ K_inv
            self._alpha_mean = K_inv @ g_hat
        else:
            # No constraints — fall back to unconstrained Laplace
            g_hat, W, L, lml = _laplace_mode(
                self._K, self._k, self._n, mu_vec
            )
            self._g_hat = g_hat
            K_reg = self._K + 1e-6 * np.eye(N)
            self._K_inv = np.linalg.solve(K_reg, np.eye(N))
            self._log_marg_lik = lml
            self._alpha_mean = self._K_inv @ g_hat

            # For predictive variance, use R&W diagonal form
            W_sqrt = np.sqrt(np.maximum(W, 1e-10))
            Omega = np.diag(W)
            H = Omega + self._K_inv
            H_inv = np.linalg.inv(H)
            self._A = self._K_inv - self._K_inv @ H_inv @ self._K_inv

        self._base_gp = base_gp
        return self

    def fit(self, X, successes, trials, constraints, optimize_sigma=True):
        """Fit the constrained GP by jointly optimizing all parameters.

        NOTE: Joint optimization often fails because KL constraints conflict
        with LML at late checkpoints. Prefer fit_two_stage() instead.

        Args:
            X: (N, 2) input features (log_checkpoint, prefill_tokens)
            successes: (N,) success counts per cell
            trials: (N,) trial counts per cell
            constraints: tuple of (C_i, C_j, C_kl) arrays, or list of
                        (idx_pfx0, idx_pfx, kl_value) tuples
            optimize_sigma: if True, learn σ_IS; otherwise fix at 2.0

        Returns:
            self
        """
        self._X = np.asarray(X, dtype=np.float64)
        self._k = np.asarray(successes, dtype=np.float64)
        self._n = np.asarray(trials, dtype=np.float64)
        N, D = self._X.shape

        # Parse constraints (3-tuple: i, j, kl; or 4-tuple: i, j, kl, kl_var)
        if isinstance(constraints, tuple) and len(constraints) in (3, 4):
            self._C_i = np.asarray(constraints[0])
            self._C_j = np.asarray(constraints[1])
            self._C_kl = np.asarray(constraints[2])
            self._C_kl_var = np.asarray(constraints[3]) if len(constraints) == 4 else np.zeros(len(self._C_i))
        else:
            self._C_i = np.array([c[0] for c in constraints])
            self._C_j = np.array([c[1] for c in constraints])
            self._C_kl = np.array([c[2] for c in constraints])
            self._C_kl_var = np.zeros(len(self._C_i))

        n_constraints = len(self._C_i)

        # Data-driven initialization (same strategy as ExploitRateGP)
        emp_rates = np.clip(self._k / np.maximum(self._n, 1), 0.001, 0.999)
        emp_logits = np.log(emp_rates / (1.0 - emp_rates))
        log_var0 = np.log(np.clip(np.var(emp_logits), 1.0, 25.0))
        x_ranges = np.ptp(self._X, axis=0)
        log_ls0 = np.log(np.maximum(x_ranges / 2.0, 0.1))
        log_sigma_is0 = np.log(1.0)  # Base constraint noise
        alpha0 = 0.5  # KL scaling — IS is approximate in logit space
        log_slope0 = np.log(0.1)  # Per-unit-KL noise increase

        # Normalization for tanh mean
        x0 = self._X[:, 0]
        x0_min, x0_max = x0.min(), x0.max()
        self._x0_center = (x0_min + x0_max) / 2.0
        self._x0_scale = max((x0_max - x0_min) / 2.0, 1e-6)
        x0_norm = (x0 - self._x0_center) / self._x0_scale

        # Mean initialization in normalized coords
        if np.ptp(x0_norm) > 0:
            from scipy.stats import linregress
            lr = linregress(x0_norm, emp_logits)
            beta0_init = lr.intercept
            beta1_init = lr.slope
        else:
            beta0_init = float(np.median(emp_logits))
            beta1_init = 0.0
        gamma_init = 0.0

        # Bounds
        var_bounds = (np.log(0.5), np.log(100.0))
        ls_bounds = [
            (np.log(0.05 * max(r, 0.1)), np.log(10.0 * max(r, 1.0)))
            for r in x_ranges
        ]
        beta0_bounds = (-15.0, 10.0)
        beta1_bounds = (-10.0, 10.0)
        gamma_bounds = (0.0, 5.0)
        sigma_is_bounds = (np.log(0.1), np.log(20.0))
        alpha_bounds = (0.01, 2.0)
        slope_bounds = (np.log(0.01), np.log(5.0))

        if optimize_sigma and n_constraints > 0:
            # theta = [log_var, log_ls..., β₀, β₁, γ, log_sigma_is, alpha, log_slope]
            theta0 = np.concatenate([
                [log_var0], log_ls0, [beta0_init, beta1_init, gamma_init,
                                      log_sigma_is0, alpha0, log_slope0],
            ])
            bounds = [var_bounds] + ls_bounds + [
                beta0_bounds, beta1_bounds, gamma_bounds,
                sigma_is_bounds, alpha_bounds, slope_bounds,
            ]
            _opt_constraint = True
        else:
            theta0 = np.concatenate([[log_var0], log_ls0, [beta0_init, beta1_init, gamma_init]])
            bounds = [var_bounds] + ls_bounds + [beta0_bounds, beta1_bounds, gamma_bounds]
            _opt_constraint = False

        g_warm = None

        def neg_lml(theta):
            nonlocal g_warm
            variance = np.exp(theta[0])
            lengthscales = np.exp(theta[1 : 1 + D])
            beta0 = theta[1 + D]
            beta1 = theta[2 + D]
            gamma = theta[3 + D]
            warped = ExploitRateGP._tanh_warp(x0_norm, gamma)
            mu_vec = beta0 + beta1 * warped
            if _opt_constraint:
                sigma_is = np.exp(theta[4 + D])
                alpha = theta[5 + D]
                slope = np.exp(theta[6 + D])
            else:
                sigma_is = 2.0
                alpha = 1.0
                slope = 0.0

            K = matern52_kernel(self._X, self._X, variance, lengthscales)
            K += 1e-6 * variance * np.eye(N)

            try:
                g, _, _, lml = _laplace_mode_constrained(
                    K, self._k, self._n, mu_vec,
                    self._C_i, self._C_j, alpha * self._C_kl, sigma_is,
                    sigma_slope=slope,
                    g_init=g_warm,
                )
                g_warm = g
                return -lml
            except (np.linalg.LinAlgError, ValueError):
                return 1e10

        result = minimize(
            neg_lml, theta0, method="L-BFGS-B", bounds=bounds,
            options={"maxiter": 500, "ftol": 1e-8},
        )

        # Unpack optimized parameters
        variance = np.exp(result.x[0])
        lengthscales = np.exp(result.x[1 : 1 + D])
        beta0 = result.x[1 + D]
        beta1 = result.x[2 + D]
        gamma = result.x[3 + D]
        if _opt_constraint:
            sigma_is = np.exp(result.x[4 + D])
            alpha = result.x[5 + D]
            slope = np.exp(result.x[6 + D])
        else:
            sigma_is = 2.0
            alpha = 1.0
            slope = 0.0

        self.params_ = {
            "variance": variance,
            "lengthscales": lengthscales,
            "mean_intercept": beta0,
            "mean_slope": beta1,
            "mean_gamma": gamma,
            "mean_constant": beta0,  # backward compat
            "sigma_is": sigma_is,
            "alpha": alpha,
            "sigma_slope": slope,
            "n_constraints": n_constraints,
        }

        # Final Laplace at optimum
        warped = ExploitRateGP._tanh_warp(x0_norm, gamma)
        mu_vec = beta0 + beta1 * warped
        self._K = matern52_kernel(self._X, self._X, variance, lengthscales)
        self._K += 1e-6 * variance * np.eye(N)

        g_hat, Omega, K_inv, lml = _laplace_mode_constrained(
            self._K, self._k, self._n, mu_vec,
            self._C_i, self._C_j, alpha * self._C_kl, sigma_is,
            sigma_slope=slope,
        )
        self._g_hat = g_hat
        self._K_inv = K_inv
        self._log_marg_lik = lml

        # Precompute prediction quantities
        H = Omega + K_inv
        H_inv = np.linalg.inv(H)
        self._A = K_inv - K_inv @ H_inv @ K_inv
        self._alpha_mean = K_inv @ g_hat

        return self

    def _mean_function(self, X):
        """Evaluate the tanh mean function at input locations."""
        b0 = self.params_["mean_intercept"]
        b1 = self.params_["mean_slope"]
        gamma = self.params_.get("mean_gamma", 0.0)
        if b1 == 0.0:
            return b0
        x0 = np.asarray(X)[:, 0]
        t_norm = (x0 - self._x0_center) / self._x0_scale
        return b0 + b1 * ExploitRateGP._tanh_warp(t_norm, gamma)

    def predict(self, X_new):
        """Predict exploit rate at new input locations.

        Args:
            X_new: (M, D) new input locations

        Returns:
            dict with rate_mean, rate_lower, rate_upper, f_mean, f_var
        """
        X_new = np.asarray(X_new, dtype=np.float64)
        var = self.params_["variance"]
        ls = self.params_["lengthscales"]
        mu_vec = self._mean_function(X_new)

        k_star = matern52_kernel(X_new, self._X, var, ls)
        k_ss = np.full(len(X_new), var)

        f_mean = k_star @ self._alpha_mean + mu_vec

        v = k_star @ self._A
        f_var = np.maximum(k_ss - np.sum(v * k_star, axis=1), 1e-10)
        f_std = np.sqrt(f_var)

        return {
            "rate_mean": expit(f_mean),
            "rate_lower": expit(f_mean - 1.96 * f_std),
            "rate_upper": expit(f_mean + 1.96 * f_std),
            "f_mean": f_mean,
            "f_var": f_var,
        }

    def predict_surface(self, checkpoints, prefill_levels, log_checkpoint=True):
        """Predict on a grid of (checkpoint, prefill) for surface plotting.

        Same interface as ExploitRateGP.predict_surface.
        """
        ckpts = np.log(checkpoints) if log_checkpoint else np.array(checkpoints)
        pfx = np.array(prefill_levels, dtype=np.float64)
        grid_c, grid_p = np.meshgrid(ckpts, pfx, indexing="ij")
        X_grid = np.column_stack([grid_c.ravel(), grid_p.ravel()])

        preds = self.predict(X_grid)
        n_c, n_p = len(checkpoints), len(prefill_levels)
        result = {
            "checkpoints": np.array(checkpoints),
            "prefill_levels": np.array(prefill_levels),
        }
        for key in ("rate_mean", "rate_lower", "rate_upper", "f_mean", "f_var"):
            result[key] = preds[key].reshape(n_c, n_p)
        return result

    @property
    def lengthscales(self):
        return self.params_["lengthscales"] if self.params_ else None

    @property
    def variance(self):
        return self.params_["variance"] if self.params_ else None

    @property
    def mean_constant(self):
        """Backward compat: returns intercept."""
        return self.params_.get("mean_intercept", self.params_.get("mean_constant")) if self.params_ else None

    @property
    def mean_intercept(self):
        return self.params_.get("mean_intercept", self.params_.get("mean_constant")) if self.params_ else None

    @property
    def mean_slope(self):
        return self.params_.get("mean_slope", 0.0) if self.params_ else None

    @property
    def mean_gamma(self):
        return self.params_.get("mean_gamma", 0.0) if self.params_ else None

    @property
    def sigma_is(self):
        return self.params_["sigma_is"] if self.params_ else None

    @property
    def alpha(self):
        return self.params_["alpha"] if self.params_ else None

    @property
    def log_marginal_likelihood(self):
        return self._log_marg_lik


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------


def prepare_constrained_gp_data(
    evals_dir=None,
    kl_dir=None,
    exploit_type=None,
    intentional_only=True,
    log_checkpoint=True,
    kl_cap=None,
    include_checkpoints=None,
    preloaded_eval_df=None,
    preloaded_kl_df=None,
):
    """Load eval + KL data and build constraint triples for the constrained GP.

    For each (checkpoint, prefill > 0) cell that has KL data, creates a
    pairwise constraint linking it to the corresponding (checkpoint, prefill=0)
    cell.

    Args:
        evals_dir: Path to evals directory
        kl_dir: Path to KL directory (default: evals_dir/../kl)
        exploit_type: if set, filter to this exploit type
        intentional_only: if True, exclude non-intentional exploit types
        log_checkpoint: if True, use log(checkpoint) as first feature
        kl_cap: if set, cap KL values at this maximum (nats). The
                logit ≈ log approximation breaks for large KL, so capping
                prevents nonsensical constraints at high-rate cells.

    Returns:
        X: (N, 2) array of (log_checkpoint, prefill_tokens)
        successes: (N,) success counts
        trials: (N,) trial counts
        constraints: tuple of (C_i, C_j, C_kl) arrays
        meta: DataFrame with checkpoint, prefill_tokens, n_trials, n_successes,
              rate, kl columns
    """
    from pathlib import Path
    from .data import load_per_problem_results, load_kl_results

    if evals_dir is not None:
        evals_dir = Path(evals_dir)
    if kl_dir is None and evals_dir is not None:
        kl_dir = evals_dir.parent / "kl"
    if kl_dir is not None:
        kl_dir = Path(kl_dir)

    # Load and aggregate eval data
    if preloaded_eval_df is not None:
        eval_df = preloaded_eval_df
    else:
        eval_df = load_per_problem_results(evals_dir)
    if include_checkpoints is not None:
        eval_df = eval_df[eval_df["checkpoint"].isin(include_checkpoints)]
    if intentional_only:
        eval_df = eval_df[eval_df["exploit_type"].isin(INTENTIONAL_TYPES)]
    if exploit_type:
        eval_df = eval_df[eval_df["exploit_type"] == exploit_type]
    if len(eval_df) == 0:
        raise ValueError(f"No eval data after filtering (exploit_type={exploit_type})")

    agg = (
        eval_df.groupby(["checkpoint", "prefill_tokens"])
        .agg(n_trials=("exploit_success", "count"), n_successes=("exploit_success", "sum"))
        .reset_index()
    )
    agg["rate"] = agg["n_successes"] / agg["n_trials"]

    # Build X array
    ckpt_vals = (
        np.log(agg["checkpoint"].values) if log_checkpoint
        else agg["checkpoint"].values
    )
    X = np.column_stack([
        ckpt_vals.astype(np.float64),
        agg["prefill_tokens"].values.astype(np.float64),
    ])
    successes = agg["n_successes"].values.astype(np.float64)
    trials = agg["n_trials"].values.astype(np.float64)

    # Build index mapping: (checkpoint, prefill_tokens) → row index
    idx_map = {}
    for idx, row in agg.iterrows():
        idx_map[(row["checkpoint"], row["prefill_tokens"])] = idx

    # Load KL data and build constraint triples
    C_i_list, C_j_list, C_kl_list, C_kl_var_list = [], [], [], []
    if preloaded_kl_df is not None:
        kl_df = preloaded_kl_df
    else:
        kl_df = load_kl_results(kl_dir)

    if kl_df is not None:
        if include_checkpoints is not None:
            kl_df = kl_df[kl_df["checkpoint"].isin(include_checkpoints)]
        if intentional_only:
            kl_df = kl_df[kl_df["exploit_type"].isin(INTENTIONAL_TYPES)]
        if exploit_type:
            kl_df = kl_df[kl_df["exploit_type"] == exploit_type]

        # Mean and variance of KL per (checkpoint, prefill_tokens)
        kl_agg = (
            kl_df.groupby(["checkpoint", "prefill_tokens"])["kl_divergence"]
            .agg(["mean", "var", "count"])
            .reset_index()
        )
        kl_agg.columns = ["checkpoint", "prefill_tokens", "kl_mean", "kl_var", "kl_count"]
        # Fill NaN variance (single sample) with 0
        kl_agg["kl_var"] = kl_agg["kl_var"].fillna(0.0)

        # Create constraint for each (ckpt, pfx>0) → (ckpt, 0)
        for _, kl_row in kl_agg.iterrows():
            ckpt = kl_row["checkpoint"]
            pfx = kl_row["prefill_tokens"]
            kl_val = kl_row["kl_mean"]
            kl_var = kl_row["kl_var"]

            # Skip invalid: pfx=0 (no constraint to self), NaN, or non-positive KL
            if pfx == 0 or np.isnan(kl_val) or kl_val <= 0:
                continue

            idx_pfx0 = idx_map.get((ckpt, 0))
            idx_pfx = idx_map.get((ckpt, pfx))

            if idx_pfx0 is not None and idx_pfx is not None:
                effective_kl = min(kl_val, kl_cap) if kl_cap is not None else kl_val
                C_i_list.append(idx_pfx0)
                C_j_list.append(idx_pfx)
                C_kl_list.append(effective_kl)
                C_kl_var_list.append(float(kl_var))

        # Add KL to meta
        kl_meta = kl_agg[["checkpoint", "prefill_tokens", "kl_mean"]].rename(
            columns={"kl_mean": "kl"}
        )
        agg = agg.merge(kl_meta, on=["checkpoint", "prefill_tokens"], how="left")

    agg["kl"] = agg.get("kl", pd.Series(dtype=float)).fillna(0.0)

    constraints = (
        np.array(C_i_list, dtype=np.intp),
        np.array(C_j_list, dtype=np.intp),
        np.array(C_kl_list, dtype=np.float64),
        np.array(C_kl_var_list, dtype=np.float64),
    )

    return X, successes, trials, constraints, agg


def prepare_offset_gp_data(
    evals_dir,
    kl_dir=None,
    exploit_type=None,
    intentional_only=True,
    log_checkpoint=True,
):
    """Load eval + KL data and aggregate for the offset GP.

    Returns one observation per (checkpoint, prefill_tokens) cell with
    success/trial counts and KL offset (KL=0 for prefill=0).

    Args:
        evals_dir: Path to evals directory
        kl_dir: Path to KL directory (default: evals_dir/../kl)
        exploit_type: if set, filter to this exploit type
        intentional_only: if True, exclude non-intentional exploit types
        log_checkpoint: if True, use log(checkpoint) as input

    Returns:
        X: (N,) array of log(checkpoint) values
        successes: (N,) success counts
        trials: (N,) trial counts
        offsets: (N,) KL divergence offsets
        meta: DataFrame with checkpoint, prefill_tokens, n_trials, n_successes, rate, kl
    """
    from pathlib import Path
    from .data import load_per_problem_results, load_kl_results

    evals_dir = Path(evals_dir)
    if kl_dir is None:
        kl_dir = evals_dir.parent / "kl"
    kl_dir = Path(kl_dir)

    # Load eval data
    eval_df = load_per_problem_results(evals_dir)
    if intentional_only:
        eval_df = eval_df[eval_df["exploit_type"].isin(INTENTIONAL_TYPES)]
    if exploit_type:
        eval_df = eval_df[eval_df["exploit_type"] == exploit_type]
    if len(eval_df) == 0:
        raise ValueError(f"No eval data after filtering (exploit_type={exploit_type})")

    # Aggregate eval data per (checkpoint, prefill_tokens)
    eval_agg = (
        eval_df.groupby(["checkpoint", "prefill_tokens"])
        .agg(n_trials=("exploit_success", "count"), n_successes=("exploit_success", "sum"))
        .reset_index()
    )
    eval_agg["rate"] = eval_agg["n_successes"] / eval_agg["n_trials"]

    # Load KL data
    kl_df = load_kl_results(kl_dir)
    if kl_df is not None:
        if intentional_only:
            kl_df = kl_df[kl_df["exploit_type"].isin(INTENTIONAL_TYPES)]
        if exploit_type:
            kl_df = kl_df[kl_df["exploit_type"] == exploit_type]

        # Aggregate KL per (checkpoint, prefill_tokens): mean across tasks
        kl_agg = (
            kl_df.groupby(["checkpoint", "prefill_tokens"])["kl_divergence"]
            .mean()
            .reset_index()
            .rename(columns={"kl_divergence": "kl"})
        )

        # Merge KL into eval aggregates
        meta = eval_agg.merge(kl_agg, on=["checkpoint", "prefill_tokens"], how="left")
    else:
        meta = eval_agg.copy()

    # KL=0 for prefill=0 (no tokens to diverge on)
    meta["kl"] = meta.get("kl", pd.Series(dtype=float)).fillna(0.0)

    ckpt_vals = np.log(meta["checkpoint"].values) if log_checkpoint else meta["checkpoint"].values.astype(np.float64)
    X = ckpt_vals.astype(np.float64)
    successes = meta["n_successes"].values.astype(np.float64)
    trials = meta["n_trials"].values.astype(np.float64)
    offsets = meta["kl"].values.astype(np.float64)

    return X, successes, trials, offsets, meta


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_gp_surface(gp, checkpoints, prefill_levels, title=None, output_path=None, meta=None):
    """Plot the GP-predicted exploit rate surface as a heatmap.

    Args:
        gp: fitted ExploitRateGP
        checkpoints: array of checkpoint values (original scale)
        prefill_levels: array of prefill token counts
        title: plot title
        output_path: if set, save figure to this path
        meta: optional DataFrame from prepare_gp_data, used to overlay empirical rates
    """
    import matplotlib.pyplot as plt

    surface = gp.predict_surface(checkpoints, prefill_levels)
    rate = surface["rate_mean"]
    ci_width = surface["rate_upper"] - surface["rate_lower"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Rate surface
    im0 = axes[0].imshow(
        rate.T,
        aspect="auto",
        origin="lower",
        cmap="RdYlBu_r",
        vmin=0,
        vmax=1,
        extent=[0, len(checkpoints) - 1, prefill_levels[0], prefill_levels[-1]],
    )
    axes[0].set_xticks(range(len(checkpoints)))
    axes[0].set_xticklabels(checkpoints, rotation=45, fontsize=8)
    axes[0].set_xlabel("Checkpoint")
    axes[0].set_ylabel("Prefill tokens")
    axes[0].set_title("P(exploit) — GP posterior mean")
    plt.colorbar(im0, ax=axes[0], label="Exploit rate")

    # Overlay empirical points
    if meta is not None:
        ckpt_to_idx = {c: i for i, c in enumerate(checkpoints)}
        for _, row in meta.iterrows():
            ci = ckpt_to_idx.get(row["checkpoint"])
            if ci is not None:
                axes[0].plot(
                    ci,
                    row["prefill_tokens"],
                    "ko",
                    markersize=max(1, min(6, row["n_trials"] / 100)),
                    alpha=0.3,
                )

    # CI width surface
    im1 = axes[1].imshow(
        ci_width.T,
        aspect="auto",
        origin="lower",
        cmap="viridis",
        extent=[0, len(checkpoints) - 1, prefill_levels[0], prefill_levels[-1]],
    )
    axes[1].set_xticks(range(len(checkpoints)))
    axes[1].set_xticklabels(checkpoints, rotation=45, fontsize=8)
    axes[1].set_xlabel("Checkpoint")
    axes[1].set_ylabel("Prefill tokens")
    axes[1].set_title("95% CI width")
    plt.colorbar(im1, ax=axes[1], label="CI width")

    suptitle = title or "GP Exploit Rate Surface"
    ls = gp.lengthscales
    mean_slope = getattr(gp, "mean_slope", None) or gp.params_.get("mean_slope", 0.0)
    mean_gamma = getattr(gp, "mean_gamma", None) or gp.params_.get("mean_gamma", 0.0)
    if mean_slope:
        mean_str = f"β₀={gp.mean_constant:.2f}, β₁={mean_slope:.2f}, γ={mean_gamma:.2f}"
    else:
        mean_str = f"μ={gp.mean_constant:.2f}"
    fig.suptitle(
        f"{suptitle}\n"
        f"var={gp.variance:.2f}, ℓ_ckpt={ls[0]:.2f}, ℓ_prefill={ls[1]:.2f}, {mean_str}",
        fontsize=11,
    )
    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()

    return fig


def plot_gp_trajectories(gp, checkpoints, prefill_levels, title=None, output_path=None, meta=None):
    """Plot GP-predicted exploit rate trajectories (rate vs checkpoint at each prefill).

    Args:
        gp: fitted ExploitRateGP
        checkpoints: array of checkpoint values (original scale)
        prefill_levels: list of prefill levels to plot trajectories for
        title: plot title
        output_path: if set, save figure
        meta: optional DataFrame to overlay empirical rates
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 6))
    cmap = plt.cm.viridis
    colors = [cmap(i / max(len(prefill_levels) - 1, 1)) for i in range(len(prefill_levels))]

    surface = gp.predict_surface(checkpoints, prefill_levels)

    for j, pfx in enumerate(prefill_levels):
        rate = surface["rate_mean"][:, j]
        lower = surface["rate_lower"][:, j]
        upper = surface["rate_upper"][:, j]
        ax.plot(checkpoints, rate, color=colors[j], label=f"prefill={pfx}")
        ax.fill_between(checkpoints, lower, upper, color=colors[j], alpha=0.15)

    # Overlay empirical
    if meta is not None:
        for j, pfx in enumerate(prefill_levels):
            sub = meta[meta["prefill_tokens"] == pfx]
            if len(sub) > 0:
                ax.scatter(
                    sub["checkpoint"],
                    sub["rate"],
                    color=colors[j],
                    s=15,
                    alpha=0.6,
                    zorder=5,
                )

    ax.set_xlabel("Checkpoint")
    ax.set_ylabel("P(exploit)")
    ax.set_title(title or "GP Exploit Rate Trajectories")
    ax.legend(fontsize=8, ncol=2)
    ax.set_xscale("log")
    ax.set_ylim(-0.02, 1.02)
    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()

    return fig


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------

INTENTIONAL_TYPES = frozenset({
    "argument_injection_leak",
    "error_code_abuse",
    "import_hook_side_channel",
    "inspect_module_abuse",
    "test_cases_honor_system",
    "trace_profiler_hook_oracle",
    "validator_honor_system",
    "verifier_logic_override",
})


def prepare_gp_data(
    evals_dir,
    exploit_type=None,
    intentional_only=True,
    log_checkpoint=True,
):
    """Load eval data and aggregate for GP fitting.

    Returns one observation per (checkpoint, prefill_tokens) cell with
    success/trial counts.

    Args:
        evals_dir: Path to evals directory (contains checkpoint-*_prefill*.jsonl)
        exploit_type: if set, filter to this exploit type
        intentional_only: if True, exclude non-intentional exploit types
        log_checkpoint: if True, use log(checkpoint) as first feature

    Returns:
        X: (N, 2) array of (log_checkpoint, prefill_tokens)
        successes: (N,) success counts
        trials: (N,) trial counts
        meta: DataFrame with checkpoint, prefill_tokens, n_trials, n_successes, rate
    """
    from pathlib import Path

    from .data import load_per_problem_results

    evals_dir = Path(evals_dir)
    eval_df = load_per_problem_results(evals_dir)

    if intentional_only:
        eval_df = eval_df[eval_df["exploit_type"].isin(INTENTIONAL_TYPES)]
    if exploit_type:
        eval_df = eval_df[eval_df["exploit_type"] == exploit_type]

    if len(eval_df) == 0:
        raise ValueError(f"No data after filtering (exploit_type={exploit_type})")

    agg = (
        eval_df.groupby(["checkpoint", "prefill_tokens"])
        .agg(n_trials=("exploit_success", "count"), n_successes=("exploit_success", "sum"))
        .reset_index()
    )
    agg["rate"] = agg["n_successes"] / agg["n_trials"]

    ckpt_vals = np.log(agg["checkpoint"].values) if log_checkpoint else agg["checkpoint"].values
    X = np.column_stack([ckpt_vals.astype(np.float64), agg["prefill_tokens"].values.astype(np.float64)])
    successes = agg["n_successes"].values.astype(np.float64)
    trials = agg["n_trials"].values.astype(np.float64)

    return X, successes, trials, agg


def compute_gp_features(
    evals_dir,
    intentional_only=True,
):
    """Compute GP-derived features per (exploit_type, checkpoint) for the binary predictor.

    For each exploit type, fits a 2D GP and extracts:
    - gp_rate_p0: GP predicted rate at prefill=0
    - gp_log_rate_p0: log of GP rate at prefill=0
    - gp_rate_max: GP max rate across prefill levels at this checkpoint
    - gp_ci_width_p0: 95% CI width at prefill=0 (uncertainty measure)

    Args:
        evals_dir: Path to evals directory
        intentional_only: if True, only process intentional exploit types

    Returns:
        DataFrame with columns: exploit_type, checkpoint, gp_rate_p0, gp_log_rate_p0,
        gp_rate_max, gp_ci_width_p0
    """
    from pathlib import Path
    evals_dir = Path(evals_dir)

    rows = []
    exploit_types = sorted(INTENTIONAL_TYPES) if intentional_only else None

    if exploit_types is None:
        from .data import load_per_problem_results
        eval_df = load_per_problem_results(evals_dir)
        exploit_types = sorted(eval_df["exploit_type"].dropna().unique())

    for et in exploit_types:
        try:
            X, succ, trial, meta = prepare_gp_data(
                evals_dir, exploit_type=et, intentional_only=False
            )
        except ValueError:
            continue

        gp = ExploitRateGP()
        gp.fit(X, succ, trial)

        checkpoints = sorted(meta["checkpoint"].unique())
        prefill_levels = sorted(meta["prefill_tokens"].unique())

        # Predict at all (checkpoint, prefill) pairs
        surface = gp.predict_surface(checkpoints, prefill_levels)

        # Find prefill=0 column index
        pfx_array = np.array(prefill_levels)
        p0_idx = np.where(pfx_array == 0)[0]
        if len(p0_idx) == 0:
            continue
        p0_idx = p0_idx[0]

        for i, ckpt in enumerate(checkpoints):
            rate_p0 = float(surface["rate_mean"][i, p0_idx])
            lower_p0 = float(surface["rate_lower"][i, p0_idx])
            upper_p0 = float(surface["rate_upper"][i, p0_idx])
            rate_max = float(surface["rate_mean"][i, :].max())

            rows.append({
                "exploit_type": et,
                "checkpoint": ckpt,
                "gp_rate_p0": rate_p0,
                "gp_log_rate_p0": np.log(max(rate_p0, 1e-10)),
                "gp_rate_max": rate_max,
                "gp_ci_width_p0": upper_p0 - lower_p0,
            })

    return pd.DataFrame(rows)


def compute_constrained_gp_features(
    evals_dir,
    kl_dir=None,
    intentional_only=True,
    alpha=0.15,
    sigma_is=1.5,
    optimize_constraints=False,
):
    """Compute IS-constrained GP features per (exploit_type, checkpoint) for the binary predictor.

    Like compute_gp_features(), but uses ExploitRateConstrainedGP.fit_two_stage()
    which propagates signal from high-prefill cells through KL constraints. This
    should produce better pfx=0 estimates at early checkpoints where Laplace is
    prior-dominated.

    Args:
        evals_dir: Path to evals directory
        kl_dir: Path to KL directory (default: evals_dir/../kl)
        intentional_only: if True, only process intentional exploit types
        alpha: KL scaling factor for constraints
        sigma_is: base noise std for pairwise constraints
        optimize_constraints: if True, optimize alpha and sigma_is per exploit type

    Returns:
        DataFrame with columns: exploit_type, checkpoint, gp_rate_p0, gp_log_rate_p0,
        gp_rate_max, gp_ci_width_p0
    """
    from pathlib import Path
    evals_dir = Path(evals_dir)
    if kl_dir is None:
        kl_dir = evals_dir.parent / "kl"
    kl_dir = Path(kl_dir)

    rows = []
    exploit_types = sorted(INTENTIONAL_TYPES) if intentional_only else None

    if exploit_types is None:
        from .data import load_per_problem_results
        eval_df = load_per_problem_results(evals_dir)
        exploit_types = sorted(eval_df["exploit_type"].dropna().unique())

    for et in exploit_types:
        try:
            X, succ, trial, constraints, meta = prepare_constrained_gp_data(
                evals_dir, kl_dir=kl_dir, exploit_type=et, intentional_only=False,
            )
        except ValueError:
            continue

        gp = ExploitRateConstrainedGP()
        gp.fit_two_stage(
            X, succ, trial, constraints,
            alpha=alpha, sigma_is=sigma_is,
            optimize_constraints=optimize_constraints,
        )

        checkpoints = sorted(meta["checkpoint"].unique())
        prefill_levels = sorted(meta["prefill_tokens"].unique())

        surface = gp.predict_surface(checkpoints, prefill_levels)

        pfx_array = np.array(prefill_levels)
        p0_idx = np.where(pfx_array == 0)[0]
        if len(p0_idx) == 0:
            continue
        p0_idx = p0_idx[0]

        for i, ckpt in enumerate(checkpoints):
            rate_p0 = float(surface["rate_mean"][i, p0_idx])
            lower_p0 = float(surface["rate_lower"][i, p0_idx])
            upper_p0 = float(surface["rate_upper"][i, p0_idx])
            rate_max = float(surface["rate_mean"][i, :].max())

            rows.append({
                "exploit_type": et,
                "checkpoint": ckpt,
                "gp_rate_p0": rate_p0,
                "gp_log_rate_p0": np.log(max(rate_p0, 1e-10)),
                "gp_rate_max": rate_max,
                "gp_ci_width_p0": upper_p0 - lower_p0,
            })

    return pd.DataFrame(rows)


def compute_constrained_gp_features_at_cutoff(
    include_checkpoints,
    target_checkpoints,
    evals_dir=None,
    kl_dir=None,
    preloaded_eval_df=None,
    preloaded_kl_df=None,
    intentional_only=True,
    alpha=0.15,
    sigma_is=1.5,
    optimize_constraints=False,
    jensen_correction=False,
):
    """Compute IS-constrained GP features using only data from include_checkpoints.

    Fits one GP per exploit type on data from include_checkpoints only,
    then predicts at prefill=0 for target_checkpoints (which may include
    future checkpoints for extrapolation).

    Args:
        include_checkpoints: List of checkpoints to fit the GP on
        target_checkpoints: List of checkpoints to predict at (can be outside
            include_checkpoints for extrapolation)
        evals_dir: Path to evals directory (if preloaded not provided)
        kl_dir: Path to KL directory (if preloaded not provided)
        preloaded_eval_df: Pre-loaded eval DataFrame (skips disk I/O)
        preloaded_kl_df: Pre-loaded KL DataFrame (skips disk I/O)
        intentional_only: if True, only process intentional exploit types
        alpha: KL scaling factor for constraints
        sigma_is: base noise std for pairwise constraints
        optimize_constraints: if True, optimize alpha and sigma_is per exploit type
        jensen_correction: if True, use per-cell Jensen gap variance correction

    Returns:
        DataFrame with columns: exploit_type, checkpoint, gp_rate_p0, gp_log_rate_p0
    """
    rows = []
    exploit_types = sorted(INTENTIONAL_TYPES) if intentional_only else None

    if exploit_types is None:
        if preloaded_eval_df is not None:
            eval_df = preloaded_eval_df
        else:
            from .data import load_per_problem_results
            eval_df = load_per_problem_results(evals_dir)
        exploit_types = sorted(eval_df["exploit_type"].dropna().unique())

    for et in exploit_types:
        try:
            X, succ, trial, constraints, meta = prepare_constrained_gp_data(
                evals_dir=evals_dir,
                kl_dir=kl_dir,
                exploit_type=et,
                intentional_only=False,
                include_checkpoints=include_checkpoints,
                preloaded_eval_df=preloaded_eval_df,
                preloaded_kl_df=preloaded_kl_df,
            )
        except ValueError:
            continue

        if len(X) < 2:
            continue

        gp = ExploitRateConstrainedGP()
        try:
            gp.fit_two_stage(
                X, succ, trial, constraints,
                alpha=alpha, sigma_is=sigma_is,
                optimize_constraints=optimize_constraints,
                jensen_correction=jensen_correction,
            )
        except Exception:
            continue

        # Predict at (target_checkpoint, prefill=0) for each target
        X_pred = np.array([[np.log(c), 0.0] for c in target_checkpoints])
        preds = gp.predict(X_pred)

        for i, ckpt in enumerate(target_checkpoints):
            rate_p0 = float(preds["rate_mean"][i])
            rows.append({
                "exploit_type": et,
                "checkpoint": ckpt,
                "gp_rate_p0": rate_p0,
                "gp_log_rate_p0": np.log(max(rate_p0, 1e-10)),
            })

    return pd.DataFrame(rows)


def compare_gp_vs_laplace(
    evals_dir,
    kl_dir=None,
    exploit_type=None,
    intentional_only=True,
    laplace_eps=0.5,
):
    """Compare GP exploit rate predictions against Laplace-smoothed rates.

    Fits a GP to the eval data and computes Laplace-smoothed rates from
    compute_pooled_exploit_rate_scaling, then compares them at each
    (checkpoint, prefill_tokens) cell.

    Args:
        evals_dir: Path to evals directory
        kl_dir: Path to KL directory (default: evals_dir/../kl)
        exploit_type: if set, filter to this exploit type
        intentional_only: if True, exclude non-intentional exploit types
        laplace_eps: Laplace smoothing parameter

    Returns:
        dict with keys:
            comparison_df: DataFrame comparing rates at each cell
            gp: fitted ExploitRateGP
            laplace_df: DataFrame from compute_pooled_exploit_rate_scaling
            metrics: dict of summary comparison metrics
    """
    from pathlib import Path
    from .data import load_per_problem_results, load_kl_results
    from .scaling import compute_pooled_exploit_rate_scaling

    evals_dir = Path(evals_dir)
    if kl_dir is None:
        kl_dir = evals_dir.parent / "kl"
    kl_dir = Path(kl_dir)

    # --- GP side ---
    X, successes, trials, meta = prepare_gp_data(
        evals_dir, exploit_type=exploit_type, intentional_only=intentional_only
    )
    gp = ExploitRateGP()
    gp.fit(X, successes, trials)

    # Predict at training points
    preds = gp.predict(X)
    meta = meta.copy()
    meta["gp_rate"] = preds["rate_mean"]
    meta["gp_lower"] = preds["rate_lower"]
    meta["gp_upper"] = preds["rate_upper"]

    # --- Laplace-smoothed side ---
    eval_df = load_per_problem_results(evals_dir)
    kl_df = load_kl_results(kl_dir)

    if intentional_only:
        eval_df = eval_df[eval_df["exploit_type"].isin(INTENTIONAL_TYPES)]
        if kl_df is not None:
            kl_df = kl_df[kl_df["exploit_type"].isin(INTENTIONAL_TYPES)]
    if exploit_type:
        eval_df = eval_df[eval_df["exploit_type"] == exploit_type]
        if kl_df is not None:
            kl_df = kl_df[kl_df["exploit_type"] == exploit_type]

    checkpoints = sorted(meta["checkpoint"].unique())
    laplace_df = None
    if kl_df is not None and len(kl_df) > 0:
        laplace_df = compute_pooled_exploit_rate_scaling(
            kl_df, checkpoints, eval_df=eval_df, laplace_eps=laplace_eps
        )

    # --- Merge for comparison ---
    if laplace_df is not None and len(laplace_df) > 0:
        # Laplace gives one row per checkpoint (pooled across prefills)
        laplace_lookup = dict(zip(laplace_df["checkpoint"], laplace_df["exploit_lower_bound"]))
        meta["laplace_lb"] = meta["checkpoint"].map(laplace_lookup)
    else:
        meta["laplace_lb"] = np.nan

    # Empirical rate
    meta["empirical_rate"] = meta["n_successes"] / meta["n_trials"]

    # Compute residuals
    meta["gp_residual"] = meta["gp_rate"] - meta["empirical_rate"]
    meta["gp_abs_residual"] = np.abs(meta["gp_residual"])

    # Summary metrics
    mae = float(meta["gp_abs_residual"].mean())
    rmse = float(np.sqrt((meta["gp_residual"] ** 2).mean()))

    # Coverage: fraction of empirical rates within GP 95% CI
    in_ci = (meta["empirical_rate"] >= meta["gp_lower"]) & (meta["empirical_rate"] <= meta["gp_upper"])
    coverage = float(in_ci.mean())

    # Zero-cell analysis: how does GP handle cells with 0 successes?
    zero_cells = meta[meta["n_successes"] == 0]
    n_zero = len(zero_cells)
    gp_at_zeros = zero_cells["gp_rate"].mean() if n_zero > 0 else None

    metrics = {
        "mae": mae,
        "rmse": rmse,
        "coverage_95ci": coverage,
        "n_cells": len(meta),
        "n_zero_cells": n_zero,
        "gp_mean_at_zeros": gp_at_zeros,
        "gp_params": gp.params_,
        "log_marginal_likelihood": gp.log_marginal_likelihood,
    }

    return {
        "comparison_df": meta,
        "gp": gp,
        "laplace_df": laplace_df,
        "metrics": metrics,
    }
