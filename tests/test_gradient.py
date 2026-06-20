"""
test_gradient.py — Verify lambda_j_grad and integ_j_grad via finite differences.

Central finite differences on the natural parameters are compared to the
analytical gradient (chain-ruled through the sqrt-parameterization).
"""

import numpy as np
import pytest


def _build_catalog(N=15, seed=42):
    """Create a tiny toy catalog for gradient testing."""
    rng = np.random.default_rng(seed)
    t = np.sort(rng.uniform(0, 50, N))
    x = rng.uniform(0, 1, N)
    y = rng.uniform(0, 1, N)
    m = rng.uniform(0, 2, N)
    bk = np.ones(N) * 0.01
    z = rng.uniform(5, 25, N)
    return t, x, y, z, m, bk


def _identity_norms():
    return {
        'G_norm': 1.0, 'F_norm': None, 'H_norm': 1.0,
        'G_grad': (0.0, 0.0), 'F_grad': (None, None, None), 'H_grad': 0.0,
    }


def test_lambda_j_grad_2d_finite_difference():
    """lambda_j_grad (mver=1, 2D) matches finite differences."""
    from etas.src.lambda_funcs import lambda_j, lambda_j_grad

    t, x, y, z, m, bk = _build_catalog()
    theta = np.sqrt(np.array([0.01, 0.5, 0.01, 1.0, 1.3, 0.005, 2.0, 1.0]))
    j = 5

    _, grad_analytic = lambda_j_grad(theta, j, t, x, y, None, m, bk,
                                     np.inf, np.inf, mver=1)
    h_fd = 1e-5
    grad_fd = np.zeros_like(theta)

    for k in range(len(theta)):
        th_p = theta.copy(); th_p[k] += h_fd
        th_m = theta.copy(); th_m[k] -= h_fd
        fp = lambda_j(th_p, j, t, x, y, None, m, bk, np.inf, np.inf, mver=1)
        fm = lambda_j(th_m, j, t, x, y, None, m, bk, np.inf, np.inf, mver=1)
        grad_fd[k] = (fp - fm) / (2 * h_fd)

    # Relative tolerance (gradients are small for tiny catalogs)
    for k in range(len(theta)):
        denom = max(abs(grad_analytic[k]), abs(grad_fd[k]), 1e-12)
        rel_err = abs(grad_analytic[k] - grad_fd[k]) / denom
        assert rel_err < 1e-4, (
            f"Grad[{k}] mismatch: analytic={grad_analytic[k]:.8e} "
            f"fd={grad_fd[k]:.8e} rel_err={rel_err:.2e}")


def test_lambda_j_grad_3d_finite_difference():
    """lambda_j_grad (mver=1, 3D) matches finite differences."""
    from etas.src.lambda_funcs import lambda_j, lambda_j_grad

    t, x, y, z, m, bk = _build_catalog()
    theta = np.sqrt(np.array([0.01, 0.5, 0.01, 1.0, 1.3, 0.005, 2.0, 1.0, 1.5]))
    tperiod = np.array([5.0, 50.0, 30.0])
    j = 5

    _, grad_analytic = lambda_j_grad(theta, j, t, x, y, z, m, bk,
                                     np.inf, np.inf, mver=1,
                                     is_3d=True, tperiod=tperiod)
    h_fd = 1e-5
    grad_fd = np.zeros_like(theta)

    for k in range(len(theta)):
        th_p = theta.copy(); th_p[k] += h_fd
        th_m = theta.copy(); th_m[k] -= h_fd
        fp = lambda_j(th_p, j, t, x, y, z, m, bk, np.inf, np.inf, mver=1,
                      is_3d=True, tperiod=tperiod)
        fm = lambda_j(th_m, j, t, x, y, z, m, bk, np.inf, np.inf, mver=1,
                      is_3d=True, tperiod=tperiod)
        grad_fd[k] = (fp - fm) / (2 * h_fd)

    for k in range(len(theta)):
        denom = max(abs(grad_analytic[k]), abs(grad_fd[k]), 1e-12)
        rel_err = abs(grad_analytic[k] - grad_fd[k]) / denom
        assert rel_err < 1e-3, (
            f"3D Grad[{k}] mismatch: analytic={grad_analytic[k]:.8e} "
            f"fd={grad_fd[k]:.8e} rel_err={rel_err:.2e}")


def test_lambda_j_grad_with_renorm():
    """Gradient with renorm norms matches finite differences on renormed lambda."""
    from etas.src.lambda_funcs import lambda_j, lambda_j_grad
    from etas.src.renorm import compute_all_norms

    t, x, y, z, m, bk = _build_catalog()
    theta = np.sqrt(np.array([0.01, 0.5, 0.01, 1.0, 1.3, 0.005, 2.0, 1.0]))
    param = theta ** 2
    eps_t, eps_s = 1e-4, 1e-4
    j = 5

    norms = compute_all_norms(param, m, mver=1, eps_t=eps_t, eps_s=eps_s)

    _, grad_analytic = lambda_j_grad(theta, j, t, x, y, None, m, bk,
                                     np.inf, np.inf, mver=1, norms=norms)
    h_fd = 1e-5
    grad_fd = np.zeros_like(theta)

    for k in range(len(theta)):
        th_p = theta.copy(); th_p[k] += h_fd
        th_m = theta.copy(); th_m[k] -= h_fd
        # Recompute norms for each perturbed theta
        norms_p = compute_all_norms(th_p**2, m, mver=1, eps_t=eps_t, eps_s=eps_s)
        norms_m = compute_all_norms(th_m**2, m, mver=1, eps_t=eps_t, eps_s=eps_s)
        fp = lambda_j(th_p, j, t, x, y, None, m, bk, np.inf, np.inf, mver=1, norms=norms_p)
        fm = lambda_j(th_m, j, t, x, y, None, m, bk, np.inf, np.inf, mver=1, norms=norms_m)
        grad_fd[k] = (fp - fm) / (2 * h_fd)

    for k in range(len(theta)):
        denom = max(abs(grad_analytic[k]), abs(grad_fd[k]), 1e-12)
        rel_err = abs(grad_analytic[k] - grad_fd[k]) / denom
        assert rel_err < 1e-3, (
            f"Renorm Grad[{k}] mismatch: analytic={grad_analytic[k]:.8e} "
            f"fd={grad_fd[k]:.8e} rel_err={rel_err:.2e}")
