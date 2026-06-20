"""
test_renorm.py — Analytic renormalization constants vs numerical integration.

Validates that the closed-form G_norm, F_norm, and their partial derivatives
match scipy.integrate.quad to machine precision (1e-8).
"""

import numpy as np
import pytest
from scipy.integrate import quad

from etas.src.renorm import (temporal_cutoff, temporal_norm, temporal_norm_grad,
                              spatial_cutoff, spatial_norm, spatial_norm_grad,
                              compute_all_norms)


def test_temporal_norm_matches_numerical():
    """G_norm (analytic) == integral_0^T_max g(dt) dt (numerical)."""

    for (c, p, eps) in [(0.005, 1.2, 1e-4),
                         (0.01, 1.3, 1e-5),
                         (0.1, 1.5, 1e-3),
                         (0.02, 2.0, 1e-6)]:
        T = temporal_cutoff(c, p, eps)
        G = temporal_norm(c, p, T_max=T)

        g = lambda dt: (p - 1) / c * (1 + dt / c) ** (-p)
        G_num, _ = quad(g, 0, T)

        assert abs(G - G_num) < 1e-8, (
            f"G_norm mismatch: analytic={G:.10f} numerical={G_num:.10f} "
            f"c={c} p={p} eps={eps}")


def test_spatial_norm_matches_numerical():
    """F_norm(m_j) (analytic) == integral_disk f(r|m_j) r dr dtheta (numerical)."""

    D, gamma, q, eps_s = 0.005, 1.0, 2.0, 1e-4
    m_j = np.array([0.0, 0.5, 1.0, 1.5, 2.0])
    R = spatial_cutoff(D, gamma, q, eps_s, m_j)
    F = spatial_norm(D, gamma, q, m_j, R_max=R)

    for i, mj in enumerate(m_j):
        sig = D * np.exp(gamma * mj)
        # Polar integral: f(r) * 2*pi*r dr
        f_r = lambda r: (2 * np.pi * r * (q - 1) / (np.pi * sig)
                         * (1 + r**2 / sig) ** (-q))
        F_num, _ = quad(f_r, 0, R[i])
        assert abs(F[i] - F_num) < 1e-8, (
            f"F_norm mismatch at m={mj}: analytic={F[i]:.10f} "
            f"numerical={F_num:.10f}")


def test_norms_are_one_when_untruncated():
    """With eps_t=eps_s=None all norms should be exactly 1.0."""

    G = temporal_norm(0.01, 1.3)
    assert G == 1.0, f"G_norm should be 1 when untruncated, got {G}"

    m_j = np.array([0.0, 1.0, 2.0])
    F = spatial_norm(0.005, 1.0, 2.0, m_j)
    assert np.all(F == 1.0), f"F_norm should be 1 when untruncated, got {F}"


def test_truncated_kernel_integrates_to_one():
    """g_trunc(dt) = g(dt)/G_norm must integrate to 1 over [0, T_max]."""

    c, p, eps = 0.01, 1.3, 1e-4
    T = temporal_cutoff(c, p, eps)
    G = temporal_norm(c, p, T_max=T)

    g_trunc = lambda dt: (p - 1) / c * (1 + dt / c) ** (-p) / G
    integral, _ = quad(g_trunc, 0, T)
    assert abs(integral - 1.0) < 1e-10, (
        f"Truncated temporal kernel does not integrate to 1: {integral}")


def test_spatial_truncated_kernel_integrates_to_one():
    """f_trunc(r|m_j) = f(r|m_j)/F_norm(m_j) must integrate to 1 over disk."""

    D, gamma, q, eps_s = 0.005, 1.0, 2.0, 1e-4
    m_j = 0.5
    R = spatial_cutoff(D, gamma, q, eps_s, np.array([m_j]))[0]
    F = spatial_norm(D, gamma, q, np.array([m_j]), R_max=R)[0]

    sig = D * np.exp(gamma * m_j)
    f_trunc = lambda r: (2 * np.pi * r * (q - 1) / (np.pi * sig)
                        * (1 + r**2 / sig) ** (-q) / F)
    integral, _ = quad(f_trunc, 0, R)
    assert abs(integral - 1.0) < 1e-10, (
        f"Truncated spatial kernel does not integrate to 1: {integral}")


def test_temporal_norm_grad_finite_difference():
    """dG/dc and dG/dp match central finite differences."""

    c, p, eps = 0.01, 1.3, 1e-4
    G, dG_dc, dG_dp = temporal_norm_grad(c, p, eps)

    h = 1e-6
    G_cp = temporal_norm(c + h, p, eps)
    G_cm = temporal_norm(c - h, p, eps)
    dG_dc_fd = (G_cp - G_cm) / (2 * h)
    assert abs(dG_dc - dG_dc_fd) < 1e-5, (
        f"dG/dc mismatch: analytic={dG_dc:.8f} fd={dG_dc_fd:.8f}")

    G_pp = temporal_norm(c, p + h, eps)
    G_pm = temporal_norm(c, p - h, eps)
    dG_dp_fd = (G_pp - G_pm) / (2 * h)
    assert abs(dG_dp - dG_dp_fd) < 1e-5, (
        f"dG/dp mismatch: analytic={dG_dp:.8f} fd={dG_dp_fd:.8f}")


def test_spatial_norm_grad_finite_difference():
    """dF/dD, dF/dq, dF/dgamma match central finite differences."""

    D, gamma, q = 0.005, 1.0, 2.0
    eps_s = 1e-4
    m_j = np.array([0.5, 1.5])

    F, dF_dD, dF_dq, dF_dg = spatial_norm_grad(D, gamma, q, m_j, eps_s=eps_s)

    h = 1e-6
    # dF/dD
    F_Dp = spatial_norm(D + h, gamma, q, m_j, eps_s=eps_s)
    F_Dm = spatial_norm(D - h, gamma, q, m_j, eps_s=eps_s)
    dF_dD_fd = (F_Dp - F_Dm) / (2 * h)
    assert np.max(np.abs(dF_dD - dF_dD_fd)) < 1e-5

    # dF/dq
    F_qp = spatial_norm(D, gamma, q + h, m_j, eps_s=eps_s)
    F_qm = spatial_norm(D, gamma, q - h, m_j, eps_s=eps_s)
    dF_dq_fd = (F_qp - F_qm) / (2 * h)
    assert np.max(np.abs(dF_dq - dF_dq_fd)) < 1e-5

    # dF/dgamma
    F_gp = spatial_norm(D, gamma + h, q, m_j, eps_s=eps_s)
    F_gm = spatial_norm(D, gamma - h, q, m_j, eps_s=eps_s)
    dF_dg_fd = (F_gp - F_gm) / (2 * h)
    assert np.max(np.abs(dF_dg - dF_dg_fd)) < 1e-5


def test_compute_all_norms_roundtrip():
    """compute_all_norms returns consistent T_max, R_max, and norm values."""

    param = np.array([0.01, 0.5, 0.01, 1.0, 1.3, 0.005, 2.0, 1.0])
    m = np.array([0.0, 1.0, 2.0])
    eps_t, eps_s = 1e-4, 1e-4

    norms = compute_all_norms(param, m, mver=1, eps_t=eps_t, eps_s=eps_s)

    # Check G_norm
    G_expected = temporal_norm(param[2], param[4], eps_t)
    assert abs(norms['G_norm'] - G_expected) < 1e-12

    # Check F_norm
    F_expected = spatial_norm(param[5], param[7], param[6], m, eps_s=eps_s)
    assert np.max(np.abs(norms['F_norm'] - F_expected)) < 1e-12

    # H_norm should be 1 (no depth truncation)
    assert norms['H_norm'] == 1.0
