"""
test_pruning_accuracy.py — Pruned (renormalized) vs full log-likelihood.

With renormalization the pruned model is a genuine approximation of the full
model.  This test asserts that the relative log-likelihood difference stays
below a tolerance for a synthetic catalog.
"""

import numpy as np
import pytest


def test_negloglik_pruned_close_to_full():
    """The pruned renormalized negloglik should be close to the full negloglik."""
    from etas.src.optimizer import _loglkhd
    from etas.src.renorm import compute_all_norms

    np.random.seed(42)
    N = 30
    t = np.sort(np.random.uniform(0, 50, N))
    x = np.random.uniform(0, 1, N)
    y = np.random.uniform(0, 1, N)
    m = np.random.uniform(0, 2, N)
    rpoly = np.array([[0,0],[1,0],[1,1],[0,1],[0,0]], dtype=float)
    tperiod = np.array([5.0, 50.0])

    # revents: [tt, xx, yy, mm, flag, bkgd, prob, lambd]
    revents = np.column_stack([
        t, x, y, m,
        np.ones(N),     # flag=1 (all target)
        np.ones(N) * 0.01,  # bkgd
        np.ones(N),         # prob
        np.ones(N),         # lambd
    ])

    integ0 = 0.5  # rough value for this toy setup
    theta = np.sqrt(np.array([0.01, 0.5, 0.01, 1.0, 1.3, 0.005, 2.0, 1.0]))
    param = theta ** 2

    # Full (no truncation)
    nll_full = _loglkhd(theta, revents, rpoly, tperiod, integ0,
                        mver=1, tau_cut=np.inf, r_cut=np.inf)

    # Pruned with renormalization
    eps_t, eps_s = 1e-4, 1e-4
    norms = compute_all_norms(param, m, mver=1, eps_t=eps_t, eps_s=eps_s)
    tau_cut = norms['T_max']
    r_cut_max = float(np.max(norms['R_max']))
    nll_pruned = _loglkhd(theta, revents, rpoly, tperiod, integ0,
                          mver=1, tau_cut=tau_cut, r_cut=r_cut_max,
                          eps_t=eps_t, eps_s=eps_s, norms=norms)

    # The pruned model is an approximation; relative error should be small.
    rel_err = abs(nll_pruned - nll_full) / max(abs(nll_full), 1e-10)
    assert rel_err < 0.05, (
        f"Pruned negloglik too far from full: full={nll_full:.4f} "
        f"pruned={nll_pruned:.4f} rel_err={rel_err:.4f}")

    # With tighter epsilon the error should shrink.
    eps_t2, eps_s2 = 1e-6, 1e-6
    norms2 = compute_all_norms(param, m, mver=1, eps_t=eps_t2, eps_s=eps_s2)
    nll_pruned2 = _loglkhd(theta, revents, rpoly, tperiod, integ0,
                           mver=1, tau_cut=norms2['T_max'],
                           r_cut=float(np.max(norms2['R_max'])),
                           eps_t=eps_t2, eps_s=eps_s2, norms=norms2)
    rel_err2 = abs(nll_pruned2 - nll_full) / max(abs(nll_full), 1e-10)
    assert rel_err2 <= rel_err + 1e-10, (
        "Tighter epsilon should not increase error")
