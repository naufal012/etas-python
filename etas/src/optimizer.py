"""
Davidon-Fletcher-Powell optimizer for the ETAS model.

Faithful translation of ``fit.c`` and the ``fitfun`` / ``fitfunMP`` methods in
``fitMP.cpp``.  Implements the DFP quasi-Newton optimization with line search
for maximum likelihood estimation of ETAS model parameters.

Extensions over the R reference:

* Renormalization constants (``G_norm``, ``F_norm``, ``H_norm``) and their
  gradients are precomputed **once per likelihood evaluation** from the current
  parameter vector, not inside the per-event loop.
* A single :class:`etas.src.neighbors.NeighborIndex` is built per likelihood
  evaluation when cutoffs are finite, so the intensity sum uses KDTree-pruned
  parent lists instead of O(N^2) distance materialization.
"""

import numpy as np
import math
import sys

from .renorm import compute_all_norms
from .neighbors import NeighborIndex
from .backend import get_xp as _get_xp


def _xp():
    return _get_xp()


def _norm(x):
    """Euclidean norm of a vector."""
    return math.sqrt(sum(xi * xi for xi in x))


class _DataContext:
    """Holds catalog arrays resident on the active backend for the whole fit.

    Built once in :func:`dfp_fit` and passed to every likelihood/gradient/
    line-search evaluation.  This eliminates the per-eval GPU<->CPU round-trip
    that previously happened at the top of ``_loglkhd`` / ``_loglkhd_gr``
    (``revents.get()`` followed by ``xp.asarray(col)`` on every column).
    """

    __slots__ = ('t', 'x', 'y', 'm', 'flag_np', 'bk', 'z',
                 'px', 'py', 'N', 'revents', 'rpoly', 'm_np', 'revents_np')

    def __init__(self, revents, rpoly, is_3d):
        xp = _xp()
        # Keep a CPU view of revents for KDTree construction and indexing.
        self.revents = revents
        self.rpoly = rpoly
        rev_np = revents.get() if hasattr(revents, 'get') else (
            revents if isinstance(revents, np.ndarray) else np.asarray(revents))
        rpo_np = rpoly.get() if hasattr(rpoly, 'get') else (
            rpoly if isinstance(rpoly, np.ndarray) else np.asarray(rpoly))

        self.revents_np = rev_np
        self.N = rev_np.shape[0]
        # Resident backend arrays (GPU when active).
        self.t = xp.asarray(rev_np[:, 0])
        self.x = xp.asarray(rev_np[:, 1])
        self.y = xp.asarray(rev_np[:, 2])
        self.m = xp.asarray(rev_np[:, 3])
        # flag is used in Python `if` comparisons → keep as numpy int array.
        self.flag_np = np.asarray(rev_np[:, 4]).astype(int)
        self.bk = xp.asarray(rev_np[:, 5])
        self.z = xp.asarray(rev_np[:, 8]) if rev_np.shape[1] > 8 else None
        self.m_np = np.asarray(rev_np[:, 3])          # CPU magnitudes for renorm
        self.px = xp.asarray(rpo_np[:, 0])
        self.py = xp.asarray(rpo_np[:, 1])


def _compute_norms(tht, m_np, mver, is_3d, eps_t, eps_s, eps_z, Z_max):
    """Renormalization constants from current ``tht`` (depends on params)."""
    param = tht ** 2
    return compute_all_norms(param, m_np, mver, eps_t=eps_t, eps_s=eps_s,
                             eps_z=eps_z, Z_max=Z_max)


def _build_neighbors(revents_np, tau_cut, r_cut):
    """KDTree neighbor lists.  Depends only on cutoffs + coordinates, NOT on
    parameters — so it should be built once per fit, not per probe.

    Parameters
    ----------
    revents_np : np.ndarray
        CPU-side event array (already extracted via .get() or from ctx).
    """
    use_kd = ((tau_cut is not None and np.isfinite(tau_cut))
              or (r_cut is not None and np.isfinite(r_cut)))
    if not use_kd:
        return None
    nbr_index = NeighborIndex(
        np.asarray(revents_np[:, 1]),
        np.asarray(revents_np[:, 2]),
        np.asarray(revents_np[:, 0]))
    nbr_index.set_cutoffs(tau_cut, r_cut)
    return nbr_index.query_all(tau_cut, r_cut)


def _precompute(tht, revents, mver, is_3d, eps_t, eps_s, eps_z, Z_max, tau_cut, r_cut):
    """Backward-compatible wrapper: renorm + neighbor lists (old call sites).

    Prefer :func:`_compute_norms` + :func:`_build_neighbors` directly.
    """
    rev_np = revents.get() if hasattr(revents, 'get') else (
        revents if isinstance(revents, np.ndarray) else np.asarray(revents))
    m_np = np.asarray(rev_np[:, 3])
    norms = _compute_norms(tht, m_np, mver, is_3d, eps_t, eps_s, eps_z, Z_max)
    nbr_lists = _build_neighbors(rev_np, tau_cut, r_cut)
    return norms, None, nbr_lists


def _loglkhd(tht, revents, rpoly, tperiod, integ0, mver, tau_cut, r_cut,
             is_3d=False, eps_t=None, eps_s=None, eps_z=None,
             norms=None, nbr_lists=None, ctx=None):
    """Minus log-likelihood function of the ETAS model.

    When ``norms`` / ``nbr_lists`` are None they are recomputed from ``tht``;
    supplying them lets callers reuse precomputed state across the function
    and gradient evaluations at the same point.

    When ``ctx`` (a :class:`_DataContext`) is supplied the catalog arrays are
    read from it instead of being re-extracted from ``revents`` on every call
    (avoids GPU<->CPU round-trips).
    """
    from .lambda_funcs import lambda_j, integ_j

    if ctx is not None:
        t = ctx.t; x = ctx.x; y = ctx.y; m = ctx.m; bk = ctx.bk; z = ctx.z
        px = ctx.px; py = ctx.py; flag = ctx.flag_np; N = ctx.N
        # --- Batch path: accumulate GPU results, sync once at end ---
        xp = _xp()
        tstart2 = tperiod[0]
        tlength = tperiod[1]
        Z_max = tperiod[2] if (is_3d and len(tperiod) > 2) else None

        fv1_parts = []
        fv2_parts = []

        for j in range(N):
            if flag[j] == 1:
                nbr_j = nbr_lists[j] if nbr_lists is not None else None
                s = lambda_j(tht, j, t, x, y, z, m, bk, tau_cut, r_cut,
                             mver, is_3d, tperiod, norms=norms, nbr_idx=nbr_j)
                fv1_parts.append(s)
            # integ_j does not need neighbor pruning (polygon integral is exact).
            fv2_j = integ_j(tht, j, t, x, y, m, px, py, tstart2, tlength, mver,
                             is_3d=is_3d, norms=norms, Z_max=Z_max)
            fv2_parts.append(fv2_j)

        # Single sync for fv1: stack, log, sum
        if fv1_parts:
            fv1_arr = xp.stack(fv1_parts)
            valid = fv1_arr > 1.0e-25
            safe = xp.where(valid, fv1_arr, 1.0)
            fv1 = float(xp.sum(xp.where(valid, xp.log(safe), -100.0)))
        else:
            fv1 = 0.0

        # Single sync for fv2
        if fv2_parts:
            fv2 = float(xp.sum(xp.stack(fv2_parts)))
        else:
            fv2 = 0.0
        fv2 += float(tht[0]) * float(tht[0]) * integ0

        return -fv1 + fv2

    # --- Legacy per-event path (no ctx) ---
    xp = _xp()
    if norms is None or nbr_lists is None:
        Z_max = tperiod[2] if (is_3d and len(tperiod) > 2) else None
        norms, _, nbr_lists = _precompute(tht, revents, mver, is_3d,
                                          eps_t, eps_s, eps_z, Z_max,
                                          tau_cut, r_cut)

    revents_np = revents.get() if hasattr(revents, 'get') else (
        revents if isinstance(revents, np.ndarray) else np.asarray(revents))
    N = revents_np.shape[0]
    t = xp.asarray(revents_np[:, 0])
    x = xp.asarray(revents_np[:, 1])
    y = xp.asarray(revents_np[:, 2])
    m = xp.asarray(revents_np[:, 3])
    flag = np.asarray(revents_np[:, 4]).astype(int)
    bk = xp.asarray(revents_np[:, 5])
    z = xp.asarray(revents_np[:, 8]) if revents_np.shape[1] > 8 else None

    rpoly_np = rpoly.get() if hasattr(rpoly, 'get') else (
        rpoly if isinstance(rpoly, np.ndarray) else np.asarray(rpoly))
    px = xp.asarray(rpoly_np[:, 0])
    py = xp.asarray(rpoly_np[:, 1])

    tstart2 = tperiod[0]
    tlength = tperiod[1]
    Z_max = tperiod[2] if (is_3d and len(tperiod) > 2) else None

    fv1 = 0.0
    fv2 = 0.0

    for j in range(N):
        if flag[j] == 1:
            nbr_j = nbr_lists[j] if nbr_lists is not None else None
            s = lambda_j(tht, j, t, x, y, z, m, bk, tau_cut, r_cut,
                         mver, is_3d, tperiod, norms=norms, nbr_idx=nbr_j)
            if s > 1.0e-25:
                fv1 += math.log(float(s))
            else:
                fv1 -= 100.0
        # integ_j does not need neighbor pruning (polygon integral is exact).
        z_j = float(z[j]) if is_3d else None
        fv2 += float(integ_j(tht, j, t, x, y, m, px, py, tstart2, tlength, mver,
                             is_3d=is_3d, norms=norms, z_j=z_j, Z_max=Z_max))

    fv2 += float(tht[0]) * float(tht[0]) * integ0

    return -fv1 + fv2


def _loglkhd_gr(tht, revents, rpoly, tperiod, integ0, mver, tau_cut, r_cut,
                is_3d=False, eps_t=None, eps_s=None, eps_z=None,
                norms=None, nbr_lists=None, ctx=None):
    """Minus log-likelihood and its gradient."""
    from .lambda_funcs import lambda_j_grad, integ_j_grad

    xp = _xp()

    if ctx is not None:
        t = ctx.t; x = ctx.x; y = ctx.y; m = ctx.m; bk = ctx.bk; z = ctx.z
        px = ctx.px; py = ctx.py; flag = ctx.flag_np; N = ctx.N

        # --- Batch path: accumulate GPU results, sync once at end ---
        dimparam = len(tht)
        tstart2 = tperiod[0]
        tlength = tperiod[1]
        Z_max = tperiod[2] if (is_3d and len(tperiod) > 2) else None

        fv1_parts = []
        df1_parts = []
        fv2_parts = []
        df2_parts = []

        for j in range(N):
            if flag[j] == 1:
                nbr_j = nbr_lists[j] if nbr_lists is not None else None
                fv_j, df_j = lambda_j_grad(
                    tht, j, t, x, y, z, m, bk, tau_cut, r_cut,
                    mver, is_3d, tperiod, norms=norms, nbr_idx=nbr_j)
                fv1_parts.append(fv_j)
                df1_parts.append(df_j)

            fv2_j, df2_j = integ_j_grad(
                tht, j, t, x, y, m, px, py, tstart2, tlength, mver,
                is_3d=is_3d, norms=norms, Z_max=Z_max)
            fv2_parts.append(fv2_j)
            df2_parts.append(df2_j)

        # Single sync for fv1 + df1
        if fv1_parts:
            fv1_arr = xp.stack(fv1_parts)
            df1_stack = xp.stack(df1_parts)
            valid = fv1_arr > 1.0e-25
            safe_fv1 = xp.where(valid, fv1_arr, 1.0)
            fv1 = float(xp.sum(xp.where(valid, xp.log(safe_fv1), -100.0)))
            # df1 = sum_j df_j / fv_j for valid events, 0 for invalid
            inv_denom = xp.where(valid[:, None], 1.0 / safe_fv1[:, None], 0.0)
            df1_gpu = xp.sum(df1_stack * inv_denom, axis=0)
            df1 = np.asarray(df1_gpu)
        else:
            fv1 = 0.0
            df1 = np.zeros(dimparam)

        # Single sync for fv2 + df2
        if fv2_parts:
            fv2 = float(xp.sum(xp.stack(fv2_parts)))
            df2 = np.asarray(xp.sum(xp.stack(df2_parts), axis=0))
        else:
            fv2 = 0.0
            df2 = np.zeros(dimparam)
        fv2 += float(tht[0]) * float(tht[0]) * integ0
        df2[0] += integ0 * float(tht[0]) * 2

        fv = -fv1 + fv2
        dfv = -df1 + df2
        return fv, dfv

    # --- Legacy per-event path (no ctx) ---
    if norms is None or nbr_lists is None:
        Z_max = tperiod[2] if (is_3d and len(tperiod) > 2) else None
        norms, _, nbr_lists = _precompute(tht, revents, mver, is_3d,
                                          eps_t, eps_s, eps_z, Z_max,
                                          tau_cut, r_cut)

    revents_np = revents.get() if hasattr(revents, 'get') else (
        revents if isinstance(revents, np.ndarray) else np.asarray(revents))
    N = revents_np.shape[0]
    t = xp.asarray(revents_np[:, 0])
    x = xp.asarray(revents_np[:, 1])
    y = xp.asarray(revents_np[:, 2])
    m = xp.asarray(revents_np[:, 3])
    flag = np.asarray(revents_np[:, 4]).astype(int)
    bk = xp.asarray(revents_np[:, 5])
    z = xp.asarray(revents_np[:, 8]) if revents_np.shape[1] > 8 else None

    rpoly_np = rpoly.get() if hasattr(rpoly, 'get') else (
        rpoly if isinstance(rpoly, np.ndarray) else np.asarray(rpoly))
    px = xp.asarray(rpoly_np[:, 0])
    py = xp.asarray(rpoly_np[:, 1])

    dimparam = len(tht)
    tstart2 = tperiod[0]
    tlength = tperiod[1]
    Z_max = tperiod[2] if (is_3d and len(tperiod) > 2) else None

    fv1 = 0.0
    fv2 = 0.0
    df1 = xp.zeros(dimparam)
    df2 = xp.zeros(dimparam)

    for j in range(N):
        if flag[j] == 1:
            nbr_j = nbr_lists[j] if nbr_lists is not None else None
            fv1_temp, g1_temp = lambda_j_grad(
                tht, j, t, x, y, z, m, bk, tau_cut, r_cut,
                mver, is_3d, tperiod, norms=norms, nbr_idx=nbr_j)
            fv1_temp = float(fv1_temp)
            if fv1_temp > 1.0e-25:
                fv1 += math.log(fv1_temp)
            else:
                fv1 -= 100.0
            for i in range(dimparam):
                df1[i] += float(g1_temp[i]) / fv1_temp if fv1_temp > 1e-25 else 0.0

        z_j = float(z[j]) if is_3d else None
        fv2_temp, g2_temp = integ_j_grad(
            tht, j, t, x, y, m, px, py, tstart2, tlength, mver,
            is_3d=is_3d, norms=norms, z_j=z_j, Z_max=Z_max)
        fv2 += float(fv2_temp)
        for i in range(dimparam):
            df2[i] += float(g2_temp[i])

    fv2 += float(tht[0]) * float(tht[0]) * integ0
    df2[0] += integ0 * float(tht[0]) * 2

    fv = -fv1 + fv2
    dfv = -df1 + df2
    return fv, dfv


def _linesearch(tht, h, fv, ram, dimparam,
                revents, rpoly, tperiod, integ0, mver, tau_cut, r_cut,
                is_3d=False, eps_t=None, eps_s=None, eps_z=None,
                nbr_lists=None, ctx=None):
    """Line search along direction ``h``.

    ``nbr_lists`` (and ``ctx``) are reused across all trial step sizes, so the
    KDTree is built **once per line search**, not once per probe.  Only the
    renormalization constants (which depend on ``tht``) are recomputed inside
    each ``_loglkhd`` call.
    """
    const2 = 1.0e-16

    if ram <= 1.0e-30:
        ram = 0.1

    hnorm = _norm(h)
    if hnorm > 1:
        ram = ram / hnorm

    ram1 = 0.0
    ram2 = ram
    fv1 = fv

    xNew = tht + ram2 * h
    fv2 = _loglkhd(xNew, revents, rpoly, tperiod, integ0, mver, tau_cut, r_cut,
                   is_3d=is_3d, eps_t=eps_t, eps_s=eps_s, eps_z=eps_z,
                   nbr_lists=nbr_lists, ctx=ctx)

    if fv2 > fv1:
        max_iters = 50
        iters = 0
        while iters < max_iters:
            iters += 1
            ram3 = ram2
            fv3 = fv2
            ram2 = ram3 * 0.1
            if ram2 * hnorm < const2:
                return fv, 0.0
            xNew = tht + ram2 * h
            fv2 = _loglkhd(xNew, revents, rpoly, tperiod, integ0, mver, tau_cut, r_cut,
                           is_3d=is_3d, eps_t=eps_t, eps_s=eps_s, eps_z=eps_z,
                           nbr_lists=nbr_lists, ctx=ctx)
            if math.isnan(fv2) or math.isinf(fv2):
                continue
            if fv2 <= fv1:
                break
    else:
        max_iters = 50
        iters = 0
        while iters < max_iters:
            iters += 1
            ram3 = ram2 * 2.0
            xNew = tht + ram3 * h
            fv3 = _loglkhd(xNew, revents, rpoly, tperiod, integ0, mver, tau_cut, r_cut,
                           is_3d=is_3d, eps_t=eps_t, eps_s=eps_s, eps_z=eps_z,
                           nbr_lists=nbr_lists, ctx=ctx)
            if math.isnan(fv3) or math.isinf(fv3):
                break
            if fv3 > fv2:
                break
            ram1 = ram2
            ram2 = ram3
            fv1 = fv2
            fv2 = fv3

    # Parabolic interpolation
    a1 = (ram3 - ram2) * fv1
    a2 = (ram1 - ram3) * fv2
    a3 = (ram2 - ram1) * fv3
    b2 = (a1 + a2 + a3) * 2.0
    b1 = a1 * (ram3 + ram2) + a2 * (ram1 + ram3) + a3 * (ram2 + ram1)

    if math.isnan(b2) or b2 == 0:
        return fv2, ram2

    ram = b1 / b2
    xNew = tht + ram * h
    fv_new = _loglkhd(xNew, revents, rpoly, tperiod, integ0, mver, tau_cut, r_cut,
                      is_3d=is_3d, eps_t=eps_t, eps_s=eps_s, eps_z=eps_z,
                      nbr_lists=nbr_lists, ctx=ctx)

    if math.isnan(fv_new) or math.isinf(fv_new):
        fv_new = fv2
        ram = ram2

    if ram > ram2:
        if fv_new <= fv2:
            ram1 = ram2
            ram2 = ram
            fv1 = fv2
            fv2 = fv_new
        else:
            ram3 = ram
            fv3 = fv_new
    else:
        if fv_new >= fv2:
            ram1 = ram
            fv1 = fv_new
        else:
            ram3 = ram2
            ram2 = ram
            fv3 = fv2
            fv2 = fv_new

    # Second parabolic interpolation
    a1 = (ram3 - ram2) * fv1
    a2 = (ram1 - ram3) * fv2
    a3 = (ram2 - ram1) * fv3
    b2 = (a1 + a2 + a3) * 2.0
    b1 = a1 * (ram3 + ram2) + a2 * (ram1 + ram3) + a3 * (ram2 + ram1)

    if b2 == 0:
        return fv2, ram2

    ram = b1 / b2
    xNew = tht + ram * h
    fv_new = _loglkhd(xNew, revents, rpoly, tperiod, integ0, mver, tau_cut, r_cut,
                      is_3d=is_3d, eps_t=eps_t, eps_s=eps_s, eps_z=eps_z,
                      nbr_lists=nbr_lists, ctx=ctx)

    if fv2 < fv_new:
        ram = ram2
        fv_new = fv2

    return fv_new, ram


def dfp_fit(theta, revents, rpoly, tperiod, integ0, ihess,
            verbose, ndiv, eps, mver, tau_cut, r_cut, is_3d=False,
            eps_t=None, eps_s=None, eps_z=None):
    """DFP quasi-Newton optimization driver."""
    xp = _xp()

    revents_xp = xp.asarray(revents)
    rpoly_xp = xp.asarray(rpoly)

    # Build the resident data context ONCE — arrays stay on the active backend
    # (GPU when active) for the entire fit, eliminating per-eval round-trips.
    ctx = _DataContext(revents_xp, rpoly_xp, is_3d)

    # DFP inner loops use Python-level scalar math (sum(), *, +) on
    # dimparam-sized vectors — keep those on NumPy even when the engine is GPU.
    theta_np = theta.get() if hasattr(theta, 'get') else theta
    tht = np.asarray(theta_np).copy().astype(np.float64)
    dimparam = len(tht)

    if verbose:
        print("\tstart Davidon-Fletcher-Powell procedure ...")

    tau1 = eps
    tau2 = eps
    eps1 = eps
    eps2 = eps
    const1 = 1.0e-17

    ramda = 0.05
    ihess_np = ihess.get() if hasattr(ihess, 'get') else ihess
    h = np.asarray(ihess_np).copy().astype(np.float64)

    s = np.zeros(dimparam)
    dx = np.zeros(dimparam)
    g0 = np.zeros(dimparam)
    dg = np.zeros(dimparam)
    wrk = np.zeros(dimparam)

    Z_max = tperiod[2] if (is_3d and len(tperiod) > 2) else None
    # Neighbor lists depend ONLY on cutoffs + coordinates (constant for the
    # whole fit), so build them ONCE here — not inside every _linesearch probe.
    nbr_lists = _build_neighbors(ctx.revents_np, tau_cut, r_cut)
    norms = _compute_norms(tht, ctx.m_np, mver, is_3d, eps_t, eps_s, eps_z, Z_max)

    fv, g = _loglkhd_gr(tht, revents_xp, rpoly_xp, tperiod, integ0,
                        mver, tau_cut, r_cut, is_3d=is_3d,
                        eps_t=eps_t, eps_s=eps_s, eps_z=eps_z,
                        norms=norms, nbr_lists=nbr_lists, ctx=ctx)
    # DFP inner loops require numpy scalars; convert gradient to numpy.
    g = g.get() if hasattr(g, 'get') else np.asarray(g)

    if verbose:
        print(f"Function Value = {fv:.4f}")
        for i in range(dimparam):
            print(f"Gradient[{i+1}] = {g[i]:.2f}\t"
                  f"theta[{i+1}] = {tht[i]:.6f}")

    def _make_result(tht, fv, g, h):
        return {
            'estimate': tht.copy(),
            'loglik': -fv,
            'gradient': g.copy(),
            'aic': 2.0 * (fv + dimparam),
            'ihessian': h.copy()
        }

    for iteration in range(1, 11):
        for ic in range(dimparam):
            if ic > 0 or iteration > 1:
                dg = g - g0
                for i in range(dimparam):
                    wrk[i] = sum(dg[j] * h[i, j] for j in range(dimparam))
                s1 = sum(wrk[i] * dg[i] for i in range(dimparam))
                s2 = sum(dx[i] * dg[i] for i in range(dimparam))

                if s1 <= const1 or s2 <= const1:
                    if verbose:
                        print(f"loglikelihood = {-fv:.5f}\t"
                              f"AIC = {2*(fv+dimparam):.5f}")
                        for i in range(dimparam):
                            print(f"theta[{i+1}] = {tht[i]**2:.8f}\t"
                                  f"gradient[{i+1}] = {g[i]:.4f}")
                    return _make_result(tht, fv, g, h)

                if s1 <= s2:
                    stem = s1 / s2 + 1.0
                    for i in range(dimparam):
                        for j in range(i, dimparam):
                            h[i, j] -= (dx[i] * wrk[j] + wrk[i] * dx[j] -
                                        dx[i] * dx[j] * stem) / s2
                            h[j, i] = h[i, j]
                else:
                    for i in range(dimparam):
                        for j in range(i, dimparam):
                            h[i, j] += (dx[i] * dx[j] / s2 -
                                        wrk[i] * wrk[j] / s1)
                            h[j, i] = h[i, j]

            ss = 0.0
            for i in range(dimparam):
                total = sum(h[i, j] * g[j] for j in range(dimparam))
                ss += total * total
                s[i] = -total

            s1 = sum(s[i] * g[i] for i in range(dimparam))
            s2 = sum(g[i] * g[i] for i in range(dimparam))

            ds2 = math.sqrt(s2)
            gtem = abs(s1) / ds2 if ds2 > 0 else 0.0
            if gtem <= tau1 and ds2 <= tau2:
                if verbose:
                    print(f"loglikelihood = {-fv:.5f}\t"
                          f"AIC = {2*(fv+dimparam):.5f}")
                    for i in range(dimparam):
                        print(f"theta[{i+1}] = {tht[i]**2:.8f}\t"
                              f"gradient[{i+1}] = {g[i]:.4f}")
                return _make_result(tht, fv, g, h)

            if s1 >= 0:
                h[:, :] = np.eye(dimparam)
                s = -s

            ed = fv
            if verbose:
                print("\nline search along the specified direction ...")

            ed, ramda = _linesearch(
                tht, s, ed, ramda, dimparam,
                revents_xp, rpoly_xp, tperiod, integ0, mver, tau_cut, r_cut,
                is_3d=is_3d, eps_t=eps_t, eps_s=eps_s, eps_z=eps_z,
                nbr_lists=nbr_lists, ctx=ctx)

            if verbose:
                print(f" zeta = {ramda:.6f}")

            s1 = 0.0
            for i in range(dimparam):
                dx[i] = s[i] * ramda
                s1 += dx[i] * dx[i]
                g0[i] = g[i]
                tht[i] += dx[i]

            fv0 = fv
            # Only renorm constants change (depend on tht); neighbor lists are
            # invariant for the whole fit, so reuse them.
            norms = _compute_norms(tht, ctx.m_np, mver, is_3d,
                                   eps_t, eps_s, eps_z, Z_max)
            fv, g = _loglkhd_gr(tht, revents_xp, rpoly_xp, tperiod, integ0,
                                mver, tau_cut, r_cut, is_3d=is_3d,
                                eps_t=eps_t, eps_s=eps_s, eps_z=eps_z,
                                norms=norms, nbr_lists=nbr_lists, ctx=ctx)
            g = g.get() if hasattr(g, 'get') else np.asarray(g)

            if verbose:
                print(f"Function Value = {fv:.4f}")
                for i in range(dimparam):
                    print(f"Gradient[{i+1}] = {g[i]:.2f}\t"
                          f"theta[{i+1}] = {tht[i]:.6f}")

            s2 = sum(g[i] * g[i] for i in range(dimparam))
            if math.sqrt(s2) > tau2:
                continue

            if (fv0 / fv - 1.0 < eps1 and math.sqrt(s1) < eps2):
                if verbose:
                    print(f"loglikelihood = {-fv:.5f}\t"
                          f"AIC = {2*(fv+dimparam):.5f}")
                    for i in range(dimparam):
                        print(f"theta[{i+1}] = {tht[i]**2:.8f}\t"
                              f"gradient[{i+1}] = {g[i]:.4f}")
                return _make_result(tht, fv, g, h)

    return None


def etasfit(theta, revents, rpoly, tperiod, integ0, ihess,
            verbose, ndiv, eps, mver, tau_cut, r_cut, is_3d=False,
            eps_t=None, eps_s=None, eps_z=None):
    """Public entry point: MLE of ETAS parameters via DFP.

    ``theta`` is supplied in natural (un-squared) parameters; we square-root
    internally to match the DFP convention.
    """
    xp = _xp()
    tht = xp.sqrt(xp.asarray(theta))

    cfit = dfp_fit(tht, revents, rpoly, tperiod, integ0, ihess,
                   verbose, ndiv, eps, mver, tau_cut, r_cut, is_3d=is_3d,
                   eps_t=eps_t, eps_s=eps_s, eps_z=eps_z)

    if cfit is None:
        raise RuntimeError(
            "Maximum Likelihood optimization failed to converge.\n"
            "Please try a better starting point.")

    H = cfit['ihessian']
    tht_est = cfit['estimate']

    inv_tht = 1.0 / tht_est
    avcov = 0.25 * xp.diag(inv_tht) @ H @ xp.diag(inv_tht)

    return {
        'estimate': tht_est ** 2,
        'loglik': cfit['loglik'],
        'gradient': cfit['gradient'],
        'aic': cfit['aic'],
        'ihessian': H,
        'avcov': avcov
    }
