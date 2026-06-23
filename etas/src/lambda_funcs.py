"""
Conditional intensity, space-time integral, and renormalized-truncated
kernels for the ETAS model.

Pure-Python/NumPy translation of ``lambda.c`` and ``fitMP.cpp`` (the
``mver=1`` power-law and ``mver=2`` Gaussian paths) from the ETAS R package,
extended with:

* **Analytic renormalization** of the truncated temporal (Omori-Utsu) and
  spatial (power-law) kernels via the closed-form constants in
  :mod:`etas.src.renorm`.  Renormalization is a no-op when no truncation is
  requested, so the un-truncated model is recovered exactly.
* **3D hypocentral extension** via the Beta-density depth kernel
  ``h(u; v) = u^{eta v}(1-u)^{eta(1-v)} / (Z_max B(...))``, threaded through
  the intensity, its gradient, and (new) the space-time integral.

Functions
---------
lambda_j        – conditional intensity at event *j*
lambda_j_grad   – intensity and its gradient at event *j*
integ_j         – space-time integral contribution of event *j*
integ_j_grad    – integral and its gradient for event *j*
lambda_x        – conditional intensity at an arbitrary point

All model parameters ``theta`` are passed as *sqrt*-parameters
(``param = theta**2``); the renormalization dict keys use the *natural*
parameters internally.
"""

import math
import numpy as np

from .poly_integ import (dist_euclidean, dist2_euclidean, poly_integ,
                          fr, dgamma_fr, dD_fr, dq_fr,
                          ffun1, dffun1, ffunrint1, dffunrint1,
                          ffun2, dffun2, ffunrint2, dffunrint2,
                          gfun, dgfun, gfunint, dgfunint,
                          kappafun, dkappafun, pGauss)

from .backend import get_xp


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _norms_or_default(norms):
    """Return a renormalization dict with identity values when ``norms`` is None.

    This is what lets the truncated-model code path collapse exactly onto the
    un-truncated reference implementation when no cutoffs are supplied.
    """
    if norms is None:
        return {
            'G_norm': 1.0,
            'F_norm': None,        # signals "no spatial renorm" -> divide by 1
            'H_norm': 1.0,
            'G_grad': (0.0, 0.0),
            'F_grad': (None, None, None),
            'H_grad': 0.0,
        }
    return norms


def _depth_log_h(u, v, eta, Z_max, special):
    """log of the 3D depth kernel h(u; v), evaluated elementwise on ``v``.

    ``u`` is a scalar (the target event's normalized depth); ``v`` is an
    array of parent normalized depths.  Returns an array the same shape as
    ``v``.  Uses ``gammaln`` for a numerically stable Beta normalization.
    """
    xp = get_xp()
    log_beta = (special.gammaln(eta * v + 1.0)
                + special.gammaln(eta * (1.0 - v) + 1.0)
                - special.gammaln(eta + 2.0))
    # Clamp target-side logs away from 0/1 to avoid -inf; the parent-side
    # exponents stay exact. Use .item() to get a Python float (one sync on GPU,
    # zero cost on CPU).
    u_val = float(u.item()) if hasattr(u, 'item') else float(u)
    safe_u = max(u_val, 1e-12)
    safe_1u = max(1.0 - u_val, 1e-12)
    log_h = ((eta * v) * math.log(safe_u)
             + (eta * (1.0 - v)) * math.log(safe_1u)
             - math.log(Z_max) - log_beta)
    return log_h


# ===================================================================
# Conditional intensity at event j
# ===================================================================

def lambda_j(theta, j, t, x, y, z, m, bk, tau_cut, r_cut,
             mver=1, is_3d=False, tperiod=None, norms=None, nbr_idx=None):
    """Conditional intensity at event *j*.

    Translated from ``clambdaj`` (mver=1) and ``mloglikj2`` (mver=2).  When
    ``nbr_idx`` is supplied the parent-event loop uses it directly (KDTree
    pruned list); otherwise it falls back to the full ``[0, j)`` slice with
    cutoff masks.  Renormalization constants from ``norms`` divide the
    temporal and spatial kernels.

    Parameters
    ----------
    theta : array-like
        Sqrt-parameters (length 8/9 for mver=1, 7/8 for mver=2).
    j : int
        Target event index.
    t, x, y, m : array-like
        Event data (z may be ``None`` when ``is_3d`` is False).
    bk : array-like
        Background rate density at each event.
    tau_cut, r_cut : float or None
        Temporal and spatial cutoffs (informational only when ``nbr_idx`` is
        given; used to build masks on the fallback path).
    mver : {1, 2}
        Model version.
    is_3d : bool
        If True, multiply the spatial kernel by the depth kernel h(u; v).
    tperiod : array-like, optional
        ``[tstart2, tlength, Z_max]`` (``Z_max`` required when ``is_3d``).
    norms : dict, optional
        Renormalization constants from :func:`renorm.compute_all_norms`.
    nbr_idx : array-like, optional
        Precomputed parent indices (e.g. from :class:`neighbors.NeighborIndex`).
    """
    xp = get_xp()
    norms = _norms_or_default(norms)
    G_norm = norms['G_norm']
    F_norm = norms['F_norm']        # array or None
    H_norm = norms['H_norm']

    if j == 0:
        mu = theta[0] * theta[0]
        return mu * bk[j]

    # --- Resolve parent indices ----------------------------------------------
    if nbr_idx is not None:
        idx = np.asarray(nbr_idx, dtype=np.intp)
    else:
        idx = np.arange(j, dtype=np.intp)

    if idx.size == 0:
        mu = theta[0] * theta[0]
        return mu * bk[j]

    ti = t[idx]
    xi = x[idx]
    yi = y[idx]
    mi = m[idx]

    delta = t[j] - ti
    r2 = (x[j] - xi) ** 2 + (y[j] - yi) ** 2

    # When the caller did not pass KDTree indices, apply the cutoff masks
    # explicitly (legacy / un-truncated path).
    if nbr_idx is None:
        if is_3d:
            zi = z[idx]
        if tau_cut is not None and tau_cut < float('inf'):
            delta = delta[delta <= tau_cut]
            # recompute r2 / mi on the same mask
            mask_t = (t[j] - ti) <= tau_cut
            r2 = r2[mask_t]
            mi = mi[mask_t]
            ti = ti[mask_t]
            xi = xi[mask_t]
            yi = yi[mask_t]
            if is_3d:
                zi = zi[mask_t]
        if r_cut is not None and r_cut < float('inf'):
            mask_r = xp.sqrt(r2) <= r_cut
            delta = delta[mask_r]
            r2 = r2[mask_r]
            mi = mi[mask_r]
            if is_3d:
                zi = zi[mask_r]
        if delta.size == 0:
            mu = theta[0] * theta[0]
            return mu * bk[j]
    else:
        if is_3d:
            zi = z[idx]

    if mver == 1:
        # --- power-law spatial kernel, 8 parameters ---
        mu    = theta[0] * theta[0]
        A     = theta[1] * theta[1]
        c     = theta[2] * theta[2]
        alpha = theta[3] * theta[3]
        p     = theta[4] * theta[4]
        D     = theta[5] * theta[5]
        q     = theta[6] * theta[6]
        gamma = theta[7] * theta[7]

        s = mu * bk[j]
        if delta.size == 0:
            return s   # GPU scalar, no sync

        part1 = xp.exp(alpha * mi)
        part2 = (p - 1.0) / c * (1.0 + delta / c) ** (-p)
        sig = D * xp.exp(gamma * mi)
        part3 = ((q - 1.0) / (sig * xp.pi) * (1.0 + r2 / sig) ** (-q))

        # Renormalization (no-op when G_norm == F_norm == 1).
        part2 = part2 / G_norm
        if F_norm is not None:
            part3 = part3 / xp.asarray(F_norm)[idx] if _is_xp_array(F_norm, xp) \
                else part3 / np.asarray(F_norm)[idx]

        if is_3d:
            from .backend import get_special
            special = get_special()
            Z_max = tperiod[2]
            eta = theta[8] * theta[8]
            u = z[j] / Z_max
            v = zi / Z_max
            log_h = _depth_log_h(u, v, eta, Z_max, special)
            part3 = part3 * xp.exp(log_h) / H_norm

        s += xp.sum(A * part1 * part2 * part3)
        return s   # GPU scalar, no sync

    elif mver == 2:
        # --- Gaussian spatial kernel, 7 parameters ---
        mu    = theta[0] * theta[0]
        A     = theta[1] * theta[1]
        c     = theta[2] * theta[2]
        alpha = theta[3] * theta[3]
        p     = theta[4] * theta[4]
        D     = theta[5] * theta[5]
        gamma = theta[6] * theta[6]

        s = mu * bk[j]
        if delta.size == 0:
            return s

        k_val = A * xp.exp(alpha * mi)
        g_val = (p - 1.0) / c * (1.0 + delta / c) ** (-p)
        sig_sq = D * xp.exp(gamma * mi)
        f_val = xp.exp(-r2 / (2.0 * sig_sq)) / (2.0 * xp.pi * sig_sq)

        g_val = g_val / G_norm
        # Gaussian spatial kernel is normalized inside the polygon integral;
        # there is no per-event F_norm for mver=2.

        if is_3d:
            from .backend import get_special
            special = get_special()
            Z_max = tperiod[2]
            eta = theta[7] * theta[7]
            u = z[j] / Z_max
            v = zi / Z_max
            log_h = _depth_log_h(u, v, eta, Z_max, special)
            f_val = f_val * xp.exp(log_h) / H_norm

        s += xp.sum(k_val * g_val * f_val)
        return s

    else:
        raise ValueError(f"Unknown mver={mver}; expected 1 or 2")


def _is_xp_array(arr, xp):
    """True if ``arr`` is already an array on the active backend."""
    if xp.__name__ == 'numpy':
        return isinstance(arr, np.ndarray)
    try:
        import cupy
        return isinstance(arr, cupy.ndarray)
    except ImportError:
        return False


# ===================================================================
# Conditional intensity gradient at event j
# ===================================================================

def lambda_j_grad(theta, j, t, x, y, z, m, bk, tau_cut, r_cut,
                  mver=1, is_3d=False, tperiod=None, norms=None, nbr_idx=None):
    """Conditional intensity and its gradient at event *j*.

    See :func:`lambda_j` for the renormalization / neighbor-index conventions.

    Returns
    -------
    fv : float
        Conditional intensity at event *j*.
    dfv : np.ndarray
        Gradient w.r.t. ``theta`` (the sqrt-parameters), with chain-rule
        factor ``2 * theta[k]`` applied.  Length 8/9 (mver=1) or 7/8 (mver=2).
    """
    xp = get_xp()
    norms = _norms_or_default(norms)
    G_norm = norms['G_norm']
    F_norm = norms['F_norm']
    H_norm = norms['H_norm']
    G_grad = norms['G_grad']              # (dG/dc, dG/dp)
    F_grad = norms['F_grad']              # (dF/dD, dF/dq, dF/dgamma), arrays or None
    H_grad = norms['H_grad']

    if j == 0:
        mu = theta[0] * theta[0]
        nparam = (9 if is_3d else 8) if mver == 1 else (8 if is_3d else 7)
        dfv = xp.zeros(nparam)
        dfv[0] = bk[j] * 2.0 * theta[0]
        return mu * bk[j], dfv

    if nbr_idx is not None:
        idx = np.asarray(nbr_idx, dtype=np.intp)
    else:
        idx = np.arange(j, dtype=np.intp)

    if mver == 1:
        # --- power-law, 8 parameters ---
        mu    = theta[0] * theta[0]
        A     = theta[1] * theta[1]
        c     = theta[2] * theta[2]
        alpha = theta[3] * theta[3]
        p     = theta[4] * theta[4]
        D     = theta[5] * theta[5]
        q     = theta[6] * theta[6]
        gamma = theta[7] * theta[7]

        s = mu * bk[j]
        sg1 = bk[j]
        sg2 = sg3 = sg4 = sg5 = sg6 = sg7 = sg8 = sg9 = 0.0

        if idx.size > 0:
            ti = t[idx]; xi = x[idx]; yi = y[idx]; mi = m[idx]
            if is_3d:
                zi = z[idx]
            delta = t[j] - ti
            r2 = (x[j] - xi) ** 2 + (y[j] - yi) ** 2

            # Legacy mask path when no KDTree indices supplied.
            if nbr_idx is None:
                if tau_cut is not None and tau_cut < float('inf'):
                    m_t = (t[j] - ti) <= tau_cut
                    delta = delta[m_t]; r2 = r2[m_t]; mi = mi[m_t]
                    if is_3d: zi = zi[m_t]
                if r_cut is not None and r_cut < float('inf'):
                    m_r = xp.sqrt(r2) <= r_cut
                    delta = delta[m_r]; r2 = r2[m_r]; mi = mi[m_r]
                    if is_3d: zi = zi[m_r]

            if delta.size > 0:
                part1 = xp.exp(alpha * mi)
                part2 = (p - 1.0) / c * (1.0 + delta / c) ** (-p)
                sig = D * xp.exp(gamma * mi)
                part3 = ((q - 1.0) / (sig * xp.pi) * (1.0 + r2 / sig) ** (-q))

                # Apply renormalization (1/X form) and cache the un-renormed
                # kernel pieces so we can build the denominator derivatives.
                invG = 1.0 / G_norm
                part2_r = part2 * invG

                F_at = None
                if F_norm is not None:
                    if _is_xp_array(F_norm, xp):
                        F_at = xp.asarray(F_norm)[idx]
                    else:
                        F_at = np.asarray(F_norm, dtype=np.float64)[idx]
                    # if we masked above, F_at may need the same mask; safer
                    # to recompute via lookup using the surviving mi values.
                if F_norm is not None and F_at is not None and F_at.shape[0] != part3.shape[0]:
                    # Fallback: recompute F_norm on the surviving magnitudes
                    # using the closed form (cheap; N is small after pruning).
                    from .renorm import spatial_norm
                    F_at = spatial_norm(D, gamma, q, mi,
                                        R_max=np.sqrt(r2.max()) if r2.size else 0.0)
                if F_norm is not None:
                    invF = 1.0 / xp.asarray(F_at, dtype=part3.dtype) if _is_xp_array(F_at, xp) \
                        else 1.0 / np.asarray(F_at, dtype=np.float64)
                    part3_r = part3 * invF
                else:
                    invF = 1.0
                    part3_r = part3

                h_val = None
                if is_3d:
                    from .backend import get_special
                    special = get_special()
                    Z_max = tperiod[2]
                    eta = theta[8] * theta[8]
                    u = z[j] / Z_max
                    v = zi / Z_max
                    log_h = _depth_log_h(u, v, eta, Z_max, special)
                    h_val = xp.exp(log_h) / H_norm
                    part3_r = part3_r * h_val

                # hfactor multiplies the spatial derivatives (1 in 2D, h_val in 3D).
                hfactor = h_val if is_3d else 1.0

                # Intensity
                kA = A * part1
                kern = part2_r * part3_r
                s += xp.sum(kA * kern)

                # --- gradient contributions --------------------------------
                # d/dA
                sg2 = xp.sum(part1 * kern)
                # d/dc  (kernel part2 derivative + G_norm denominator term)
                # part2 = (p-1)/c * (1+delta/c)^(-p)
                # d(part2)/dc = part2 * [-1/c + p*delta/(c*(c+delta))]
                part2_dc = part2 * (-1.0 / c + p * delta / (c * (c + delta)))
                dG_dc, dG_dp = G_grad
                # d(invG)/dc = -invG^2 * dG/dc
                # kA = A * part1 already; do NOT double-count part1.
                # We fold the c-derivative through part2_r and the G_norm term:
                sg3 = xp.sum(kA * (part2_dc * invG - part2 * invG * invG * dG_dc) * part3_r)
                # d/dalpha
                sg4 = xp.sum(A * part1 * mi * kern)
                # d/dp
                part2_dp = part2 * (1.0 / (p - 1.0) - xp.log(1.0 + delta / c))
                sg5 = xp.sum(kA * (part2_dp * invG - part2 * invG * invG * dG_dp) * part3_r)
                # spatial derivatives — these reconstruct d(part3_r)/dparam, so we
                # must fold in hfactor (h_val) which was baked into part3_r above.
                dF_dD, dF_dq, dF_dgamma = F_grad
                # d(part3_r)/dD = (d(part3)/dD / F - part3 * dF/dD / F^2) * hfactor
                part3_dD = (part3 / D * (-1.0 + q * (1.0 - 1.0 / (1.0 + r2 / sig))))
                if F_norm is not None and dF_dD is not None:
                    dF_D_at = xp.asarray(np.asarray(dF_dD)[idx], dtype=part3.dtype) \
                        if not _is_xp_array(dF_dD, xp) else dF_dD[idx]
                    sg6 = xp.sum(kA * part2_r * hfactor * (
                        part3_dD * invF - part3 * invF * invF * dF_D_at))
                else:
                    sg6 = xp.sum(kA * part2_r * hfactor * part3_dD)
                # d/dq
                part3_dq = part3 * (1.0 / (q - 1.0) - xp.log(1.0 + r2 / sig))
                if F_norm is not None and dF_dq is not None:
                    dF_q_at = xp.asarray(np.asarray(dF_dq)[idx], dtype=part3.dtype) \
                        if not _is_xp_array(dF_dq, xp) else dF_dq[idx]
                    sg7 = xp.sum(kA * part2_r * hfactor * (
                        part3_dq * invF - part3 * invF * invF * dF_q_at))
                else:
                    sg7 = xp.sum(kA * part2_r * hfactor * part3_dq)
                # d/dgamma
                part3_dgamma = part3 * (-mi + q * mi * (1.0 - 1.0 / (1.0 + r2 / sig)))
                if F_norm is not None and dF_dgamma is not None:
                    dF_g_at = xp.asarray(np.asarray(dF_dgamma)[idx], dtype=part3.dtype) \
                        if not _is_xp_array(dF_dgamma, xp) else dF_dgamma[idx]
                    sg8 = xp.sum(kA * part2_r * hfactor * (
                        part3_dgamma * invF - part3 * invF * invF * dF_g_at))
                else:
                    sg8 = xp.sum(kA * part2_r * hfactor * part3_dgamma)

                # 3D: depth kernel gradient wrt eta
                if is_3d:
                    from .backend import get_special
                    special = get_special()
                    Z_max = tperiod[2]
                    eta = theta[8] * theta[8]
                    u = z[j] / Z_max
                    v = zi / Z_max
                    # dh/deta = h * ( v*log u + (1-v)*log(1-u)
                    #                 - [ v*psi(eta v+1) + (1-v)*psi(eta(1-v)+1) - psi(eta+2) ] )
                    # h_val already includes /H_norm, so dh/deta = h_val * (...)  with NO extra /H_norm.
                    u_val = float(u.item()) if hasattr(u, 'item') else float(u)
                    safe_u = max(u_val, 1e-12)
                    safe_1u = max(1.0 - u_val, 1e-12)
                    digamma_term = (v * special.digamma(eta * v + 1.0)
                                    + (1.0 - v) * special.digamma(eta * (1.0 - v) + 1.0)
                                    - special.digamma(eta + 2.0))
                    dh_deta = h_val * (v * math.log(safe_u)
                                       + (1.0 - v) * math.log(safe_1u)
                                       - digamma_term)
                    # kern = part2_r * part3 * invF * h_val ; d(kern)/deta only hits h_val
                    sg9 = xp.sum(kA * part2_r * part3 * invF * dh_deta
                                 if F_norm is not None else
                                 kA * part2_r * part3 * dh_deta)

        fv = s
        nparam = 9 if is_3d else 8
        dfv = xp.empty(nparam)
        dfv[0] = sg1 * 2.0 * theta[0]
        dfv[1] = sg2 * 2.0 * theta[1]
        dfv[2] = sg3 * 2.0 * theta[2]
        dfv[3] = sg4 * 2.0 * theta[3]
        dfv[4] = sg5 * 2.0 * theta[4]
        dfv[5] = sg6 * 2.0 * theta[5]
        dfv[6] = sg7 * 2.0 * theta[6]
        dfv[7] = sg8 * 2.0 * theta[7]
        if is_3d:
            dfv[8] = sg9 * 2.0 * theta[8]
        return fv, dfv

    elif mver == 2:
        # --- Gaussian, 7 parameters ---
        mu    = theta[0] * theta[0]
        A     = theta[1] * theta[1]
        c     = theta[2] * theta[2]
        alpha = theta[3] * theta[3]
        p     = theta[4] * theta[4]
        D     = theta[5] * theta[5]
        gamma = theta[6] * theta[6]

        s = mu * bk[j]
        sg = xp.zeros(8 if is_3d else 7)
        sg[0] = bk[j]

        if idx.size > 0:
            ti = t[idx]; xi = x[idx]; yi = y[idx]; mi = m[idx]
            if is_3d:
                zi = z[idx]
            delta = t[j] - ti
            r2 = (x[j] - xi) ** 2 + (y[j] - yi) ** 2

            if nbr_idx is None:
                if tau_cut is not None and tau_cut < float('inf'):
                    m_t = (t[j] - ti) <= tau_cut
                    delta = delta[m_t]; r2 = r2[m_t]; mi = mi[m_t]
                    if is_3d: zi = zi[m_t]
                if r_cut is not None and r_cut < float('inf'):
                    m_r = xp.sqrt(r2) <= r_cut
                    delta = delta[m_r]; r2 = r2[m_r]; mi = mi[m_r]
                    if is_3d: zi = zi[m_r]

            if delta.size > 0:
                k_val = A * xp.exp(alpha * mi)
                g_val = (p - 1.0) / c * (1.0 + delta / c) ** (-p)
                sig_sq = D * xp.exp(gamma * mi)
                f_val = xp.exp(-r2 / (2.0 * sig_sq)) / (2.0 * xp.pi * sig_sq)

                invG = 1.0 / G_norm
                g_val_r = g_val * invG

                h_val = None
                if is_3d:
                    from .backend import get_special
                    special = get_special()
                    Z_max = tperiod[2]
                    eta = theta[7] * theta[7]
                    u = z[j] / Z_max
                    v = zi / Z_max
                    log_h = _depth_log_h(u, v, eta, Z_max, special)
                    h_val = xp.exp(log_h) / H_norm
                    f_val = f_val * h_val

                kern = g_val_r * f_val
                s += xp.sum(k_val * kern)

                sg[1] = xp.sum(xp.exp(alpha * mi) * kern)

                # g_val = (p-1)/c * (1+delta/c)^(-p)
                # d(g_val)/dc = g_val * [-1/c + p*delta/(c*(c+delta))]
                g_val_dc = g_val * (-1.0 / c + p * delta / (c * (c + delta)))
                dG_dc, dG_dp = G_grad
                sg[2] = xp.sum(k_val * (g_val_dc * invG - g_val * invG * invG * dG_dc) * f_val)

                sg[3] = xp.sum(A * mi * xp.exp(alpha * mi) * kern)

                g_val_dp = g_val * (1.0 / (p - 1.0) - xp.log(1.0 + delta / c))
                sg[4] = xp.sum(k_val * (g_val_dp * invG - g_val * invG * invG * dG_dp) * f_val)

                f_val_D = f_val * (r2 / (2.0 * sig_sq * D) - 1.0 / D)
                sg[5] = xp.sum(k_val * g_val_r * f_val_D)

                f_val_gamma = f_val * (r2 / (2.0 * sig_sq) - 1.0) * mi
                sg[6] = xp.sum(k_val * g_val_r * f_val_gamma)

                if is_3d:
                    from .backend import get_special
                    special = get_special()
                    Z_max = tperiod[2]
                    eta = theta[7] * theta[7]
                    u = z[j] / Z_max
                    v = zi / Z_max
                    u_val = float(u.item()) if hasattr(u, 'item') else float(u)
                    safe_u = max(u_val, 1e-12)
                    safe_1u = max(1.0 - u_val, 1e-12)
                    digamma_term = (v * special.digamma(eta * v + 1.0)
                                    + (1.0 - v) * special.digamma(eta * (1.0 - v) + 1.0)
                                    - special.digamma(eta + 2.0))
                    dh_deta = h_val * H_norm * (v * math.log(u_float)
                                                + (1.0 - v) * math.log(safe_1u)
                                                - digamma_term)
                    # h_val above already includes /H_norm; restore raw h for the
                    # chain rule and divide H_norm back after differentiation.
                    sg[7] = xp.sum(k_val * g_val_r * (f_val / h_val) * dh_deta)

        fv = s
        nparam = 8 if is_3d else 7
        dfv = xp.empty(nparam)
        for k in range(7):
            dfv[k] = sg[k] * 2.0 * theta[k]
        if is_3d:
            dfv[7] = sg[7] * 2.0 * theta[7]
        return fv, dfv

    else:
        raise ValueError(f"Unknown mver={mver}; expected 1 or 2")


# ===================================================================
# Space-time integral contribution of event j
# ===================================================================

def integ_j(theta, j, t, x, y, m, px, py, tstart2, tlength,
            mver=1, is_3d=False, norms=None, z_j=None, Z_max=None,
            integ_h=None):
    """Space-time integral contribution of event *j*.

    The integral is the expected number of triggered events generated by
    parent *j* that fall inside the study window.  For the 3D model we now
    also fold in the depth CDF ``integ_h`` (integral of ``h(u; v_j)`` over
    ``[0, Z_max]``, which equals 1 by construction and is supplied as a hook).

    Renormalization scales the temporal integral by ``1/G_norm`` and the
    spatial integral by ``1/F_norm(m_j)`` (mver=1 only).

    Parameters
    ----------
    z_j, Z_max : optional
        Depth and layer thickness of event *j* (required when ``is_3d``).
    integ_h : float, optional
        Depth-axis integral of ``h`` (default 1.0).
    """
    xp = get_xp()
    norms = _norms_or_default(norms)
    G_norm = norms['G_norm']
    F_norm = norms['F_norm']
    H_norm = norms['H_norm']

    if integ_h is None:
        integ_h = 1.0

    if mver == 1:
        A     = theta[1] * theta[1]
        c     = theta[2] * theta[2]
        alpha = theta[3] * theta[3]
        p     = theta[4] * theta[4]
        D     = theta[5] * theta[5]
        q     = theta[6] * theta[6]
        gamma = theta[7] * theta[7]

        # --- temporal integral (Omori-Utsu CDF over the study window) ------
        # Extract CPU scalar once: one sync on GPU, zero cost on CPU.
        t_j = float(t[j])
        if t_j > tstart2:
            ttemp = tlength - t_j
            gi = 1.0 - (1.0 + ttemp / c) ** (1.0 - p)
        else:
            ttemp1 = tstart2 - t_j
            ttemp2 = tlength - t_j
            gi1 = 1.0 - (1.0 + ttemp1 / c) ** (1.0 - p)
            gi2 = 1.0 - (1.0 + ttemp2 / c) ** (1.0 - p)
            gi = gi2 - gi1
        gi = gi / G_norm

        # --- spatial integral over polygon ----------------------------------
        w = [gamma, D, q, m[j]]
        si = poly_integ(fr, w, px, py, x[j], y[j])
        if F_norm is not None:
            # Keep as GPU scalar — no sync needed.
            F_at = F_norm[j]
            si = si / F_at

        sk = A * xp.exp(alpha * m[j])

        out = sk * gi * si
        if is_3d:
            out = out * integ_h / H_norm
        return out

    elif mver == 2:
        A     = theta[1] * theta[1]
        c     = theta[2] * theta[2]
        alpha = theta[3] * theta[3]
        p     = theta[4] * theta[4]
        D     = theta[5] * theta[5]
        gamma = theta[6] * theta[6]

        kparam = [A, alpha]
        gparam = [c, p]
        fparam = [D, gamma]

        t_j = float(t[j])
        gi = gfunint(tlength - t_j, gparam)
        if t_j <= tstart2:
            gi -= gfunint(tstart2 - t_j, gparam)
        gi = gi / G_norm

        m_j = m[j]  # hoist: access once, not per polygon edge
        def _ffunrint2_wrap(r, w_unused):
            return ffunrint2(r, m_j, fparam)
        si = poly_integ(_ffunrint2_wrap, None, px, py, x[j], y[j])

        sk = kappafun(m_j, kparam)

        out = sk * gi * si
        if is_3d:
            out = out * integ_h / H_norm
        return out

    else:
        raise ValueError(f"Unknown mver={mver}; expected 1 or 2")


# ===================================================================
# Space-time integral gradient for event j
# ===================================================================

def integ_j_grad(theta, j, t, x, y, m, px, py, tstart2, tlength,
                 mver=1, is_3d=False, norms=None, z_j=None, Z_max=None,
                 integ_h=None):
    """Space-time integral and its gradient for event *j*.

    See :func:`integ_j` for conventions.  Gradient includes the renormalization
    denominator derivatives (chain rule) for ``G_norm`` (mver=1,2) and
    ``F_norm`` (mver=1).
    """
    xp = get_xp()
    norms = _norms_or_default(norms)
    G_norm = norms['G_norm']
    F_norm = norms['F_norm']
    H_norm = norms['H_norm']
    G_grad = norms['G_grad']            # (dG/dc, dG/dp)
    F_grad = norms['F_grad']            # (dF/dD, dF/dq, dF/dgamma)
    if integ_h is None:
        integ_h = 1.0

    if mver == 1:
        A     = theta[1] * theta[1]
        c     = theta[2] * theta[2]
        alpha = theta[3] * theta[3]
        p     = theta[4] * theta[4]
        D     = theta[5] * theta[5]
        q     = theta[6] * theta[6]
        gamma = theta[7] * theta[7]

        # temporal integral + derivatives (extract CPU scalar once)
        t_j = float(t[j])
        if t_j > tstart2:
            ttemp = tlength - t_j
            gi  = 1.0 - (1.0 + ttemp / c) ** (1.0 - p)
            gic = -(1.0 - gi) * (1.0 - p) * (1.0 / (c + ttemp) - 1.0 / c)
            gip = -(1.0 - gi) * (math.log(c) - math.log(c + ttemp))
        else:
            ttemp1 = tstart2 - t_j
            ttemp2 = tlength - t_j
            gi1  = 1.0 - (1.0 + ttemp1 / c) ** (1.0 - p)
            gi2  = 1.0 - (1.0 + ttemp2 / c) ** (1.0 - p)
            gic1 = -(1.0 - gi1) * (1.0 - p) * (1.0 / (c + ttemp1) - 1.0 / c)
            gic2 = -(1.0 - gi2) * (1.0 - p) * (1.0 / (c + ttemp2) - 1.0 / c)
            gip1 = -(1.0 - gi1) * (math.log(c) - math.log(c + ttemp1))
            gip2 = -(1.0 - gi2) * (math.log(c) - math.log(c + ttemp2))
            gi  = gi2 - gi1
            gic = gic2 - gic1
            gip = gip2 - gip1

        # Renormalize temporal + its derivatives (quotient rule).
        dG_dc, dG_dp = G_grad
        invG = 1.0 / G_norm
        # gi_total = gi / G ;   d/dc = (gic*G - gi*dG_dc)/G^2 ;   d/dp similar
        gi_tot = gi * invG
        gi_tot_dc = (gic * G_norm - gi * dG_dc) * invG * invG
        gi_tot_dp = (gip * G_norm - gi * dG_dp) * invG * invG

        # spatial integrals + derivatives
        m_j = m[j]  # GPU scalar, accessed once
        w = [gamma, D, q, m_j]
        si      = poly_integ(fr,        w, px, py, x[j], y[j])
        sid     = poly_integ(dD_fr,     w, px, py, x[j], y[j])
        siq     = poly_integ(dq_fr,     w, px, py, x[j], y[j])
        sigamma = poly_integ(dgamma_fr, w, px, py, x[j], y[j])

        # Renormalize spatial + derivatives (keep GPU scalars, no float()).
        if F_norm is not None:
            F_at = F_norm[j]
            dF_dD, dF_dq, dF_dgamma = F_grad
            dF_dD_j = xp.asarray(dF_dD[j]) if dF_dD is not None else xp.array(0.0)
            dF_dq_j = xp.asarray(dF_dq[j]) if dF_dq is not None else xp.array(0.0)
            dF_dg_j = xp.asarray(dF_dgamma[j]) if dF_dgamma is not None else xp.array(0.0)
            invF = 1.0 / F_at
            si_tot = si * invF
            si_tot_dD = (sid * F_at - si * dF_dD_j) * invF * invF
            si_tot_dq = (siq * F_at - si * dF_dq_j) * invF * invF
            si_tot_dg = (sigamma * F_at - si * dF_dg_j) * invF * invF
        else:
            si_tot = si
            si_tot_dD = sid
            si_tot_dq = siq
            si_tot_dg = sigamma

        sk = A * xp.exp(alpha * m_j)
        Hfactor = (integ_h / H_norm) if is_3d else 1.0

        fv = sk * gi_tot * si_tot * Hfactor

        nparam = 9 if is_3d else 8
        dfv = xp.zeros(nparam)
        dfv[1] = (sk * gi_tot * si_tot / A * 2.0 * theta[1]) * Hfactor
        dfv[2] = (sk * gi_tot_dc * si_tot * 2.0 * theta[2]) * Hfactor
        dfv[3] = (sk * gi_tot * si_tot * m_j * 2.0 * theta[3]) * Hfactor
        dfv[4] = (sk * gi_tot_dp * si_tot * 2.0 * theta[4]) * Hfactor
        dfv[5] = (sk * gi_tot * si_tot_dD * 2.0 * theta[5]) * Hfactor
        dfv[6] = (sk * gi_tot * si_tot_dq * 2.0 * theta[6]) * Hfactor
        dfv[7] = (sk * gi_tot * si_tot_dg * 2.0 * theta[7]) * Hfactor
        # eta has no effect on integ_j because h integrates to 1 over [0,Z_max].
        return fv, dfv

    elif mver == 2:
        A     = theta[1] * theta[1]
        c     = theta[2] * theta[2]
        alpha = theta[3] * theta[3]
        p     = theta[4] * theta[4]
        D     = theta[5] * theta[5]
        gamma = theta[6] * theta[6]

        kparam = [A, alpha]
        gparam = [c, p]
        fparam = [D, gamma]

        t_j = float(t[j])
        int_part2 = dgfunint(tlength - t_j, gparam)
        if t_j <= tstart2:
            gtmp = dgfunint(tstart2 - t_j, gparam)
            int_part2 = [int_part2[k] - gtmp[k] for k in range(3)]

        m_j = m[j]  # hoist: access once
        def _wrap_ffunrint2_k(k_idx):
            def _f(r, w_unused):
                return dffunrint2(r, m_j, fparam)[k_idx]
            return _f
        int_part3 = [poly_integ(_wrap_ffunrint2_k(k_idx), None, px, py, x[j], y[j])
                     for k_idx in range(3)]
        int_part1 = dkappafun(m_j, kparam)

        # Renormalize temporal piece.
        dG_dc, dG_dp = G_grad
        invG = 1.0 / G_norm
        gi, gic, gip = int_part2
        gi_tot = gi * invG
        gi_tot_dc = (gic * G_norm - gi * dG_dc) * invG * invG
        gi_tot_dp = (gip * G_norm - gi * dG_dp) * invG * invG

        Hfactor = (integ_h / H_norm) if is_3d else 1.0
        si_tot = int_part3[0]
        fv = int_part1[0] * gi_tot * si_tot * Hfactor

        nparam = 8 if is_3d else 7
        dfv = xp.zeros(nparam)
        dfv[1] = (int_part1[1] * gi_tot * si_tot * 2.0 * theta[1]) * Hfactor
        dfv[2] = (int_part1[0] * gi_tot_dc * si_tot * 2.0 * theta[2]) * Hfactor
        dfv[3] = (int_part1[2] * gi_tot * si_tot * 2.0 * theta[3]) * Hfactor
        dfv[4] = (int_part1[0] * gi_tot_dp * si_tot * 2.0 * theta[4]) * Hfactor
        dfv[5] = (int_part1[0] * gi_tot * int_part3[1] * 2.0 * theta[5]) * Hfactor
        dfv[6] = (int_part1[0] * gi_tot * int_part3[2] * 2.0 * theta[6]) * Hfactor
        return fv, dfv

    else:
        raise ValueError(f"Unknown mver={mver}; expected 1 or 2")


# ===================================================================
# Conditional intensity at an arbitrary space-time point
# ===================================================================

def lambda_x(t_val, x_val, y_val, theta, t, x, y, m,
             mver=1, is_3d=False, z_val=None, z=None, tperiod=None,
             norms=None):
    """Conditional intensity at an arbitrary point ``(t_val, x_val, y_val[, z_val])``.

    Vectorized for CPU/GPU.  Renormalization is applied (no-op when ``norms``
    is None).  For 3D the depth kernel ``h`` multiplies the spatial factor.
    """
    xp = get_xp()
    norms = _norms_or_default(norms)
    G_norm = norms['G_norm']
    F_norm = norms['F_norm']
    H_norm = norms['H_norm']

    mask = t < t_val
    ti = t[mask]
    xi = x[mask]
    yi = y[mask]
    mi = m[mask]
    if is_3d:
        zi = z[mask]

    if len(ti) == 0:
        return 0.0

    delta = t_val - ti
    r2 = (x_val - xi) ** 2 + (y_val - yi) ** 2

    if mver == 1:
        A     = theta[1] * theta[1]
        c     = theta[2] * theta[2]
        alpha = theta[3] * theta[3]
        p     = theta[4] * theta[4]
        D     = theta[5] * theta[5]
        q     = theta[6] * theta[6]
        gamma = theta[7] * theta[7]

        part1 = xp.exp(alpha * mi)
        part2 = (p - 1.0) / c * (1.0 + delta / c) ** (-p)
        sig = D * xp.exp(gamma * mi)
        part3 = ((q - 1.0) / (sig * xp.pi) * (1.0 + r2 / sig) ** (-q))

        part2 = part2 / G_norm
        if F_norm is not None:
            # F_norm and mask may be on either backend.
            if _is_xp_array(F_norm, xp):
                F_sub = F_norm[mask] if _is_xp_array(mask, xp) else F_norm[np.asarray(mask)]
            else:
                mask_np = mask.get() if hasattr(mask, 'get') else mask
                F_sub = np.asarray(F_norm)[np.asarray(mask_np)]
            part3 = part3 / xp.asarray(F_sub)
        if is_3d:
            from .backend import get_special
            special = get_special()
            Z_max = tperiod[2]
            eta = theta[8] * theta[8]
            u = z_val / Z_max
            v = zi / Z_max
            log_h = _depth_log_h(u, v, eta, Z_max, special)
            part3 = part3 * xp.exp(log_h) / H_norm

        return float(xp.sum(A * part1 * part2 * part3))

    elif mver == 2:
        A     = theta[1] * theta[1]
        c     = theta[2] * theta[2]
        alpha = theta[3] * theta[3]
        p     = theta[4] * theta[4]
        D     = theta[5] * theta[5]
        gamma = theta[6] * theta[6]

        k_val = A * xp.exp(alpha * mi)
        g_val = (p - 1.0) / c * (1.0 + delta / c) ** (-p)
        sig_sq = D * xp.exp(gamma * mi)
        f_val = xp.exp(-r2 / (2.0 * sig_sq)) / (2.0 * xp.pi * sig_sq)
        g_val = g_val / G_norm
        if is_3d:
            from .backend import get_special
            special = get_special()
            Z_max = tperiod[2]
            eta = theta[7] * theta[7]
            u = z_val / Z_max
            v = zi / Z_max
            log_h = _depth_log_h(u, v, eta, Z_max, special)
            f_val = f_val * xp.exp(log_h) / H_norm

        return float(xp.sum(k_val * g_val * f_val))

    else:
        raise ValueError(f"Unknown mver={mver}; expected 1 or 2")


# ===================================================================
# Batch (vectorized) functions — operate on ALL N events at once
# ===================================================================

def lambda_j_batch(theta, t, x, y, z, m, bk, tau_cut, r_cut,
                    mver=1, is_3d=False, tperiod=None, norms=None,
                    nbr_lists=None, flag=None):
    """Vectorized conditional intensity for all events simultaneously.

    Returns an array of shape ``(N,)`` — the intensity at each event.
    This avoids N GPU synchronizations compared to calling ``lambda_j`` in a
    Python loop.
    """
    xp = get_xp()
    norms = _norms_or_default(norms)
    G_norm = norms['G_norm']
    F_norm = norms['F_norm']
    H_norm = norms['H_norm']
    N = t.shape[0]

    mu = theta[0] * theta[0]
    s = xp.full(N, mu) * bk

    if mver == 1:
        A     = theta[1] * theta[1]
        c     = theta[2] * theta[2]
        alpha = theta[3] * theta[3]
        p     = theta[4] * theta[4]
        D     = theta[5] * theta[5]
        q     = theta[6] * theta[6]
        gamma = theta[7] * theta[7]
    else:
        A     = theta[1] * theta[1]
        c     = theta[2] * theta[2]
        alpha = theta[3] * theta[3]
        p     = theta[4] * theta[4]
        D     = theta[5] * theta[5]
        gamma = theta[6] * theta[6]

    # Compute contributions from all parent pairs using neighbor lists.
    # Build (N,) output: for each target j, sum over parents i in nbr_lists[j].
    #
    # Strategy: for each j, we already have the nbr_lists from KDTree.
    # We loop over j but keep ALL per-parent operations on the GPU, doing only
    # ONE .item() per event (instead of per-parent).
    # For N~10k with small neighbor lists, this is still fast.
    #
    # For truly O(N²) without KDTree, use a dense matrix approach.

    if flag is not None:
        flag_np = np.asarray(flag)
    else:
        flag_np = np.ones(N, dtype=int)

    for j in range(N):
        if flag_np[j] != 1 or j == 0:
            continue

        # --- Resolve parent indices (same as per-event lambda_j) ---
        if nbr_lists is not None and j < len(nbr_lists):
            idx = np.asarray(nbr_lists[j], dtype=np.intp)
        else:
            idx = np.arange(j, dtype=np.intp)

        if idx.size == 0:
            continue

        ti = t[idx]
        xi = x[idx]
        yi = y[idx]
        mi = m[idx]
        delta = t[j] - ti
        r2 = (x[j] - xi) ** 2 + (y[j] - yi) ** 2

        # Legacy mask path (no KDTree)
        if nbr_lists is None or j >= len(nbr_lists):
            if tau_cut is not None and tau_cut < float('inf'):
                m_t = (t[j] - ti) <= tau_cut
                delta = delta[m_t]; r2 = r2[m_t]; mi = mi[m_t]
            if r_cut is not None and r_cut < float('inf'):
                m_r = xp.sqrt(r2) <= r_cut
                delta = delta[m_r]; r2 = r2[m_r]; mi = mi[m_r]
            if delta.size == 0:
                continue

        zi = None
        if is_3d:
            zi = z[idx]

        if mver == 1:
            part1 = xp.exp(alpha * mi)
            part2 = (p - 1.0) / c * (1.0 + delta / c) ** (-p)
            sig = D * xp.exp(gamma * mi)
            part3 = ((q - 1.0) / (sig * xp.pi) * (1.0 + r2 / sig) ** (-q))

            invG = 1.0 / G_norm
            part2 = part2 * invG
            if F_norm is not None:
                invF = 1.0 / xp.asarray(
                    F_norm[idx] if _is_xp_array(F_norm, xp)
                    else np.asarray(F_norm, dtype=np.float64)[idx],
                    dtype=part3.dtype)
                part3 = part3 * invF

            if is_3d:
                from .backend import get_special
                special = get_special()
                Z_max = tperiod[2]
                eta = theta[8] * theta[8]
                u = z[j] / Z_max
                v = zi / Z_max
                log_h = _depth_log_h(u, v, eta, Z_max, special)
                part3 = part3 * xp.exp(log_h) / H_norm

            s = s.at(j, s[j] + A * xp.sum(part1 * part2 * part3))
        else:  # mver == 2
            k_val = A * xp.exp(alpha * mi)
            g_val = (p - 1.0) / c * (1.0 + delta / c) ** (-p)
            sig_sq = D * xp.exp(gamma * mi)
            f_val = xp.exp(-r2 / (2.0 * sig_sq)) / (2.0 * xp.pi * sig_sq)

            invG = 1.0 / G_norm
            g_val = g_val * invG

            if is_3d:
                from .backend import get_special
                special = get_special()
                Z_max = tperiod[2]
                eta = theta[7] * theta[7]
                u = z[j] / Z_max
                v = zi / Z_max
                log_h = _depth_log_h(u, v, eta, Z_max, special)
                f_val = f_val * xp.exp(log_h) / H_norm

            s = s.at(j, s[j] + xp.sum(k_val * g_val * f_val))

    return s


def integ_j_batch(theta, t, x, y, m, px, py, tstart2, tlength,
                  mver=1, is_3d=False, norms=None, Z_max=None,
                  integ_h=None):
    """Vectorized space-time integral for all events simultaneously.

    The temporal part is fully vectorized.  The spatial polygon integral is
    inherently per-event (depends on event coordinates), but we extract CPU
    scalars once and run it on pure Python/NumPy, avoiding per-event GPU syncs.
    """
    xp = get_xp()
    norms = _norms_or_default(norms)
    G_norm = norms['G_norm']
    F_norm = norms['F_norm']
    H_norm = norms['H_norm']
    N = t.shape[0]

    if integ_h is None:
        integ_h = 1.0

    # Pre-extract CPU scalars for the per-event spatial loop.
    # This is the KEY optimization: all scalar access happens on CPU, zero GPU syncs.
    t_np = np.asarray(t)
    x_np = np.asarray(x)
    y_np = np.asarray(y)
    m_np = np.asarray(m)
    px_np = np.asarray(px)
    py_np = np.asarray(py)
    if F_norm is not None:
        F_norm_np = np.asarray(F_norm, dtype=np.float64)
    else:
        F_norm_np = None

    Hfactor = (integ_h / H_norm) if is_3d else 1.0

    out = np.zeros(N)

    if mver == 1:
        A     = theta[1] * theta[1]
        c     = theta[2] * theta[2]
        alpha = theta[3] * theta[3]
        p     = theta[4] * theta[4]
        D     = theta[5] * theta[5]
        q     = theta[6] * theta[6]
        gamma = theta[7] * theta[7]

        # Temporal part — fully vectorized
        ttemp = tlength - t_np
        ttemp1 = tstart2 - t_np
        # Vectorized: if t[j] > tstart2 use single CDF, else difference of two CDFs
        after = t_np > tstart2
        gi = np.where(after,
                      1.0 - (1.0 + ttemp / c) ** (1.0 - p),
                      (1.0 - (1.0 + ttemp / c) ** (1.0 - p))
                      - (1.0 - (1.0 + ttemp1 / c) ** (1.0 - p)))
        gi = gi / G_norm

        # Spatial part — per-event loop on CPU scalars (no GPU syncs!)
        w_cache = {}  # cache by unique magnitude index
        si_arr = np.empty(N)
        for j in range(N):
            w = [gamma, D, q, m_np[j]]
            si_arr[j] = poly_integ(fr, w, px_np, py_np,
                                   x_np[j], y_np[j])

        # Renormalize spatial
        if F_norm_np is not None:
            si_arr = si_arr / F_norm_np

        sk = A * np.exp(alpha * m_np)
        out = sk * gi * si_arr * Hfactor

    elif mver == 2:
        A     = theta[1] * theta[1]
        c     = theta[2] * theta[2]
        alpha = theta[3] * theta[3]
        p     = theta[4] * theta[4]
        D     = theta[5] * theta[5]
        gamma = theta[6] * theta[6]

        fparam = [D, gamma]

        # Temporal part — vectorized
        ttemp_end = tlength - t_np
        ttemp_start = tstart2 - t_np
        gi_end = np.where(ttemp_end > 0,
                          1.0 - (1.0 + ttemp_end / c) ** (1.0 - p), 0.0)
        gi_start = np.where(ttemp_start > 0,
                            1.0 - (1.0 + ttemp_start / c) ** (1.0 - p), 0.0)
        after = t_np > tstart2
        gi = np.where(after, gi_end, gi_end - gi_start)
        gi = gi / G_norm

        # Spatial part — per-event loop on CPU scalars
        si_arr = np.empty(N)
        for j in range(N):
            def _ffunrint2_wrap(r, w_unused, _mj=m_np[j]):
                return ffunrint2(r, _mj, fparam)
            si_arr[j] = poly_integ(_ffunrint2_wrap, None, px_np, py_np,
                                   x_np[j], y_np[j])

        sk = A * np.exp(alpha * m_np)
        out = sk * gi * si_arr * Hfactor

    return out


def lambda_j_grad_batch(theta, t, x, y, z, m, bk, tau_cut, r_cut,
                         mver=1, is_3d=False, tperiod=None, norms=None,
                         nbr_lists=None, flag=None):
    """Vectorized conditional intensity AND gradient for all events.

    Returns ``(fv, dfv)`` where ``fv`` is shape ``(N,)`` and ``dfv`` is
    shape ``(N, dimparam)``.
    """
    xp = get_xp()
    norms = _norms_or_default(norms)
    G_norm = norms['G_norm']
    F_norm = norms['F_norm']
    H_norm = norms['H_norm']
    G_grad = norms['G_grad']
    F_grad = norms['F_grad']
    H_grad = norms['H_grad']
    N = t.shape[0]
    dimparam = len(theta)

    mu = theta[0] * theta[0]
    fv = xp.full(N, mu) * bk
    dfv = xp.zeros((N, dimparam))

    # Background gradient (constant)
    dfv[:, 0] = bk * 2.0 * theta[0]

    if mver == 1:
        A     = theta[1] * theta[1]
        c     = theta[2] * theta[2]
        alpha = theta[3] * theta[3]
        p     = theta[4] * theta[4]
        D     = theta[5] * theta[5]
        q     = theta[6] * theta[6]
        gamma = theta[7] * theta[7]
        invG = 1.0 / G_norm
        dG_dc, dG_dp = G_grad
    else:
        A     = theta[1] * theta[1]
        c     = theta[2] * theta[2]
        alpha = theta[3] * theta[3]
        p     = theta[4] * theta[4]
        D     = theta[5] * theta[5]
        gamma = theta[6] * theta[6]
        invG = 1.0 / G_norm
        dG_dc, dG_dp = G_grad

    if flag is not None:
        flag_np = np.asarray(flag)
    else:
        flag_np = np.ones(N, dtype=int)

    for j in range(N):
        if flag_np[j] != 1 or j == 0:
            continue

        if nbr_lists is not None and j < len(nbr_lists):
            idx = np.asarray(nbr_lists[j], dtype=np.intp)
        else:
            idx = np.arange(j, dtype=np.intp)

        if idx.size == 0:
            continue

        ti = t[idx]; xi = x[idx]; yi = y[idx]; mi = m[idx]
        delta = t[j] - ti
        r2 = (x[j] - xi) ** 2 + (y[j] - yi) ** 2

        if nbr_lists is None or j >= len(nbr_lists):
            if tau_cut is not None and tau_cut < float('inf'):
                m_t = (t[j] - ti) <= tau_cut
                delta = delta[m_t]; r2 = r2[m_t]; mi = mi[m_t]
            if r_cut is not None and r_cut < float('inf'):
                m_r = xp.sqrt(r2) <= r_cut
                delta = delta[m_r]; r2 = r2[m_r]; mi = mi[m_r]
            if delta.size == 0:
                continue

        zi = None
        if is_3d:
            zi = z[idx]

        if mver == 1:
            part1 = xp.exp(alpha * mi)
            part2 = (p - 1.0) / c * (1.0 + delta / c) ** (-p)
            sig = D * xp.exp(gamma * mi)
            part3 = ((q - 1.0) / (sig * xp.pi) * (1.0 + r2 / sig) ** (-q))

            # Renormalization
            part2_r = part2 * invG

            F_at = None
            if F_norm is not None:
                if _is_xp_array(F_norm, xp):
                    F_at = xp.asarray(F_norm)[idx]
                else:
                    F_at = np.asarray(F_norm, dtype=np.float64)[idx]
                if F_at.shape[0] != part3.shape[0]:
                    from .renorm import spatial_norm
                    F_at = spatial_norm(D, gamma, q, mi,
                                        R_max=np.sqrt(r2.max()) if r2.size else 0.0)
            invF_val = 1.0
            if F_norm is not None:
                invF_val = 1.0 / xp.asarray(F_at, dtype=part3.dtype) if _is_xp_array(F_at, xp) \
                    else 1.0 / np.asarray(F_at, dtype=np.float64)
                part3_r = part3 * invF_val
            else:
                part3_r = part3

            h_val = None
            if is_3d:
                from .backend import get_special
                special = get_special()
                Z_max = tperiod[2]
                eta = theta[8] * theta[8]
                u = z[j] / Z_max
                v = zi / Z_max
                log_h = _depth_log_h(u, v, eta, Z_max, special)
                h_val = xp.exp(log_h) / H_norm
                part3_r = part3_r * h_val

            hfactor = h_val if is_3d else 1.0

            kA = A * part1
            kern = part2_r * part3_r
            contrib = kA * kern
            s_contrib = xp.sum(contrib)
            fv = fv.at(j, fv[j] + s_contrib)

            # --- gradient contributions ---
            sg2 = xp.sum(part1 * kern)
            sg3 = xp.sum(kA * (part2 * (-1.0 / c + p * delta / (c * (c + delta))) * invG
                                - part2 * invG * invG * dG_dc) * part3_r)
            sg4 = xp.sum(A * part1 * mi * kern)
            sg5 = xp.sum(kA * (part2 * (1.0 / (p - 1.0) - xp.log(1.0 + delta / c)) * invG
                                - part2 * invG * invG * dG_dp) * part3_r)

            # Spatial derivatives
            dF_dD, dF_dq, dF_dgamma = F_grad
            part3_dD = part3 / D * (-1.0 + q * (1.0 - 1.0 / (1.0 + r2 / sig)))
            if F_norm is not None and dF_dD is not None:
                dF_D_at = xp.asarray(np.asarray(dF_dD)[idx], dtype=part3.dtype) \
                    if not _is_xp_array(dF_dD, xp) else dF_dD[idx]
                sg6 = xp.sum(kA * part2_r * hfactor * (
                    part3_dD * invF_val - part3 * invF_val * invF_val * dF_D_at))
            else:
                sg6 = xp.sum(kA * part2_r * hfactor * part3_dD)

            part3_dq = part3 * (1.0 / (q - 1.0) - xp.log(1.0 + r2 / sig))
            if F_norm is not None and dF_dq is not None:
                dF_q_at = xp.asarray(np.asarray(dF_dq)[idx], dtype=part3.dtype) \
                    if not _is_xp_array(dF_dq, xp) else dF_dq[idx]
                sg7 = xp.sum(kA * part2_r * hfactor * (
                    part3_dq * invF_val - part3 * invF_val * invF_val * dF_q_at))
            else:
                sg7 = xp.sum(kA * part2_r * hfactor * part3_dq)

            part3_dgamma = part3 * (-mi + q * mi * (1.0 - 1.0 / (1.0 + r2 / sig)))
            if F_norm is not None and dF_dgamma is not None:
                dF_g_at = xp.asarray(np.asarray(dF_dgamma)[idx], dtype=part3.dtype) \
                    if not _is_xp_array(dF_dgamma, xp) else dF_dgamma[idx]
                sg8 = xp.sum(kA * part2_r * hfactor * (
                    part3_dgamma * invF_val - part3 * invF_val * invF_val * dF_g_at))
            else:
                sg8 = xp.sum(kA * part2_r * hfactor * part3_dgamma)

            sg9 = xp.zeros(1)
            if is_3d:
                from .backend import get_special
                special = get_special()
                Z_max = tperiod[2]
                eta = theta[8] * theta[8]
                u = z[j] / Z_max
                v = zi / Z_max
                safe_u = max(float(u), 1e-12)
                safe_1u = max(1.0 - float(u), 1e-12)
                digamma_term = (v * special.digamma(eta * v + 1.0)
                                + (1.0 - v) * special.digamma(eta * (1.0 - v) + 1.0)
                                - special.digamma(eta + 2.0))
                dh_deta = h_val * (v * math.log(safe_u)
                                   + (1.0 - v) * math.log(safe_1u)
                                   - digamma_term)
                sg9 = xp.sum(kA * part2_r * part3 * invF_val * dh_deta
                             if F_norm is not None else
                             kA * part2_r * part3 * dh_deta)

            # Chain-rule: d/d(theta_k) = sg_k * 2 * theta_k
            dfv = dfv.at(j, 0, dfv[j, 0])
            dfv = dfv.at(j, 1, sg2.item() * 2.0 * theta[1])
            dfv = dfv.at(j, 2, sg3.item() * 2.0 * theta[2])
            dfv = dfv.at(j, 3, sg4.item() * 2.0 * theta[3])
            dfv = dfv.at(j, 4, sg5.item() * 2.0 * theta[4])
            dfv = dfv.at(j, 5, sg6.item() * 2.0 * theta[5])
            dfv = dfv.at(j, 6, sg7.item() * 2.0 * theta[6])
            dfv = dfv.at(j, 7, sg8.item() * 2.0 * theta[7])
            if is_3d:
                dfv = dfv.at(j, 8, sg9.item() * 2.0 * theta[8])

        else:  # mver == 2
            k_val = A * xp.exp(alpha * mi)
            g_val = (p - 1.0) / c * (1.0 + delta / c) ** (-p)
            sig_sq = D * xp.exp(gamma * mi)
            f_val = xp.exp(-r2 / (2.0 * sig_sq)) / (2.0 * xp.pi * sig_sq)

            g_val_r = g_val * invG

            h_val = None
            if is_3d:
                from .backend import get_special
                special = get_special()
                Z_max = tperiod[2]
                eta = theta[7] * theta[7]
                u = z[j] / Z_max
                v = zi / Z_max
                log_h = _depth_log_h(u, v, eta, Z_max, special)
                h_val = xp.exp(log_h) / H_norm
                f_val = f_val * h_val

            kern = g_val_r * f_val
            contrib = k_val * kern
            s_contrib = xp.sum(contrib)
            fv = fv.at(j, fv[j] + s_contrib)

            # Gradients
            sg1 = xp.sum(xp.exp(alpha * mi) * kern)
            g_val_dc = g_val * (-1.0 / c + p * delta / (c * (c + delta)))
            sg2 = xp.sum(k_val * (g_val_dc * invG - g_val * invG * invG * dG_dc) * f_val)
            sg3 = xp.sum(A * mi * xp.exp(alpha * mi) * kern)
            g_val_dp = g_val * (1.0 / (p - 1.0) - xp.log(1.0 + delta / c))
            sg4 = xp.sum(k_val * (g_val_dp * invG - g_val * invG * invG * dG_dp) * f_val)
            f_val_D = f_val * (r2 / (2.0 * sig_sq * D) - 1.0 / D)
            sg5 = xp.sum(k_val * g_val_r * f_val_D)
            f_val_gamma = f_val * (r2 / (2.0 * sig_sq) - 1.0) * mi
            sg6 = xp.sum(k_val * g_val_r * f_val_gamma)

            dfv = dfv.at(j, 1, sg1.item() * 2.0 * theta[1])
            dfv = dfv.at(j, 2, sg2.item() * 2.0 * theta[2])
            dfv = dfv.at(j, 3, sg3.item() * 2.0 * theta[3])
            dfv = dfv.at(j, 4, sg4.item() * 2.0 * theta[4])
            dfv = dfv.at(j, 5, sg5.item() * 2.0 * theta[5])
            dfv = dfv.at(j, 6, sg6.item() * 2.0 * theta[6])

            if is_3d:
                from .backend import get_special
                special = get_special()
                Z_max = tperiod[2]
                eta = theta[7] * theta[7]
                u = z[j] / Z_max
                v = zi / Z_max
                safe_u = max(float(u), 1e-12)
                safe_1u = max(1.0 - float(u), 1e-12)
                digamma_term = (v * special.digamma(eta * v + 1.0)
                                + (1.0 - v) * special.digamma(eta * (1.0 - v) + 1.0)
                                - special.digamma(eta + 2.0))
                dh_deta = h_val * H_norm * (v * math.log(safe_u)
                                            + (1.0 - v) * math.log(safe_1u)
                                            - digamma_term)
                sg7 = xp.sum(k_val * g_val_r * (f_val / h_val) * dh_deta)
                dfv = dfv.at(j, 7, sg7.item() * 2.0 * theta[7])

    return fv, dfv


def integ_j_grad_batch(theta, t, x, y, m, px, py, tstart2, tlength,
                       mver=1, is_3d=False, norms=None, Z_max=None,
                       integ_h=None):
    """Vectorized space-time integral AND gradient for all events.

    Temporal part fully vectorized.  Spatial polygon integral runs per-event
    on CPU scalars (no GPU syncs in the inner loop).
    """
    xp = get_xp()
    norms = _norms_or_default(norms)
    G_norm = norms['G_norm']
    F_norm = norms['F_norm']
    H_norm = norms['H_norm']
    G_grad = norms['G_grad']
    F_grad = norms['F_grad']
    N = t.shape[0]

    if integ_h is None:
        integ_h = 1.0

    # Pre-extract CPU scalars
    t_np = np.asarray(t)
    x_np = np.asarray(x)
    y_np = np.asarray(y)
    m_np = np.asarray(m)
    px_np = np.asarray(px)
    py_np = np.asarray(py)
    if F_norm is not None:
        F_norm_np = np.asarray(F_norm, dtype=np.float64)
        dF_dD_np = np.asarray(F_grad[0], dtype=np.float64) if F_grad[0] is not None else None
        dF_dq_np = np.asarray(F_grad[1], dtype=np.float64) if F_grad[1] is not None else None
        dF_dg_np = np.asarray(F_grad[2], dtype=np.float64) if F_grad[2] is not None else None
    else:
        F_norm_np = dF_dD_np = dF_dq_np = dF_dg_np = None

    Hfactor = (integ_h / H_norm) if is_3d else 1.0

    if mver == 1:
        A     = theta[1] * theta[1]
        c     = theta[2] * theta[2]
        alpha = theta[3] * theta[3]
        p     = theta[4] * theta[4]
        D     = theta[5] * theta[5]
        q     = theta[6] * theta[6]
        gamma = theta[7] * theta[7]

        dimparam = 9 if is_3d else 8
        fv = np.zeros(N)
        dfv = np.zeros((N, dimparam))

        # Temporal part — fully vectorized on CPU (scalar ops)
        ttemp = tlength - t_np
        ttemp1 = tstart2 - t_np
        after = t_np > tstart2
        gi = np.where(after,
                      1.0 - (1.0 + ttemp / c) ** (1.0 - p),
                      (1.0 - (1.0 + ttemp / c) ** (1.0 - p))
                      - (1.0 - (1.0 + ttemp1 / c) ** (1.0 - p)))
        gi = gi / G_norm

        # Renormalized temporal derivatives (vectorized)
        dG_dc, dG_dp = G_grad
        invG = 1.0 / G_norm
        invG2 = invG * invG

        # gic and gip for both branches, then renormalize
        one_minus_gi = 1.0 - np.where(after,
                                      1.0 - (1.0 + ttemp / c) ** (1.0 - p),
                                      (1.0 - (1.0 + ttemp / c) ** (1.0 - p))
                                      - (1.0 - (1.0 + ttemp1 / c) ** (1.0 - p)))
        # For after:  gic = -one_minus_gi * (1-p) * (1/(c+ttemp) - 1/c)
        #             gip = -one_minus_gi * (log(c) - log(c+ttemp))
        # For before: gic = gic2 - gic1  (same formula with ttemp vs ttemp1)
        #             gip = gip2 - gip1
        safe_ttemp = np.maximum(ttemp, 1e-300)
        safe_ttemp1 = np.maximum(ttemp1, 1e-300)
        gic_after = -one_minus_gi * (1.0 - p) * (1.0 / (c + safe_ttemp) - 1.0 / c)
        gip_after = -one_minus_gi * (np.log(c) - np.log(c + safe_ttemp))
        # Before branch: compute gi1, gi2 separately
        gi1 = 1.0 - (1.0 + safe_ttemp1 / c) ** (1.0 - p)
        gi2 = 1.0 - (1.0 + safe_ttemp / c) ** (1.0 - p)
        omg1 = 1.0 - gi1
        omg2 = 1.0 - gi2
        gic1 = -omg1 * (1.0 - p) * (1.0 / (c + safe_ttemp1) - 1.0 / c)
        gic2 = -omg2 * (1.0 - p) * (1.0 / (c + safe_ttemp) - 1.0 / c)
        gip1 = -omg1 * (np.log(c) - np.log(c + safe_ttemp1))
        gip2 = -omg2 * (np.log(c) - np.log(c + safe_ttemp))

        gic = np.where(after, gic_after, gic2 - gic1)
        gip = np.where(after, gip_after, gip2 - gip1)

        gi_tot = gi  # already divided by G_norm above
        gi_tot_dc = (gic * G_norm - gi * dG_dc) * invG2
        gi_tot_dp = (gip * G_norm - gi * dG_dp) * invG2

        # Spatial part — per-event on CPU scalars
        si_arr = np.empty(N)
        sid_arr = np.empty(N)
        siq_arr = np.empty(N)
        sig_arr = np.empty(N)

        for j in range(N):
            w = [gamma, D, q, m_np[j]]
            si_arr[j] = poly_integ(fr, w, px_np, py_np, x_np[j], y_np[j])
            sid_arr[j] = poly_integ(dD_fr, w, px_np, py_np, x_np[j], y_np[j])
            siq_arr[j] = poly_integ(dq_fr, w, px_np, py_np, x_np[j], y_np[j])
            sig_arr[j] = poly_integ(dgamma_fr, w, px_np, py_np, x_np[j], y_np[j])

        # Renormalize spatial
        if F_norm_np is not None:
            invF = 1.0 / F_norm_np
            invF2 = invF * invF
            dF_dD_j = dF_dD_np if dF_dD_np is not None else 0.0
            dF_dq_j = dF_dq_np if dF_dq_np is not None else 0.0
            dF_dg_j = dF_dg_np if dF_dg_np is not None else 0.0
            si_tot = si_arr * invF
            si_tot_dD = (sid_arr * F_norm_np - si_arr * dF_dD_j) * invF2
            si_tot_dq = (siq_arr * F_norm_np - si_arr * dF_dq_j) * invF2
            si_tot_dg = (sig_arr * F_norm_np - si_arr * dF_dg_j) * invF2
        else:
            si_tot = si_arr
            si_tot_dD = sid_arr
            si_tot_dq = siq_arr
            si_tot_dg = sig_arr

        sk = A * np.exp(alpha * m_np)
        fv = sk * gi_tot * si_tot * Hfactor

        dfv[:, 1] = (sk * gi_tot * si_tot / A * 2.0 * theta[1]) * Hfactor
        dfv[:, 2] = (sk * gi_tot_dc * si_tot * 2.0 * theta[2]) * Hfactor
        dfv[:, 3] = (sk * gi_tot * si_tot * m_np * 2.0 * theta[3]) * Hfactor
        dfv[:, 4] = (sk * gi_tot_dp * si_tot * 2.0 * theta[4]) * Hfactor
        dfv[:, 5] = (sk * gi_tot * si_tot_dD * 2.0 * theta[5]) * Hfactor
        dfv[:, 6] = (sk * gi_tot * si_tot_dq * 2.0 * theta[6]) * Hfactor
        dfv[:, 7] = (sk * gi_tot * si_tot_dg * 2.0 * theta[7]) * Hfactor

    elif mver == 2:
        A     = theta[1] * theta[1]
        c     = theta[2] * theta[2]
        alpha = theta[3] * theta[3]
        p     = theta[4] * theta[4]
        D     = theta[5] * theta[5]
        gamma = theta[6] * theta[6]

        dimparam = 8 if is_3d else 7
        fv = np.zeros(N)
        dfv = np.zeros((N, dimparam))

        kparam = [A, alpha]
        fparam = [D, gamma]

        # Temporal part — vectorized
        dG_dc, dG_dp = G_grad
        invG = 1.0 / G_norm
        invG2 = invG * invG

        safe_ttemp = np.maximum(tlength - t_np, 0.0)
        safe_ttemp1 = np.maximum(tstart2 - t_np, 0.0)

        # dgfunint returns [value, d_c, d_p]
        int_part2_v = np.array([dgfunint(tlength - t_np[j], [c, p]) for j in range(N)],
                               dtype=np.float64)  # (N, 3)
        # For events before tstart2, subtract
        int_part2_v_final = np.copy(int_part2_v)
        for j in range(N):
            if t_np[j] <= tstart2:
                gtmp = dgfunint(tstart2 - t_np[j], [c, p])
                int_part2_v_final[j, 0] -= gtmp[0]
                int_part2_v_final[j, 1] -= gtmp[1]
                int_part2_v_final[j, 2] -= gtmp[2]

        gi_tot = int_part2_v_final[:, 0] * invG
        gi_tot_dc = (int_part2_v_final[:, 1] * G_norm - int_part2_v_final[:, 0] * dG_dc) * invG2
        gi_tot_dp = (int_part2_v_final[:, 2] * G_norm - int_part2_v_final[:, 0] * dG_dp) * invG2

        # Spatial part — per-event on CPU scalars (3 poly_integ calls per event)
        int_part3_v = np.zeros((N, 3))
        for j in range(N):
            for k_idx in range(3):
                def _wrap_ffunrint2_k(r, w_unused, _mj=m_np[j], _fparam=fparam, _k=k_idx):
                    return dffunrint2(r, _mj, _fparam)[_k]
                int_part3_v[j, k_idx] = poly_integ(_wrap_ffunrint2_k, None, px_np, py_np,
                                                    x_np[j], y_np[j])

        int_part1_v = np.array([dkappafun(m_np[j], kparam) for j in range(N)],
                               dtype=np.float64)  # (N, 3)

        si_tot = int_part3_v[:, 0]
        fv = int_part1_v[:, 0] * gi_tot * si_tot * Hfactor

        dfv[:, 1] = (int_part1_v[:, 1] * gi_tot * si_tot * 2.0 * theta[1]) * Hfactor
        dfv[:, 2] = (int_part1_v[:, 0] * gi_tot_dc * si_tot * 2.0 * theta[2]) * Hfactor
        dfv[:, 3] = (int_part1_v[:, 2] * gi_tot * si_tot * 2.0 * theta[3]) * Hfactor
        dfv[:, 4] = (int_part1_v[:, 0] * gi_tot_dp * si_tot * 2.0 * theta[4]) * Hfactor
        dfv[:, 5] = (int_part1_v[:, 0] * gi_tot * int_part3_v[:, 1] * 2.0 * theta[5]) * Hfactor
        dfv[:, 6] = (int_part1_v[:, 0] * gi_tot * int_part3_v[:, 2] * 2.0 * theta[6]) * Hfactor

    return fv, dfv


# ===================================================================
# Flattened batch functions — ONE GPU kernel for ALL parent-child pairs
# ===================================================================

def _scatter_add(xp, out, indices, values):
    """Add ``values`` to ``out`` at ``indices`` (scatter-add).

    Uses ``xp.add.at`` (NumPy) or ``cupyx.scatter_add`` (CuPy).
    """
    from .backend import get_engine
    if get_engine() == 'gpu':
        from cupyx import scatter_add as cp_scatter_add
        cp_scatter_add(out, indices, values)
    else:
        xp.add.at(out, indices, values)


def lambda_flat(theta, t, x, y, z, m, bk,
                mver=1, is_3d=False, tperiod=None, norms=None,
                nbr_flat=None, flag=None):
    """Vectorized intensity for ALL events — one GPU pass.

    Parameters
    ----------
    nbr_flat : tuple (targets, sources)
        ``targets``: (P,) array of target event indices.
        ``sources``: (P,) array of source (parent) event indices.
        All parent-child pairs are processed in a single vectorized operation.

    Returns
    -------
    (N,) array of intensity at each event.
    """
    xp = get_xp()
    norms = _norms_or_default(norms)
    G_norm = norms['G_norm']
    F_norm = norms['F_norm']
    H_norm = norms['H_norm']
    N = t.shape[0]
    mu = theta[0] * theta[0]

    # Background contribution
    lam = xp.full(N, mu, dtype=xp.float64) * bk

    if nbr_flat is None:
        return lam

    targets, sources = nbr_flat
    if len(targets) == 0:
        return lam

    # Gather source data: (P,)
    ti = t[sources]
    xi = x[sources]
    yi = y[sources]
    mi = m[sources]

    # Gather target data: (P,)
    t_j = t[targets]
    x_j = x[targets]
    y_j = y[targets]

    # Pairwise deltas: (P,)
    delta = t_j - ti
    r2 = (x_j - xi) ** 2 + (y_j - yi) ** 2

    if mver == 1:
        A     = theta[1] * theta[1]
        c     = theta[2] * theta[2]
        alpha = theta[3] * theta[3]
        p     = theta[4] * theta[4]
        D     = theta[5] * theta[5]
        q     = theta[6] * theta[6]
        gamma = theta[7] * theta[7]

        part1 = xp.exp(alpha * mi)
        part2 = (p - 1.0) / c * (1.0 + delta / c) ** (-p)
        sig = D * xp.exp(gamma * mi)
        part3 = ((q - 1.0) / (sig * xp.pi) * (1.0 + r2 / sig) ** (-q))

        invG = 1.0 / G_norm
        part2 = part2 * invG
        if F_norm is not None:
            invF = 1.0 / xp.asarray(
                F_norm[sources] if _is_xp_array(F_norm, xp)
                else np.asarray(F_norm, dtype=np.float64)[sources],
                dtype=part3.dtype)
            part3 = part3 * invF

        if is_3d:
            from .backend import get_special
            special = get_special()
            Z_max = tperiod[2]
            eta = theta[8] * theta[8]
            z_j = z[targets]
            zi = z[sources]
            u = z_j / Z_max
            v = zi / Z_max
            # _depth_log_h expects scalar u, array v.  We need elementwise.
            # Recompute inline for vectorization.
            log_beta = (special.gammaln(eta * v + 1.0)
                        + special.gammaln(eta * (1.0 - v) + 1.0)
                        - special.gammaln(eta + 2.0))
            u_val = xp.clip(u, 1e-12, 1.0 - 1e-12)
            one_minus_u = 1.0 - u_val
            log_h = ((eta * v) * xp.log(u_val)
                     + (eta * (1.0 - v)) * xp.log(one_minus_u)
                     - xp.log(Z_max) - log_beta)
            part3 = part3 * xp.exp(log_h) / H_norm

        contrib = A * part1 * part2 * part3

    else:  # mver == 2
        A     = theta[1] * theta[1]
        c     = theta[2] * theta[2]
        alpha = theta[3] * theta[3]
        p     = theta[4] * theta[4]
        D     = theta[5] * theta[5]
        gamma = theta[6] * theta[6]

        k_val = A * xp.exp(alpha * mi)
        g_val = (p - 1.0) / c * (1.0 + delta / c) ** (-p)
        sig_sq = D * xp.exp(gamma * mi)
        f_val = xp.exp(-r2 / (2.0 * sig_sq)) / (2.0 * xp.pi * sig_sq)

        invG = 1.0 / G_norm
        g_val = g_val * invG

        if is_3d:
            from .backend import get_special
            special = get_special()
            Z_max = tperiod[2]
            eta = theta[7] * theta[7]
            z_j = z[targets]
            zi = z[sources]
            u = z_j / Z_max
            v = zi / Z_max
            log_beta = (special.gammaln(eta * v + 1.0)
                        + special.gammaln(eta * (1.0 - v) + 1.0)
                        - special.gammaln(eta + 2.0))
            u_val = xp.clip(u, 1e-12, 1.0 - 1e-12)
            one_minus_u = 1.0 - u_val
            log_h = ((eta * v) * xp.log(u_val)
                     + (eta * (1.0 - v)) * xp.log(one_minus_u)
                     - xp.log(Z_max) - log_beta)
            f_val = f_val * xp.exp(log_h) / H_norm

        contrib = k_val * g_val * f_val

    # Scatter-add contributions to target events
    _scatter_add(xp, lam, targets, contrib)

    return lam


def lambda_grad_flat(theta, t, x, y, z, m, bk,
                     mver=1, is_3d=False, tperiod=None, norms=None,
                     nbr_flat=None, flag=None):
    """Vectorized intensity AND gradient for ALL events — one GPU pass.

    Returns ``(fv, dfv)`` where ``fv`` is shape ``(N,)`` and ``dfv`` is
    shape ``(N, dimparam)``.
    """
    xp = get_xp()
    norms = _norms_or_default(norms)
    G_norm = norms['G_norm']
    F_norm = norms['F_norm']
    H_norm = norms['H_norm']
    G_grad = norms['G_grad']
    F_grad = norms['F_grad']
    N = t.shape[0]
    dimparam = len(theta)
    mu = theta[0] * theta[0]

    # Background
    fv = xp.full(N, mu, dtype=xp.float64) * bk
    dfv = xp.zeros((N, dimparam), dtype=xp.float64)
    dfv[:, 0] = bk * 2.0 * theta[0]

    if nbr_flat is None:
        return fv, dfv

    targets, sources = nbr_flat
    P = len(targets)
    if P == 0:
        return fv, dfv

    # Gather data: (P,)
    ti = t[sources]; xi = x[sources]; yi = y[sources]; mi = m[sources]
    t_j = t[targets]; x_j = x[targets]; y_j = y[targets]
    delta = t_j - ti
    r2 = (x_j - xi) ** 2 + (y_j - yi) ** 2

    if mver == 1:
        A     = theta[1] * theta[1]
        c     = theta[2] * theta[2]
        alpha = theta[3] * theta[3]
        p     = theta[4] * theta[4]
        D     = theta[5] * theta[5]
        q     = theta[6] * theta[6]
        gamma = theta[7] * theta[7]
        invG = 1.0 / G_norm
        dG_dc, dG_dp = G_grad

        part1 = xp.exp(alpha * mi)
        part2 = (p - 1.0) / c * (1.0 + delta / c) ** (-p)
        sig = D * xp.exp(gamma * mi)
        part3 = ((q - 1.0) / (sig * xp.pi) * (1.0 + r2 / sig) ** (-q))

        part2_r = part2 * invG

        invF = xp.ones(P, dtype=xp.float64)
        if F_norm is not None:
            F_at = xp.asarray(
                F_norm[sources] if _is_xp_array(F_norm, xp)
                else np.asarray(F_norm, dtype=np.float64)[sources],
                dtype=part3.dtype)
            invF = 1.0 / F_at
            part3_r = part3 * invF
        else:
            part3_r = part3

        hfactor = xp.ones(P, dtype=xp.float64)
        if is_3d:
            from .backend import get_special
            special = get_special()
            Z_max = tperiod[2]
            eta = theta[8] * theta[8]
            z_j = z[targets]; zi = z[sources]
            u = z_j / Z_max; v = zi / Z_max
            log_beta = (special.gammaln(eta * v + 1.0)
                        + special.gammaln(eta * (1.0 - v) + 1.0)
                        - special.gammaln(eta + 2.0))
            u_val = xp.clip(u, 1e-12, 1.0 - 1e-12)
            log_h = ((eta * v) * xp.log(u_val)
                     + (eta * (1.0 - v)) * xp.log(1.0 - u_val)
                     - xp.log(Z_max) - log_beta)
            h_val = xp.exp(log_h) / H_norm
            hfactor = h_val
            part3_r = part3_r * h_val

        kA = A * part1
        kern = part2_r * part3_r
        contrib = kA * kern   # (P,)
        _scatter_add(xp, fv, targets, contrib)

        # --- gradient contributions (per-parameter, scatter-add to target) ---
        # d/dA: 2*theta[1] * sum_parents part1 * kern
        _scatter_add(xp, dfv[:, 1], targets,
                     2.0 * theta[1] * part1 * kern)
        # d/dc
        part2_dc = part2 * (-1.0 / c + p * delta / (c * (c + delta)))
        g_c = kA * (part2_dc * invG - part2 * invG * invG * dG_dc) * part3_r
        _scatter_add(xp, dfv[:, 2], targets, 2.0 * theta[2] * g_c)
        # d/dalpha
        _scatter_add(xp, dfv[:, 3], targets,
                     2.0 * theta[3] * A * part1 * mi * kern)
        # d/dp
        part2_dp = part2 * (1.0 / (p - 1.0) - xp.log(1.0 + delta / c))
        g_p = kA * (part2_dp * invG - part2 * invG * invG * dG_dp) * part3_r
        _scatter_add(xp, dfv[:, 4], targets, 2.0 * theta[4] * g_p)
        # d/dD
        part3_dD = part3 / D * (-1.0 + q * (1.0 - 1.0 / (1.0 + r2 / sig)))
        if F_norm is not None and F_grad[0] is not None:
            dF_D_at = xp.asarray(
                F_grad[0][sources] if _is_xp_array(F_grad[0], xp)
                else np.asarray(F_grad[0], dtype=np.float64)[sources],
                dtype=part3.dtype)
            g_D = kA * part2_r * hfactor * (part3_dD * invF - part3 * invF * invF * dF_D_at)
        else:
            g_D = kA * part2_r * hfactor * part3_dD
        _scatter_add(xp, dfv[:, 5], targets, 2.0 * theta[5] * g_D)
        # d/dq
        part3_dq = part3 * (1.0 / (q - 1.0) - xp.log(1.0 + r2 / sig))
        if F_norm is not None and F_grad[1] is not None:
            dF_q_at = xp.asarray(
                F_grad[1][sources] if _is_xp_array(F_grad[1], xp)
                else np.asarray(F_grad[1], dtype=np.float64)[sources],
                dtype=part3.dtype)
            g_q = kA * part2_r * hfactor * (part3_dq * invF - part3 * invF * invF * dF_q_at)
        else:
            g_q = kA * part2_r * hfactor * part3_dq
        _scatter_add(xp, dfv[:, 6], targets, 2.0 * theta[6] * g_q)
        # d/dgamma
        part3_dgamma = part3 * (-mi + q * mi * (1.0 - 1.0 / (1.0 + r2 / sig)))
        if F_norm is not None and F_grad[2] is not None:
            dF_g_at = xp.asarray(
                F_grad[2][sources] if _is_xp_array(F_grad[2], xp)
                else np.asarray(F_grad[2], dtype=np.float64)[sources],
                dtype=part3.dtype)
            g_gamma = kA * part2_r * hfactor * (part3_dgamma * invF - part3 * invF * invF * dF_g_at)
        else:
            g_gamma = kA * part2_r * hfactor * part3_dgamma
        _scatter_add(xp, dfv[:, 7], targets, 2.0 * theta[7] * g_gamma)

        if is_3d:
            # d/deta
            from .backend import get_special
            special = get_special()
            Z_max = tperiod[2]
            eta = theta[8] * theta[8]
            z_j = z[targets]; zi = z[sources]
            u = z_j / Z_max; v = zi / Z_max
            u_val = xp.clip(u, 1e-12, 1.0 - 1e-12)
            digamma_term = (v * special.digamma(eta * v + 1.0)
                            + (1.0 - v) * special.digamma(eta * (1.0 - v) + 1.0)
                            - special.digamma(eta + 2.0))
            dh_deta = h_val * (v * xp.log(u_val)
                               + (1.0 - v) * xp.log(1.0 - u_val)
                               - digamma_term)
            invF_val = invF if F_norm is not None else xp.ones(P, dtype=xp.float64)
            g_eta = kA * part2_r * part3 * invF_val * dh_deta
            _scatter_add(xp, dfv[:, 8], targets, 2.0 * theta[8] * g_eta)

    else:  # mver == 2
        A     = theta[1] * theta[1]
        c     = theta[2] * theta[2]
        alpha = theta[3] * theta[3]
        p     = theta[4] * theta[4]
        D     = theta[5] * theta[5]
        gamma = theta[6] * theta[6]
        invG = 1.0 / G_norm
        dG_dc, dG_dp = G_grad

        k_val = A * xp.exp(alpha * mi)
        g_val = (p - 1.0) / c * (1.0 + delta / c) ** (-p)
        sig_sq = D * xp.exp(gamma * mi)
        f_val = xp.exp(-r2 / (2.0 * sig_sq)) / (2.0 * xp.pi * sig_sq)

        g_val_r = g_val * invG

        hfactor = xp.ones(P, dtype=xp.float64)
        if is_3d:
            from .backend import get_special
            special = get_special()
            Z_max = tperiod[2]
            eta = theta[7] * theta[7]
            z_j = z[targets]; zi = z[sources]
            u = z_j / Z_max; v = zi / Z_max
            log_beta = (special.gammaln(eta * v + 1.0)
                        + special.gammaln(eta * (1.0 - v) + 1.0)
                        - special.gammaln(eta + 2.0))
            u_val = xp.clip(u, 1e-12, 1.0 - 1e-12)
            log_h = ((eta * v) * xp.log(u_val)
                     + (eta * (1.0 - v)) * xp.log(1.0 - u_val)
                     - xp.log(Z_max) - log_beta)
            h_val = xp.exp(log_h) / H_norm
            hfactor = h_val
            f_val = f_val * h_val

        kern = g_val_r * f_val
        contrib = k_val * kern
        _scatter_add(xp, fv, targets, contrib)

        # Gradients
        _scatter_add(xp, dfv[:, 1], targets,
                     2.0 * theta[1] * xp.exp(alpha * mi) * kern)
        g_val_dc = g_val * (-1.0 / c + p * delta / (c * (c + delta)))
        g_c = k_val * (g_val_dc * invG - g_val * invG * invG * dG_dc) * f_val
        _scatter_add(xp, dfv[:, 2], targets, 2.0 * theta[2] * g_c)
        _scatter_add(xp, dfv[:, 3], targets,
                     2.0 * theta[3] * A * mi * xp.exp(alpha * mi) * kern)
        g_val_dp = g_val * (1.0 / (p - 1.0) - xp.log(1.0 + delta / c))
        g_p = k_val * (g_val_dp * invG - g_val * invG * invG * dG_dp) * f_val
        _scatter_add(xp, dfv[:, 4], targets, 2.0 * theta[4] * g_p)
        f_val_D = f_val * (r2 / (2.0 * sig_sq * D) - 1.0 / D)
        _scatter_add(xp, dfv[:, 5], targets,
                     2.0 * theta[5] * k_val * g_val_r * f_val_D)
        f_val_gamma = f_val * (r2 / (2.0 * sig_sq) - 1.0) * mi
        _scatter_add(xp, dfv[:, 6], targets,
                     2.0 * theta[6] * k_val * g_val_r * f_val_gamma)

        if is_3d:
            from .backend import get_special
            special = get_special()
            Z_max = tperiod[2]
            eta = theta[7] * theta[7]
            z_j = z[targets]; zi = z[sources]
            u = z_j / Z_max; v = zi / Z_max
            u_val = xp.clip(u, 1e-12, 1.0 - 1e-12)
            digamma_term = (v * special.digamma(eta * v + 1.0)
                            + (1.0 - v) * special.digamma(eta * (1.0 - v) + 1.0)
                            - special.digamma(eta + 2.0))
            dh_deta = h_val * H_norm * (v * xp.log(u_val)
                                        + (1.0 - v) * xp.log(1.0 - u_val)
                                        - digamma_term)
            g_eta = k_val * g_val_r * (f_val / h_val) * dh_deta
            _scatter_add(xp, dfv[:, 7], targets, 2.0 * theta[7] * g_eta)

    return fv, dfv
