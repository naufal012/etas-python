"""
Stochastic declustering algorithm for the ETAS model.

Equivalent of ``declust.c`` and ``cxxdeclust`` in ``fitMP.cpp``.  Computes the
background seismicity rate via Gaussian kernel smoothing, updates the
declustering probabilities, and integrates the background rate over the
spatial domain.

Extensions over the R reference:

* **Fixed index-scatter bug** in the polygon-integral accumulation (the
  previous code declared ``final_mask`` but scattered the doubly-masked
  ``val`` back onto only-``valid_mask`` indices, corrupting ``integ0``).
* **KDTree neighbor pruning** via :class:`etas.src.neighbors.NeighborIndex`,
  replacing the O(N^2) per-event mask construction in the intensity loop.
* **Analytic renormalization** of the truncated kernels (no-op when the
  ``norms`` dict is None).
* **GPU-safe depth kernel**: replaces ``max(float(u), eps)`` (broken on CuPy)
  with explicit ``xp.where`` clamping.
"""

import math
import numpy as np
from .backend import get_xp


def _depth_log_h_vec(u, v, eta, Z_max, special, xp):
    """Vectorized log of h(u; v) with GPU-safe clamping.

    ``u`` is a scalar (target depth), ``v`` is an array.  ``xp.where`` is used
    instead of Python ``max`` so this works on both NumPy and CuPy arrays.
    """
    log_beta = (special.gammaln(eta * v + 1.0)
                + special.gammaln(eta * (1.0 - v) + 1.0)
                - special.gammaln(eta + 2.0))
    u_arr = xp.asarray(float(u))
    safe_u = xp.where(u_arr > 1e-12, u_arr, xp.asarray(1e-12))
    one_minus_u = xp.asarray(1.0 - float(u))
    safe_1u = xp.where(one_minus_u > 1e-12, one_minus_u, xp.asarray(1e-12))
    return ((eta * v) * xp.log(safe_u)
            + (eta * (1.0 - v)) * xp.log(safe_1u)
            - math.log(Z_max) - log_beta)


def decluster(theta, rbwd, revents, rpoly, tperiod, ndiv, mver,
              tau_cut, r_cut, is_3d=False, norms=None, nbr_index=None):
    """Stochastic declustering: update background rates and probabilities.

    Parameters
    ----------
    theta : array-like
        **Natural** (un-squared) parameters.
    nbr_index : etas.src.neighbors.NeighborIndex or None
        Precomputed neighbor structure.  When supplied the intensity loop uses
        KDTree-pruned parent lists; otherwise it falls back to the legacy
        O(N^2) mask path.
    norms : dict or None
        Renormalization constants (see :mod:`etas.src.renorm`).
    """
    xp = get_xp()

    N = revents.shape[0]
    t = xp.asarray(revents[:, 0])
    x = xp.asarray(revents[:, 1])
    y = xp.asarray(revents[:, 2])
    if is_3d:
        z = xp.asarray(revents[:, 8])
    m = xp.asarray(revents[:, 3])
    pb = xp.asarray(revents[:, 6])

    rbwd = xp.asarray(rbwd)

    np_val = rpoly.shape[0]
    px = xp.asarray(rpoly[:, 0])
    py = xp.asarray(rpoly[:, 1])

    tstart2 = tperiod[0]
    tlength = tperiod[1]

    mu = theta[0]
    A = theta[1]
    c = theta[2]
    alpha = theta[3]
    p = theta[4]
    D = theta[5]

    if mver == 1:
        q = theta[6]
        gamma = theta[7]
        if is_3d:
            eta = theta[8]
    else:
        gamma = theta[6]
        if is_3d:
            eta = theta[7]

    if is_3d:
        Z_max = tperiod[2]

    # Renormalization defaults
    G_norm = norms['G_norm'] if norms is not None else 1.0
    F_norm = norms['F_norm'] if (norms is not None) else None
    H_norm = norms['H_norm'] if norms is not None else 1.0

    # -------------------------------------------------------------------------
    # Step 1: background rate bk[i] via Gaussian KDE (vectorized, O(N^2))
    # -------------------------------------------------------------------------
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    r2_mat = dx ** 2 + dy ** 2

    sig_sq = rbwd[None, :] ** 2
    dGauss_mat = xp.exp(-r2_mat / (2.0 * sig_sq)) / (2.0 * xp.pi * sig_sq)

    bk = xp.sum(pb[None, :] * dGauss_mat, axis=1) / (tlength - tstart2)

    # -------------------------------------------------------------------------
    # Step 2: integ0 via polygon integration (BUG FIX: scatter indices)
    # -------------------------------------------------------------------------
    poly_int = xp.zeros(N)

    for k in range(np_val - 1):
        dpx = (px[k + 1] - px[k]) / ndiv
        dpy = (py[k + 1] - py[k]) / ndiv

        for l in range(ndiv):
            x1 = px[k] + dpx * l
            y1 = py[k] + dpy * l
            x2 = px[k] + dpx * (l + 1)
            y2 = py[k] + dpy * (l + 1)

            det = ((x1 * y2 + y1 * x + x2 * y) -
                   (x2 * y1 + y2 * x + x1 * y))

            sgn_det = xp.where(det < 0, -1.0, 1.0)
            valid_mask = xp.abs(det) >= 1.0e-10
            if not xp.any(valid_mask):
                continue

            x_v = x[valid_mask]
            y_v = y[valid_mask]
            sgn_v = sgn_det[valid_mask]
            bwd_v = rbwd[valid_mask]

            r1_sq = (x1 - x_v) ** 2 + (y1 - y_v) ** 2
            r2_sq = (x2 - x_v) ** 2 + (y2 - y_v) ** 2
            dist2_edge = (x1 - x2) ** 2 + (y1 - y2) ** 2

            r1 = xp.sqrt(r1_sq)
            r2_len = xp.sqrt(r2_sq)

            denom = 2.0 * r1 * r2_len
            phi = xp.where(denom > 0,
                           (r1_sq + r2_sq - dist2_edge) / denom, 0.0)
            phi = xp.clip(phi, -1.0 + 1e-10, 1.0 - 1e-10)
            phi = xp.arccos(phi)

            r_sum = r1 + r2_len
            r_mask = r_sum > 1.0e-20
            if not xp.any(r_mask):
                continue

            # --- BUG FIX -------------------------------------------------------
            # The previous code applied ``r_mask`` to the per-element
            # quantities but then scattered ``val`` using indices derived
            # from ``valid_mask`` only, dropping the second mask and writing
            # the doubly-masked values onto wrong rows.  We now intersect
            # the two masks consistently before scattering.
            true_indices = xp.where(valid_mask)[0]
            true_indices = true_indices[r_mask]

            r1_m = r1[r_mask]
            r2_m = r2_len[r_mask]
            bwd_m = bwd_v[r_mask]
            r_sum_m = r_sum[r_mask]

            x0 = x1 + r1_m / r_sum_m * (x2 - x1)
            y0 = y1 + r1_m / r_sum_m * (y2 - y1)
            x_v_m = x_v[r_mask]
            y_v_m = y_v[r_mask]
            r0 = xp.sqrt((x0 - x_v_m) ** 2 + (y0 - y_v_m) ** 2)

            def pGauss(r, sig):
                return (1.0 - xp.exp(-(r * r) / (2.0 * sig * sig))) / (2.0 * xp.pi)

            val = sgn_v[r_mask] * (
                pGauss(r1_m, bwd_m) / 6.0 +
                pGauss(r0, bwd_m) * 2.0 / 3.0 +
                pGauss(r2_m, bwd_m) / 6.0
            ) * phi[r_mask]

            poly_int[true_indices] += val

    integ0 = xp.sum(pb * poly_int)

    # -------------------------------------------------------------------------
    # Step 3: conditional intensity lambda_i
    # -------------------------------------------------------------------------
    lam = mu * bk
    invG = 1.0 / G_norm

    # Pre-extract the KDTree query lists once if available.
    if nbr_index is not None:
        all_nbrs = nbr_index.query_all(tau_cut, r_cut)

    for i in range(1, N):
        if nbr_index is not None:
            idx = np.asarray(all_nbrs[i], dtype=np.intp)
            if idx.size == 0:
                continue
            ti = t[idx]; xi = x[idx]; yi = y[idx]; mi = m[idx]
            if is_3d:
                zi = z[idx]
            delta = t[i] - ti
            r2 = (x[i] - xi) ** 2 + (y[i] - yi) ** 2
        else:
            ti = t[:i]; xi = x[:i]; yi = y[:i]; mi = m[:i]
            if is_3d:
                zi = z[:i]
            delta = t[i] - ti
            r2 = (x[i] - xi) ** 2 + (y[i] - yi) ** 2

            if tau_cut is not None and tau_cut < float('inf'):
                mask_t = delta <= tau_cut
            else:
                mask_t = xp.ones(i, dtype=bool)
            if r_cut is not None and r_cut < float('inf'):
                mask_r = xp.sqrt(r2) <= r_cut
            else:
                mask_r = xp.ones(i, dtype=bool)
            mask = mask_t & mask_r
            delta = delta[mask]; r2 = r2[mask]; mi = mi[mask]
            if is_3d:
                zi = zi[mask]
            if delta.size == 0:
                continue

        part1 = xp.exp(alpha * mi)
        part2 = (p - 1.0) / c * (1.0 + delta / c) ** (-p)
        part2 = part2 * invG

        if mver == 1:
            sig = D * xp.exp(gamma * mi)
            part3 = ((q - 1.0) / (sig * xp.pi) * (1.0 + r2 / sig) ** (-q))
            if F_norm is not None:
                # F_norm is indexed by full-catalog parent position. When the
                # KDTree path is used `idx` already holds those positions; on
                # the legacy path we sliced [:i] and then masked, so the
                # surviving parent positions are arange(i)[mask].
                if nbr_index is not None:
                    parent_pos = idx
                else:
                    parent_pos = np.arange(i, dtype=np.intp)[np.asarray(mask)]
                F_sub = xp.asarray(np.asarray(F_norm)[parent_pos])
                part3 = part3 / F_sub
            if is_3d:
                from .backend import get_special
                special = get_special()
                u = z[i] / Z_max
                v = zi / Z_max
                log_h = _depth_log_h_vec(u, v, eta, Z_max, special, xp)
                part3 = part3 * xp.exp(log_h) / H_norm
            lam[i] += xp.sum(A * part1 * part2 * part3)
        else:
            sig_sq = D * xp.exp(gamma * mi)
            part3 = xp.exp(-r2 / (2.0 * sig_sq)) / (2.0 * xp.pi * sig_sq)
            if is_3d:
                from .backend import get_special
                special = get_special()
                u = z[i] / Z_max
                v = zi / Z_max
                log_h = _depth_log_h_vec(u, v, eta, Z_max, special, xp)
                part3 = part3 * xp.exp(log_h) / H_norm
            lam[i] += xp.sum(A * part1 * part2 * part3)

    # Background probability
    prob = xp.where(lam > 0, mu * bk / lam, 1.0)

    # Push results back to the CPU-managed revents array.
    if hasattr(bk, 'get'):
        revents[:, 5] = bk.get()
        revents[:, 6] = prob.get()
        revents[:, 7] = lam.get()
        integ0_val = float(integ0.get())
    else:
        revents[:, 5] = bk
        revents[:, 6] = prob
        revents[:, 7] = lam
        integ0_val = float(integ0)

    return {'revents': revents, 'integ0': integ0_val}
