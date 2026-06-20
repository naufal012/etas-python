"""
Spatial seismicity rate maps for the ETAS model.

Equivalent of rates.R from the R ETAS package. Computes background,
total, and clustering rates on a spatial grid, plus the conditional
intensity function at the end of the study period.
"""

import numpy as np
from ..src.geometry import longlat2xy


def rates(fit, lat_range=None, long_range=None, dimyx=None, slice_depth=None):
    """Compute spatial seismicity rate maps from a fitted ETAS model.

    Parameters
    ----------
    fit : ETASResult
        Fitted ETAS model from etas().
    lat_range : tuple of (lat_min, lat_max), optional
    long_range : tuple of (long_min, long_max), optional
    dimyx : tuple of (ny, nx), optional
        Grid dimensions. If None, auto-computed.
    slice_depth : float, optional
        For 3D models, evaluate the intensity at this specific depth.

    Returns
    -------
    dict
        Keys: 'x' (longitudes), 'y' (latitudes),
        'bkgd' (background rate), 'total' (total rate),
        'clust' (clustering coefficient), 'lamb' (conditional intensity).
    """
    from ..src.poly_integ import (
        ffun1, ffun2, gfun, kappafun, dist2_euclidean
    )
    from ..src.renorm import compute_all_norms

    catalog_obj = fit.catalog
    param = fit.param
    revents = catalog_obj.revents
    bwd = fit.bwd
    mver = fit.mver
    region_poly = catalog_obj.region_poly
    dist_unit = catalog_obj.dist_unit

    t = revents[:, 0]
    x = revents[:, 1]
    y = revents[:, 2]
    m = revents[:, 3]
    pb = revents[:, 6]

    # Robust 3D inference from the parameter NAMES rather than guessing from
    # the array length (the previous heuristic was off-by-one for mver=2).
    is_3d = ('eta' in fit.par_names)
    if is_3d:
        z = revents[:, 8]
        Z_max = np.max(z)

    tstart2 = catalog_obj.rtperiod[0]
    tlength = catalog_obj.rtperiod[1]
    N = len(t)

    # Model parameters
    mu = param[0]
    A = param[1]
    c = param[2]
    alpha = param[3]
    p = param[4]
    D = param[5]

    if mver == 1:
        q = param[6]
        gamma = param[7]
        fparam = [D, gamma, q]
        if is_3d:
            eta = param[8]
    else:
        gamma = param[6]
        fparam = [D, gamma]
        if is_3d:
            eta = param[7]

    kparam = [A, alpha]
    gparam = [c, p]

    # Renormalization constants from the fitted parameters.  When the fit used
    # no truncation these are all 1 (no-op), preserving the legacy output.
    norms = compute_all_norms(param, m, mver,
                              eps_t=getattr(fit, 'eps_t', None),
                              eps_s=getattr(fit, 'eps_s', None),
                              eps_z=getattr(fit, 'eps_z', None),
                              Z_max=Z_max if is_3d else None)
    G_norm = norms['G_norm']
    F_norm = norms['F_norm']
    H_norm = norms['H_norm']

    # Spatial extent
    if lat_range is None:
        lat_range = (region_poly['lat'].min(), region_poly['lat'].max())
    if long_range is None:
        long_range = (region_poly['long'].min(), region_poly['long'].max())

    # Project boundary to flat map
    xy_bnd = longlat2xy(
        np.array([long_range[0], long_range[1],
                  long_range[1], long_range[0]]),
        np.array([lat_range[0], lat_range[0],
                  lat_range[1], lat_range[1]]),
        region_poly, dist_unit)

    if dimyx is None:
        dx = np.ptp(xy_bnd['x'])
        dy = np.ptp(xy_bnd['y'])
        rv = dx / dy if dy > 0 else 1.0
        if rv > 1:
            dimyx = (128, round(128 * rv))
        else:
            dimyx = (round(128 / rv), 128)

    gx = np.linspace(xy_bnd['x'].min(), xy_bnd['x'].max(), dimyx[1])
    gy = np.linspace(xy_bnd['y'].min(), xy_bnd['y'].max(), dimyx[0])

    # Compute rates on grid
    bkgd = np.zeros((dimyx[1], dimyx[0]))
    total = np.zeros((dimyx[1], dimyx[0]))
    clust = np.zeros((dimyx[1], dimyx[0]))
    lamb = np.zeros((dimyx[1], dimyx[0]))

    for i in range(dimyx[1]):
        for j in range(dimyx[0]):
            sum1 = 0.0
            sum2 = 0.0

            for l in range(N):
                r2 = dist2_euclidean(x[l], y[l], gx[i], gy[j])
                sig = bwd[l]
                tmp = (np.exp(-r2 / (2.0 * sig * sig)) /
                       (2.0 * np.pi * sig * sig))
                sum1 += pb[l] * tmp
                sum2 += tmp

            bkgd[i, j] = sum1 / (tlength - tstart2)
            total[i, j] = sum2 / (tlength - tstart2)
            clust[i, j] = 1.0 - sum1 / sum2 if sum2 > 0 else 0.0
            lamb[i, j] = mu * bkgd[i, j]

            for l in range(N):
                r2 = dist2_euclidean(x[l], y[l], gx[i], gy[j])
                kappa_val = kappafun(m[l], kparam)
                g_val = gfun(tlength - t[l], gparam) / G_norm
                if mver == 1:
                    f_val = ffun1(r2, m[l], fparam) / F_norm[l]
                else:
                    f_val = ffun2(r2, m[l], fparam)

                if is_3d and slice_depth is not None:
                    import scipy.special as special
                    u = slice_depth / Z_max
                    v = z[l] / Z_max
                    log_beta = special.gammaln(eta * v + 1.0) + special.gammaln(eta * (1.0 - v) + 1.0) - special.gammaln(eta + 2.0)
                    safe_u = max(float(u), 1e-12)
                    safe_1_u = max(1.0 - float(u), 1e-12)
                    log_h = (eta * v) * np.log(safe_u) + (eta * (1.0 - v)) * np.log(safe_1_u) - np.log(Z_max) - log_beta
                    f_val = f_val * np.exp(log_h) / H_norm

                lamb[i, j] += kappa_val * g_val * f_val

    # Output coordinates in lon/lat
    out_x = np.linspace(long_range[0], long_range[1], dimyx[1])
    out_y = np.linspace(lat_range[0], lat_range[1], dimyx[0])

    return {
        'x': out_x, 'y': out_y,
        'bkgd': bkgd, 'total': total,
        'clust': clust, 'lamb': lamb
    }


def probs(fit):
    """Extract declustering probabilities.

    Parameters
    ----------
    fit : ETASResult
        Fitted ETAS model.

    Returns
    -------
    dict
        Keys: 'long', 'lat', 'prob' (probability of being triggered),
        'target' (bool, whether inside study region).
    """
    catalog_obj = fit.catalog
    longlat = catalog_obj.longlat_coord

    # Probability of being a triggered event (1 - background prob)
    pb = 1.0 - catalog_obj.revents[:, 6]

    return {
        'long': longlat['long'].values,
        'lat': longlat['lat'].values,
        'prob': pb,
        'target': catalog_obj.revents[:, 4] == 1
    }
