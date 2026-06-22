"""
Spatial seismicity rate maps for the ETAS model.

Equivalent of rates.R from the R ETAS package. Computes background,
total, and clustering rates on a spatial grid, plus the conditional
intensity function at the end of the study period.
"""

import numpy as np
from ..src.geometry import longlat2xy
from ..src.backend import get_xp


def rates(fit, lat_range=None, long_range=None, dimyx=None, slice_depth=None,
          plot=False):
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
    plot : bool, optional
        If True, produce a geographic map with coastlines, borders and
        land/ocean shading using cartopy (falls back to plain matplotlib
        if cartopy is not installed).  The returned dict includes ``'fig'``
        and ``'axes'`` keys.

    Returns
    -------
    dict
        Keys: ``'x'`` (longitudes), ``'y'`` (latitudes),
        ``'bkgd'`` (background rate), ``'total'`` (total rate),
        ``'clust'`` (clustering coefficient), ``'lamb'`` (conditional intensity).
        When ``plot=True``, additionally ``'fig'`` (matplotlib Figure) and
        ``'axes'`` (list of Axes).
    """
    from ..src.poly_integ import (
        ffun1, ffun2, gfun, kappafun, dist2_euclidean
    )
    from ..src.renorm import compute_all_norms

    xp = get_xp()

    catalog_obj = fit.catalog
    param = fit.param
    revents = catalog_obj.revents
    bwd = fit.bwd
    mver = fit.mver
    region_poly = catalog_obj.region_poly
    dist_unit = catalog_obj.dist_unit

    t = xp.asarray(revents[:, 0])
    x = xp.asarray(revents[:, 1])
    y = xp.asarray(revents[:, 2])
    m = xp.asarray(revents[:, 3])
    pb = xp.asarray(revents[:, 6])

    # Robust 3D inference from the parameter NAMES rather than guessing from
    # the array length (the previous heuristic was off-by-one for mver=2).
    is_3d = ('eta' in fit.par_names)
    if is_3d:
        z = xp.asarray(revents[:, 8])
        Z_max = float(xp.max(z))

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

    gx = xp.linspace(xy_bnd['x'].min(), xy_bnd['x'].max(), dimyx[1])
    gy = xp.linspace(xy_bnd['y'].min(), xy_bnd['y'].max(), dimyx[0])

    # ------------------------------------------------------------------
    # Vectorized rate computation: broadcast grid x events
    # ------------------------------------------------------------------
    gx_mesh, gy_mesh = xp.meshgrid(gx, gy, indexing='ij')
    gx_flat = gx_mesh.ravel()
    gy_flat = gy_mesh.ravel()

    dx_mat = gx_flat[:, None] - x[None, :]
    dy_mat = gy_flat[:, None] - y[None, :]
    r2_mat = dx_mat ** 2 + dy_mat ** 2

    sig = xp.asarray(bwd)[None, :]
    tmp = xp.exp(-r2_mat / (2.0 * sig * sig)) / (2.0 * xp.pi * sig * sig)

    sum1 = xp.sum(xp.asarray(pb)[None, :] * tmp, axis=1)
    sum2 = xp.sum(tmp, axis=1)

    bkgd_flat = sum1 / (tlength - tstart2)
    total_flat = sum2 / (tlength - tstart2)

    lamb_flat = mu * bkgd_flat

    for l in range(N):
        r2_l = r2_mat[:, l]
        kappa_val = kappafun(m[l], kparam)
        g_val = gfun(tlength - float(t[l]), gparam) / G_norm
        if mver == 1:
            f_val = ffun1(r2_l, m[l], fparam)
            F_sub = xp.asarray(F_norm)[l]
            f_val = f_val / F_sub
        else:
            f_val = ffun2(r2_l, m[l], fparam)

        if is_3d and slice_depth is not None:
            from ..src.backend import get_special
            special = get_special()
            u = slice_depth / Z_max
            v_l = float(z[l]) / Z_max
            log_beta = (special.gammaln(eta * v_l + 1.0)
                        + special.gammaln(eta * (1.0 - v_l) + 1.0)
                        - special.gammaln(eta + 2.0))
            safe_u = max(float(u), 1e-12)
            safe_1_u = max(1.0 - float(u), 1e-12)
            log_h = ((eta * v_l) * np.log(safe_u)
                     + (eta * (1.0 - v_l)) * np.log(safe_1_u)
                     - np.log(Z_max) - log_beta)
            f_val = f_val * np.exp(log_h) / H_norm

        lamb_flat += kappa_val * g_val * f_val

    clust_flat = xp.where(sum2 > 0, 1.0 - sum1 / sum2, 0.0)

    bkgd = xp.asarray(bkgd_flat).reshape(dimyx[1], dimyx[0])
    total = xp.asarray(total_flat).reshape(dimyx[1], dimyx[0])
    clust = xp.asarray(clust_flat).reshape(dimyx[1], dimyx[0])
    lamb = xp.asarray(lamb_flat).reshape(dimyx[1], dimyx[0])

    out_x = np.linspace(long_range[0], long_range[1], dimyx[1])
    out_y = np.linspace(lat_range[0], lat_range[1], dimyx[0])

    result = {
        'x': out_x, 'y': out_y,
        'bkgd': bkgd, 'total': total,
        'clust': clust, 'lamb': lamb
    }

    # ------------------------------------------------------------------
    # Optional cartopy map overlay
    # ------------------------------------------------------------------
    if plot:
        result.update(_make_map(out_x, out_y, bkgd, clust, lamb))

    return result


def _make_map(lon, lat, bkgd, clust, lamb):
    """Build a 3-panel cartopy map from rate arrays.

    Falls back to plain matplotlib if cartopy is not installed.
    """
    # Convert from GPU if needed
    bkgd_np  = bkgd.get() if hasattr(bkgd, 'get') else bkgd
    clust_np = clust.get() if hasattr(clust, 'get') else clust
    lamb_np  = lamb.get() if hasattr(lamb, 'get') else lamb

    try:
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature
        _has_cartopy = True
    except ImportError:
        _has_cartopy = False

    import matplotlib.pyplot as plt

    if _has_cartopy:
        fig, axes = plt.subplots(1, 3, figsize=(20, 7),
                                  subplot_kw={'projection': ccrs.PlateCarree()})
    else:
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    titles = ['Background Rate', 'Clustering Coefficient', 'Conditional Intensity']
    datas  = [bkgd_np, clust_np, lamb_np]
    cmaps  = ['Reds', 'coolwarm', 'inferno']

    for ax, title, data, cmap in zip(axes, titles, datas, cmaps):
        if _has_cartopy:
            im = ax.pcolormesh(lon, lat, data.T, cmap=cmap,
                               transform=ccrs.PlateCarree(), shading='auto')
            ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
            ax.add_feature(cfeature.BORDERS,  linewidth=0.3, alpha=0.6)
            ax.add_feature(cfeature.OCEAN,    facecolor='#e8f4f8', zorder=0)
            ax.add_feature(cfeature.LAND,     facecolor='#f5f0e8', zorder=0)
            gl = ax.gridlines(draw_labels=True, dms=True,
                              x_inline=False, y_inline=False,
                              linewidth=0.3, color='gray', alpha=0.5)
            gl.top_labels = False
            gl.right_labels = False
            ax.set_extent([lon.min(), lon.max(), lat.min(), lat.max()])
        else:
            im = ax.pcolormesh(lon, lat, data.T, cmap=cmap, shading='auto')
            ax.set_xlabel('Longitude')
            ax.set_ylabel('Latitude')

        ax.set_title(title, fontsize=11)
        plt.colorbar(im, ax=ax, shrink=0.65)

    study_label = ""
    fig.suptitle(f'ETAS Spatial Rate Maps{study_label}', fontsize=13, fontweight='bold')
    fig.tight_layout()

    return {'fig': fig, 'axes': axes}


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
