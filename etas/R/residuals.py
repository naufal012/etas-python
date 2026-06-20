"""
Residual analysis for the ETAS model.

Equivalent of resid.etas.R from the R ETAS package.
Computes transformed times, temporal residuals, and spatial residuals
for model diagnostics.
"""

import numpy as np
import math
import pandas as pd

from ..src.geometry import inside_polygon, longlat2xy, xy2longlat
from ..src.poly_integ import (
    dist_euclidean, dist2_euclidean, poly_integ,
    ffunrint1, ffunrint2, gfunint, kappafun,
    ffun1, ffun2, gfun
)

def _timetransform(theta, revents, rpoly, tperiod, integ0, ndiv, mver):
    """
    Transformed times (tau_i).
    Translated from cxxtimetrans in fitMP.cpp.
    """
    t = revents[:, 0]
    x = revents[:, 1]
    y = revents[:, 2]
    m = revents[:, 3]
    px = rpoly[:, 0]
    py = rpoly[:, 1]
    tstart2 = tperiod[0]
    tlength = tperiod[1]
    N = len(t)
    
    sinteg = np.zeros(N)
    out = np.zeros(N)
    
    if mver == 1:
        mu = theta[0]**2
        A = theta[1]**2
        c = theta[2]**2
        alpha = theta[3]**2
        p = theta[4]**2
        D = theta[5]**2
        q = theta[6]**2
        gamma = theta[7]**2
        kparam = [A, alpha]
        gparam = [c, p]
        fparam = [D, gamma, q]
    else:
        mu = theta[0]**2
        A = theta[1]**2
        c = theta[2]**2
        alpha = theta[3]**2
        p = theta[4]**2
        D = theta[5]**2
        gamma = theta[6]**2
        kparam = [A, alpha]
        gparam = [c, p]
        fparam = [D, gamma]

    # Spatial integrals for each event
    for i in range(N):
        if mver == 1:
            def _ffunrint1_wrap(r, w): return ffunrint1(r, m[i], fparam)
            si = poly_integ(_ffunrint1_wrap, None, px, py, x[i], y[i], ndiv=ndiv)
        else:
            def _ffunrint2_wrap(r, w): return ffunrint2(r, m[i], fparam)
            si = poly_integ(_ffunrint2_wrap, None, px, py, x[i], y[i], ndiv=ndiv)
            
        sinteg[i] = kappafun(m[i], kparam) * si

    for j in range(N):
        s = 0.0
        for i in range(j):
            if t[i] > tstart2:
                s += gfunint(t[j] - t[i], gparam) * sinteg[i]
            else:
                s += (gfunint(t[j] - t[i], gparam) - gfunint(tstart2 - t[i], gparam)) * sinteg[i]
                
        out[j] = mu * integ0 * (t[j] - tstart2) / (tlength - tstart2) + s
        
    return out

def _lambdatemporal(tg, theta, revents, rpoly, tperiod, integ0, ndiv, mver):
    """
    Temporal intensity function: integrating over the spatial domain.
    Translated from cxxlambdtemp in fitMP.cpp.
    """
    t = revents[:, 0]
    x = revents[:, 1]
    y = revents[:, 2]
    m = revents[:, 3]
    px = rpoly[:, 0]
    py = rpoly[:, 1]
    tstart2 = tperiod[0]
    tlength = tperiod[1]
    N = len(t)
    ng = len(tg)
    
    sinteg = np.zeros(N)
    out = np.zeros(ng)
    
    if mver == 1:
        mu = theta[0]**2
        A = theta[1]**2
        c = theta[2]**2
        alpha = theta[3]**2
        p = theta[4]**2
        D = theta[5]**2
        q = theta[6]**2
        gamma = theta[7]**2
        kparam = [A, alpha]
        gparam = [c, p]
        fparam = [D, gamma, q]
    else:
        mu = theta[0]**2
        A = theta[1]**2
        c = theta[2]**2
        alpha = theta[3]**2
        p = theta[4]**2
        D = theta[5]**2
        gamma = theta[6]**2
        kparam = [A, alpha]
        gparam = [c, p]
        fparam = [D, gamma]

    for i in range(N):
        if mver == 1:
            def _ffunrint1_wrap(r, w): return ffunrint1(r, m[i], fparam)
            si = poly_integ(_ffunrint1_wrap, None, px, py, x[i], y[i], ndiv=ndiv)
        else:
            def _ffunrint2_wrap(r, w): return ffunrint2(r, m[i], fparam)
            si = poly_integ(_ffunrint2_wrap, None, px, py, x[i], y[i], ndiv=ndiv)
            
        sinteg[i] = kappafun(m[i], kparam) * si

    for j in range(ng):
        s = 0.0
        for i in range(N):
            if t[i] < tg[j]:
                # In fitMP.cpp: model.gfun0(tg[j] - t[i])
                s += gfun(tg[j] - t[i], gparam) * sinteg[i]
        out[j] = mu * integ0 / (tlength - tstart2) + s
        
    return out

def _lambdaspatial(xg, yg, theta, revents, rpoly, tperiod, bwd, mver):
    """
    Spatial intensity function: integrating over the temporal domain.
    Translated from cxxlambspat in fitMP.cpp.
    """
    t = revents[:, 0]
    x = revents[:, 1]
    y = revents[:, 2]
    m = revents[:, 3]
    pb = revents[:, 6]
    tstart2 = tperiod[0]
    tlength = tperiod[1]
    N = len(t)
    ng = len(xg)
    
    out = np.zeros(ng)
    
    if mver == 1:
        mu = theta[0]**2
        A = theta[1]**2
        c = theta[2]**2
        alpha = theta[3]**2
        p = theta[4]**2
        D = theta[5]**2
        q = theta[6]**2
        gamma = theta[7]**2
        kparam = [A, alpha]
        gparam = [c, p]
        fparam = [D, gamma, q]
    else:
        mu = theta[0]**2
        A = theta[1]**2
        c = theta[2]**2
        alpha = theta[3]**2
        p = theta[4]**2
        D = theta[5]**2
        gamma = theta[6]**2
        kparam = [A, alpha]
        gparam = [c, p]
        fparam = [D, gamma]

    for j in range(ng):
        s = 0.0
        s1 = 0.0
        s2 = 0.0
        for i in range(N):
            if t[i] > tstart2:
                gint = gfunint(tlength - t[i], gparam)
            else:
                gint = gfunint(tlength - t[i], gparam) - gfunint(tstart2 - t[i], gparam)
                
            r2 = dist2_euclidean(xg[j], yg[j], x[i], y[i])
            
            if mver == 1:
                fval = ffun1(r2, m[i], fparam)
            else:
                fval = ffun2(r2, m[i], fparam)
                
            s += kappafun(m[i], kparam) * gint * fval
            
            # dGauss
            sig = bwd[i]
            d_gauss = math.exp(-r2/(2 * sig * sig)) / (2 * math.pi * sig * sig)
            s1 += d_gauss
            s2 += pb[i] * d_gauss
            
        out[j] = s + mu * s2 / (tlength - tstart2)
        
    return out

def _smooth_marks(xg, yg, marks, gx, gy, sigma):
    """
    Nadaraya-Watson kernel smoothing of marks onto a grid.
    Approximate equivalent of spatstat.explore::Smooth(Xg, sigma).
    """
    nx = len(gx)
    ny = len(gy)
    out = np.zeros((ny, nx))
    
    # Pre-calculate to avoid huge memory, do it per grid point
    for i in range(nx):
        for j in range(ny):
            r2 = (xg - gx[i])**2 + (yg - gy[j])**2
            weights = np.exp(-r2 / (2.0 * sigma**2))
            sum_w = np.sum(weights)
            if sum_w > 0:
                out[j, i] = np.sum(marks * weights) / sum_w
            else:
                out[j, i] = 0.0
    return out

def residuals(fit, type="raw", n_temp=1000, dimyx=None):
    """
    Compute residuals for a fitted ETAS model.

    Parameters
    ----------
    fit : ETASResult
        Fitted ETAS model from etas().
    type : str
        Type of residual: "raw", "reciprocal", or "pearson".
    n_temp : int
        Number of points for temporal residual evaluation.
    dimyx : tuple
        Dimensions for spatial residual grid (ny, nx).

    Returns
    -------
    dict
        Dictionary containing tau, U, tres (temporal residuals), 
        sres (spatial residuals dataframe), and type.
    """
    flg = fit.catalog.revents[:, 4] == 1
    tt = fit.catalog.revents[flg, 0]
    xx = fit.catalog.revents[flg, 1]
    yy = fit.catalog.revents[flg, 2]
    
    theta = np.sqrt(fit.param)
    
    # 1. Transformed times
    tau_all = _timetransform(
        theta, fit.catalog.revents, fit.catalog.rpoly, 
        fit.catalog.rtperiod, fit.integ0, fit.ndiv, fit.mver)
        
    # Baseline integral for start of period
    t0_arr = np.array([fit.catalog.rtperiod[0]])
    lam_t0 = _lambdatemporal(
        t0_arr, theta, fit.catalog.revents, fit.catalog.rpoly,
        fit.catalog.rtperiod, fit.integ0, fit.ndiv, fit.mver)[0]
        
    tau = tau_all[flg] - lam_t0
    
    # 2. Temporal residuals
    tg = np.linspace(tt.min(), tt.max(), n_temp)
    tlam = _lambdatemporal(
        tg, theta, fit.catalog.revents, fit.catalog.rpoly,
        fit.catalog.rtperiod, fit.integ0, fit.ndiv, fit.mver)
        
    dfun = np.array([np.sum((tt <= tg[i]) & (tt > tg[i-1])) for i in range(1, n_temp)])
    dtg = np.diff(tg)
    
    if type == "raw":
        tres = dfun - tlam[1:] * dtg
    elif type == "reciprocal":
        tres = 1.0 / tlam[1:] - dtg
    elif type == "pearson":
        tres = 1.0 / np.sqrt(tlam[1:]) - np.sqrt(tlam[1:]) * dtg
    else:
        raise ValueError("type must be raw, reciprocal, or pearson")
        
    # 3. Spatial residuals (Approximation of spatstat quadscheme)
    # We create a dummy grid inside the polygon, and append data points
    if dimyx is None:
        dx_val = np.ptp(fit.catalog.rpoly[:, 0])
        dy_val = np.ptp(fit.catalog.rpoly[:, 1])
        rv = dx_val / dy_val if dy_val > 0 else 1.0
        if rv > 1:
            dimyx = (128, round(rv * 128))
        else:
            dimyx = (round(128 / rv), 128)
            
    # Dummy grid points
    grid_x = np.linspace(fit.catalog.rpoly[:, 0].min(), fit.catalog.rpoly[:, 0].max(), dimyx[1])
    grid_y = np.linspace(fit.catalog.rpoly[:, 1].min(), fit.catalog.rpoly[:, 1].max(), dimyx[0])
    gx_mat, gy_mat = np.meshgrid(grid_x, grid_y)
    
    dummy_x = gx_mat.flatten()
    dummy_y = gy_mat.flatten()
    
    # Filter dummy points to those inside polygon
    if fit.catalog.region_win is not None:
        in_poly = inside_polygon(dummy_x, dummy_y, fit.catalog.region_win)
        dummy_x = dummy_x[in_poly]
        dummy_y = dummy_y[in_poly]
        
    cell_area = (grid_x[1] - grid_x[0]) * (grid_y[1] - grid_y[0])
    
    # Combine data and dummy
    xg = np.concatenate([xx, dummy_x])
    yg = np.concatenate([yy, dummy_y])
    wg = np.concatenate([np.zeros(len(xx)), np.full(len(dummy_x), cell_area)])
    zg = np.concatenate([np.ones(len(xx)), np.zeros(len(dummy_x))])
    
    # Calculate spatial intensity at quad points
    slam = _lambdaspatial(
        xg, yg, theta, fit.catalog.revents, fit.catalog.rpoly,
        fit.catalog.rtperiod, fit.bwd, fit.mver)
        
    if type == "raw":
        sres_marks = zg - slam * wg
    elif type == "reciprocal":
        # Avoid division by zero
        slam_safe = np.where(slam > 0, slam, np.inf)
        sres_marks = zg / slam_safe - wg
    elif type == "pearson":
        slam_safe = np.where(slam > 0, slam, np.inf)
        sres_marks = zg / np.sqrt(slam_safe) - np.sqrt(slam) * wg
        
    # Smooth spatial residuals back onto the grid
    sigma = np.mean(fit.bwd)
    sres_grid = _smooth_marks(xg, yg, sres_marks, grid_x, grid_y, sigma)
    
    # Project grid back to long/lat for dataframe
    gr_x = gx_mat.flatten()
    gr_y = gy_mat.flatten()
    proj = xy2longlat(gr_x, gr_y, fit.catalog.region_poly, fit.catalog.dist_unit)
    
    sres_df = pd.DataFrame({
        'x': proj['long'],
        'y': proj['lat'],
        'z': sres_grid.flatten()
    })
    
    # For U (uniform) Q-Q plot
    tau_diff = np.diff(tau)
    U = 1.0 - np.exp(-tau_diff)
    
    return {
        'tau': tau,
        'U': U,
        'tg': tg,
        'tres': tres,
        'sres': sres_df,
        'type': type
    }
