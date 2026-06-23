"""
Numerical integration of radially symmetric functions over a polygon region.

Pure-Python/NumPy translation of poly.c, lambda.c, and funcs.h from the
ETAS R package.  All functions operate on scalars unless otherwise noted.
"""

import math
import numpy as np

# ---------------------------------------------------------------------------
# Distance helpers
# ---------------------------------------------------------------------------

def dist_euclidean(x1, y1, x2, y2):
    """Euclidean distance between two points."""
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


def dist2_euclidean(x1, y1, x2, y2):
    """Squared Euclidean distance between two points."""
    return (x1 - x2) ** 2 + (y1 - y2) ** 2


# ---------------------------------------------------------------------------
# Triangle contribution (translated from poly.c)
# ---------------------------------------------------------------------------

def frint(func, funcpara, x1, y1, x2, y2, cx, cy):
    """
    Compute the contribution of one triangle to the polygon integral.

    Parameters
    ----------
    func : callable
        Radially symmetric function  ``func(r, funcpara) -> float``.
    funcpara : array-like
        Parameters forwarded to *func*.
    x1, y1 : float
        First vertex of the polygon edge.
    x2, y2 : float
        Second vertex of the polygon edge.
    cx, cy : float
        Centre point (pole of the radial function).

    Returns
    -------
    float
        Signed contribution of the triangle to the integral.
    """
    # Signed area (2×) of the triangle (cx,cy)-(x1,y1)-(x2,y2)
    det = (x1 * y2 + y1 * cx + x2 * cy) - (x2 * y1 + y2 * cx + x1 * cy)

    sign = -1 if det < 0 else 1

    if abs(det) < 1e-10:
        return 0.0

    r1 = dist_euclidean(x1, y1, cx, cy)
    r2 = dist_euclidean(x2, y2, cx, cy)
    r12 = dist_euclidean(x1, y1, x2, y2)

    # Angle at the centre via the cosine rule
    denom = 2.0 * r1 * r2
    if denom == 0.0:
        return 0.0
    theta = (r1 * r1 + r2 * r2 - r12 * r12) / denom
    if abs(theta) > 1.0:
        theta = 1.0 - 1e-10
    theta = math.acos(theta)

    # Foot of the perpendicular from (cx,cy) onto the edge
    if r1 + r2 > 1e-20:
        frac = r1 / (r1 + r2)
        x0 = x1 + frac * (x2 - x1)
        y0 = y1 + frac * (y2 - y1)
    else:
        return 0.0

    r0 = dist_euclidean(x0, y0, cx, cy)

    # Simpson-like quadrature over the angle
    f1 = func(r1, funcpara)
    f2 = func(r0, funcpara)
    f3 = func(r2, funcpara)

    return sign * (f1 / 6.0 + f2 * 2.0 / 3.0 + f3 / 6.0) * theta


# ---------------------------------------------------------------------------
# Polygon integration (translated from poly.c)
# ---------------------------------------------------------------------------

from .backend import get_xp

def poly_integ(func, funcpara, px, py, cx, cy, ndiv=1000):
    """
    Integrate a radially symmetric function over a closed polygon.
    Vectorized for CPU/GPU over the polygon boundary segments.
    """
    xp = get_xp()
    
    px = xp.asarray(px, dtype=float)
    py = xp.asarray(py, dtype=float)
    nv = len(px) - 1

    total = 0.0
    # Hoist the arange out of the edge loop — one allocation instead of N.
    k = xp.arange(ndiv, dtype=float)
    t0 = k / ndiv
    t1 = (k + 1.0) / ndiv

    for j in range(nv):
        # Create vectors for all ndiv segments of this edge
        sx1 = px[j] + t0 * (px[j + 1] - px[j])
        sy1 = py[j] + t0 * (py[j + 1] - py[j])
        sx2 = px[j] + t1 * (px[j + 1] - px[j])
        sy2 = py[j] + t1 * (py[j + 1] - py[j])
        
        # Calculate determinant for all segments
        det = (sx1 * sy2 + sy1 * cx + sx2 * cy) - (sx2 * sy1 + sy2 * cx + sx1 * cy)
        
        # Sign of determinant
        sign = xp.where(det < 0, -1.0, 1.0)
        
        valid_mask = xp.abs(det) >= 1e-10
        if not xp.any(valid_mask):
            continue
            
        x1_v = sx1[valid_mask]
        y1_v = sy1[valid_mask]
        x2_v = sx2[valid_mask]
        y2_v = sy2[valid_mask]
        sign_v = sign[valid_mask]
        
        r1_sq = (x1_v - cx)**2 + (y1_v - cy)**2
        r2_sq = (x2_v - cx)**2 + (y2_v - cy)**2
        r12_sq = (x1_v - x2_v)**2 + (y1_v - y2_v)**2
        
        r1 = xp.sqrt(r1_sq)
        r2 = xp.sqrt(r2_sq)
        
        denom = 2.0 * r1 * r2
        theta = xp.where(denom == 0, 0.0, (r1_sq + r2_sq - r12_sq) / denom)
        theta = xp.clip(theta, -1.0 + 1e-10, 1.0 - 1e-10)
        theta = xp.arccos(theta)
        
        r_sum = r1 + r2
        r_mask = r_sum > 1e-20
        
        if not xp.any(r_mask):
            continue
            
        r1_m = r1[r_mask]
        r2_m = r2[r_mask]
        theta_m = theta[r_mask]
        sign_m = sign_v[r_mask]
        x1_m = x1_v[r_mask]
        x2_m = x2_v[r_mask]
        y1_m = y1_v[r_mask]
        y2_m = y2_v[r_mask]
        r_sum_m = r_sum[r_mask]
        
        frac = r1_m / r_sum_m
        x0 = x1_m + frac * (x2_m - x1_m)
        y0 = y1_m + frac * (y2_m - y1_m)
        r0 = xp.sqrt((x0 - cx)**2 + (y0 - cy)**2)
        
        # The func itself needs to process xp arrays now!
        f1 = func(r1_m, funcpara)
        f2 = func(r0, funcpara)
        f3 = func(r2_m, funcpara)
        
        val = sign_m * (f1 / 6.0 + f2 * 2.0 / 3.0 + f3 / 6.0) * theta_m
        # Use .item() instead of float(xp.sum(...)) to avoid double-sync on CuPy.
        # .item() is equivalent but more idiomatic; on NumPy there is no difference.
        total += val.sum().item()
        
    return total


# ===================================================================
# Spatial CDF functions  (translated from lambda.c, lines 10-32)
# ===================================================================

def fr(r, w):
    """
    Power-law spatial CDF.

    Parameters
    ----------
    r : float
        Radius.
    w : array-like
        ``[gamma, D, q, mag]``.

    Returns
    -------
    float
    """
    gamma, D, q, mag = w[0], w[1], w[2], w[3]
    sig = D * get_xp().exp(gamma * mag)
    return (1.0 - (1.0 + r * r / sig) ** (1.0 - q)) / (2.0 * get_xp().pi)


def dgamma_fr(r, w):
    """Derivative of ``fr`` w.r.t. gamma."""
    gamma, D, q, mag = w[0], w[1], w[2], w[3]
    sig = D * get_xp().exp(gamma * mag)
    base = 1.0 + r * r / sig
    # d/dgamma sig = sig * mag
    # d/dgamma (1+r²/sig)^(1-q) = (1-q)*(1+r²/sig)^(-q) * (-r²/sig²) * sig*mag
    #                             = -(1-q)*mag*r²/sig * base^(-q)
    return (1.0 - q) * mag * (r * r / sig) * base ** (-q) / (2.0 * get_xp().pi)


def dD_fr(r, w):
    """Derivative of ``fr`` w.r.t. D."""
    gamma, D, q, mag = w[0], w[1], w[2], w[3]
    sig = D * get_xp().exp(gamma * mag)
    base = 1.0 + r * r / sig
    # d/dD sig = exp(gamma*mag)
    # d/dD (1+r²/sig)^(1-q) = (1-q)*base^(-q)*(-r²/sig²)*exp(gamma*mag)
    #                        = -(1-q)*r²/(sig*D) * base^(-q)
    return (1.0 - q) * (r * r / (sig * D)) * base ** (-q) / (2.0 * get_xp().pi)


def dq_fr(r, w):
    """Derivative of ``fr`` w.r.t. q."""
    gamma, D, q, mag = w[0], w[1], w[2], w[3]
    sig = D * get_xp().exp(gamma * mag)
    base = 1.0 + r * r / sig
    xp = get_xp()
    return xp.where(base > 0.0, base ** (1.0 - q) * xp.log(base) / (2.0 * xp.pi), 0.0)


def pGauss(r, w):
    """
    Gaussian spatial CDF.

    Parameters
    ----------
    r : float
        Radius.
    w : array-like
        ``w[0]`` is the standard-deviation parameter sigma.

    Returns
    -------
    float
    """
    sig = w[0]
    return (1.0 - get_xp().exp(-r * r / (2.0 * sig * sig))) / (2.0 * get_xp().pi)


# ===================================================================
# Spatial density functions  (translated from funcs.h)
# ===================================================================

def ffun1(r2, m, fparam):
    """
    Power-law spatial density.

    Parameters
    ----------
    r2 : float
        Squared distance.
    m : float
        Magnitude.
    fparam : array-like
        ``[D, gamma, q]``.

    Returns
    -------
    float
    """
    D, gamma, q = fparam[0], fparam[1], fparam[2]
    sig = D * get_xp().exp(gamma * m)
    return (q - 1.0) / (sig * get_xp().pi) * (1.0 + r2 / sig) ** (-q)


def dffun1(r2, m, fparam):
    """
    Power-law spatial density and its partial derivatives.

    Returns
    -------
    list of float
        ``[value, d_D, d_q, d_gamma]``
    """
    D, gamma, q = fparam[0], fparam[1], fparam[2]
    sig = D * get_xp().exp(gamma * m)
    base = 1.0 + r2 / sig
    val = (q - 1.0) / (sig * get_xp().pi) * base ** (-q)

    # d/dD
    dsig_dD = get_xp().exp(gamma * m)
    # val = (q-1)/(pi) * sig^{-1} * base^{-q}
    # d/dD = (q-1)/pi * [ -sig^{-2}*dsig_dD * base^{-q}
    #                      + sig^{-1} * (-q)*base^{-q-1}*(-r2/sig^2)*dsig_dD ]
    d_D = (-(q - 1.0) / (sig * sig * get_xp().pi) * base ** (-q)
           + (q - 1.0) * q * r2 / (sig * sig * sig * get_xp().pi) * base ** (-q - 1.0)) * dsig_dD

    # d/dq
    d_q = (1.0 / (sig * get_xp().pi) * base ** (-q)
           + (q - 1.0) / (sig * get_xp().pi) * base ** (-q) * (-get_xp().log(base)))

    # d/dgamma
    dsig_dgamma = sig * m
    d_gamma = (-(q - 1.0) / (sig * sig * get_xp().pi) * base ** (-q)
               + (q - 1.0) * q * r2 / (sig * sig * sig * get_xp().pi) * base ** (-q - 1.0)) * dsig_dgamma

    return [val, d_D, d_q, d_gamma]


def ffunrint1(r, m, fparam):
    """
    Radial integral of the power-law density (spatial CDF form).

    Parameters
    ----------
    r : float
        Radius.
    m : float
        Magnitude.
    fparam : array-like
        ``[D, gamma, q]``.

    Returns
    -------
    float
    """
    D, gamma, q = fparam[0], fparam[1], fparam[2]
    sig = D * get_xp().exp(gamma * m)
    return (1.0 - (1.0 + r * r / sig) ** (1.0 - q)) / (2.0 * get_xp().pi)


def dffunrint1(r, m, fparam):
    """
    Radial integral of power-law density and its partial derivatives.

    Returns
    -------
    list of float
        ``[value, d_D, d_q, d_gamma]``
    """
    D, gamma, q = fparam[0], fparam[1], fparam[2]
    sig = D * get_xp().exp(gamma * m)
    base = 1.0 + r * r / sig
    val = (1.0 - base ** (1.0 - q)) / (2.0 * get_xp().pi)

    # d/dD
    dsig_dD = get_xp().exp(gamma * m)
    # base^{1-q} derivative w.r.t. D:
    #   (1-q)*base^{-q}*(-r²/sig²)*dsig_dD
    db_dD = (1.0 - q) * base ** (-q) * (-r * r / (sig * sig)) * dsig_dD
    d_D = -db_dD / (2.0 * get_xp().pi)

    # d/dq
    # d/dq base^{1-q} = -base^{1-q} * ln(base)
    d_q = base ** (1.0 - q) * get_xp().log(base) / (2.0 * get_xp().pi) if base > 0.0 else 0.0

    # d/dgamma
    dsig_dgamma = sig * m
    db_dgamma = (1.0 - q) * base ** (-q) * (-r * r / (sig * sig)) * dsig_dgamma
    d_gamma = -db_dgamma / (2.0 * get_xp().pi)

    return [val, d_D, d_q, d_gamma]


def ffun2(r2, m, fparam):
    """
    Gaussian spatial density.

    Parameters
    ----------
    r2 : float
        Squared distance.
    m : float
        Magnitude.
    fparam : array-like
        ``[D, gamma]``.

    Returns
    -------
    float
    """
    D, gamma = fparam[0], fparam[1]
    sig = D * get_xp().exp(gamma * m)
    return get_xp().exp(-r2 / (2.0 * sig * sig)) / (2.0 * get_xp().pi * sig * sig)


def dffun2(r2, m, fparam):
    """
    Gaussian spatial density and its partial derivatives.

    Returns
    -------
    list of float
        ``[value, d_D, d_gamma]``
    """
    D, gamma = fparam[0], fparam[1]
    sig = D * get_xp().exp(gamma * m)
    sig2 = sig * sig
    val = get_xp().exp(-r2 / (2.0 * sig2)) / (2.0 * get_xp().pi * sig2)

    # d/dD  (sig = D*exp(gamma*m), dsig/dD = exp(gamma*m))
    dsig_dD = get_xp().exp(gamma * m)
    # d/dsig val = val * (r2/sig^3 - 2/sig)  =>  d/dD = d/dsig * dsig/dD
    d_D = val * (r2 / (sig2 * sig) - 2.0 / sig) * dsig_dD

    # d/dgamma  (dsig/dgamma = sig*m)
    dsig_dgamma = sig * m
    d_gamma = val * (r2 / (sig2 * sig) - 2.0 / sig) * dsig_dgamma

    return [val, d_D, d_gamma]


def ffunrint2(r, m, fparam):
    """
    Radial integral of the Gaussian density (spatial CDF form).

    Parameters
    ----------
    r : float
        Radius.
    m : float
        Magnitude.
    fparam : array-like
        ``[D, gamma]``.

    Returns
    -------
    float
    """
    D, gamma = fparam[0], fparam[1]
    sig = D * get_xp().exp(gamma * m)
    return (1.0 - get_xp().exp(-r * r / (2.0 * sig * sig))) / (2.0 * get_xp().pi)


def dffunrint2(r, m, fparam):
    """
    Radial integral of the Gaussian density and its partial derivatives.

    Returns
    -------
    list of float
        ``[value, d_D, d_gamma]``
    """
    D, gamma = fparam[0], fparam[1]
    sig = D * get_xp().exp(gamma * m)
    sig2 = sig * sig
    expterm = get_xp().exp(-r * r / (2.0 * sig2))
    val = (1.0 - expterm) / (2.0 * get_xp().pi)

    # d/dD
    dsig_dD = get_xp().exp(gamma * m)
    # d/dD expterm = expterm * r²/(sig^3) * dsig_dD  (from chain rule)
    d_D = -expterm * r * r / (sig2 * sig) * dsig_dD / (2.0 * get_xp().pi)

    # d/dgamma
    dsig_dgamma = sig * m
    d_gamma = -expterm * r * r / (sig2 * sig) * dsig_dgamma / (2.0 * get_xp().pi)

    return [val, d_D, d_gamma]


# ===================================================================
# Temporal functions — Omori law  (translated from funcs.h)
# ===================================================================

def gfun(t, gparam):
    """
    Omori-law temporal intensity.

    Parameters
    ----------
    t : float
        Time since the parent event.
    gparam : array-like
        ``[c, p]``.

    Returns
    -------
    float
    """
    if t <= 0.0:
        return 0.0
    c, p = gparam[0], gparam[1]
    return (p - 1.0) / c * (1.0 + t / c) ** (-p)


def dgfun(t, gparam):
    """
    Omori-law temporal intensity and its partial derivatives.

    Returns
    -------
    list of float
        ``[value, d_c, d_p]``
    """
    if t <= 0.0:
        return [0.0, 0.0, 0.0]
    c, p = gparam[0], gparam[1]
    base = 1.0 + t / c
    val = (p - 1.0) / c * base ** (-p)

    # d/dc
    # val = (p-1)*c^{-1}*(1+t/c)^{-p}
    # d/dc = (p-1)*[ -c^{-2}*base^{-p} + c^{-1}*(-p)*base^{-p-1}*(-t/c^2) ]
    #       = (p-1)*base^{-p}/c^2 * [-1 + p*t/(c*base)]
    d_c = (p - 1.0) * base ** (-p) / (c * c) * (-1.0 + p * t / (c * base))

    # d/dp
    # d/dp = 1/c * base^{-p} + (p-1)/c * base^{-p} * (-ln(base))
    #      = base^{-p}/c * [1 - (p-1)*ln(base)]
    d_p = base ** (-p) / c * (1.0 - (p - 1.0) * get_xp().log(base))

    return [val, d_c, d_p]


def gfunint(t, gparam):
    """
    Integral of Omori-law intensity from 0 to *t*.

    Returns
    -------
    float
    """
    if t <= 0.0:
        return 0.0
    c, p = gparam[0], gparam[1]
    return 1.0 - (1.0 + t / c) ** (1.0 - p)


def dgfunint(t, gparam):
    """
    Integral of Omori-law intensity and its partial derivatives.

    Returns
    -------
    list of float
        ``[value, d_c, d_p]``
    """
    if t <= 0.0:
        return [0.0, 0.0, 0.0]
    c, p = gparam[0], gparam[1]
    base = 1.0 + t / c
    val = 1.0 - base ** (1.0 - p)

    # d/dc base^{1-p} = (1-p)*base^{-p}*(-t/c^2)
    d_c = (1.0 - p) * base ** (-p) * t / (c * c)      # note: -(-...) = +

    # d/dp base^{1-p} = -base^{1-p}*ln(base)
    d_p = base ** (1.0 - p) * get_xp().log(base)           # note: -(-...) = +

    return [val, d_c, d_p]


# ===================================================================
# Productivity functions
# ===================================================================

def kappafun(m, kparam):
    """
    Productivity (expected number of offspring).

    Parameters
    ----------
    m : float
        Magnitude.
    kparam : array-like
        ``[A, alpha]``.

    Returns
    -------
    float
    """
    A, alpha = kparam[0], kparam[1]
    return A * get_xp().exp(alpha * m)


def dkappafun(m, kparam):
    """
    Productivity and its partial derivatives.

    Returns
    -------
    list of float
        ``[value, d_A, d_alpha]``
    """
    A, alpha = kparam[0], kparam[1]
    val = A * get_xp().exp(alpha * m)
    d_A = get_xp().exp(alpha * m)
    d_alpha = A * m * get_xp().exp(alpha * m)
    return [val, d_A, d_alpha]
