"""
renorm.py — Analytic renormalization constants for the truncated ETAS kernels.

The truncated (pruned) ETAS model restricts the temporal Omori-Utsu kernel to
``[0, T_max]`` and the spatial power-law kernel to a disk of radius ``R_max(m_j)``.
For the truncated kernels to remain valid probability densities, they must be
renormalized by the integrals over the surviving support:

    g_trunc(dt) = g(dt) / G_norm,        dt in [0, T_max]
    f_trunc(dr | m_j) = f(dr | m_j) / F_norm(m_j),   dr in [0, R_max(m_j)]

In the limit ``eps -> 0`` the cutoffs diverge and the normalizers tend to 1,
so the truncated model collapses exactly onto the standard (un-pruned) ETAS
model — this file therefore describes a strict generalization.

All formulas are closed-form (no numerical integration), so the constants are
cheap to evaluate once per likelihood call.  ``G_norm`` is a scalar;
``F_norm`` is a vector indexed by parent event ``j`` (depends on ``m_j``).

Reference: see MODEL.md, §2 (Renormalization Derivation) and §3 (Gradients).
"""

import numpy as np


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def _safe_pow(base, expo):
    """``base**expo`` with a non-negative floor on ``base`` for stability.

    ``base`` is always ``1 + (something >= 0)`` for our kernels, but the
    argument can round slightly below 1 in float64 when the offset is ~0,
    producing complex garbage for fractional exponents.  Clamp at the eps
    level rather than zero so ``log`` / fractional powers stay well-defined.
    """
    base = np.asarray(base, dtype=np.float64)
    return np.where(base > 1.0, base, 1.0) ** expo if base.ndim else (max(base, 1.0)) ** expo


# ---------------------------------------------------------------------------
# Temporal Omori-Utsu kernel
# ---------------------------------------------------------------------------
#   g(dt) = (p - 1) / c * (1 + dt/c)^(-p),   dt >= 0
#
# Solve g(T_max) = eps_t  for  T_max:
#   (p-1)/c * (1 + T_max/c)^(-p) = eps_t
#   1 + T_max/c = ((p-1)/(c*eps_t))^(1/p)
#   T_max = c * [ ((p-1)/(c*eps_t))^(1/p) - 1 ]
#
# Integral over [0, T_max]:
#   G_norm = integral_0^T_max g(dt) dt = 1 - (1 + T_max/c)^(1 - p)
# ---------------------------------------------------------------------------

def temporal_cutoff(c, p, eps_t):
    """Cutoff time ``T_max`` where ``g(T_max) = eps_t``.

    Parameters
    ----------
    c, p : float
        Omori-Utsu parameters (``c > 0``, ``p > 1``).
    eps_t : float
        Kernel-value threshold.

    Returns
    -------
    float
        ``T_max``.  ``+inf`` when ``eps_t`` is non-positive or ``None``
        (no temporal truncation).
    """
    if eps_t is None or not np.isfinite(eps_t) or eps_t <= 0.0:
        return np.inf
    if p <= 1.0:
        # Invalid kernel; return +inf so renormalization is a no-op downstream
        # (the optimizer will see a non-normalizable model and back off).
        return np.inf
    inner = (p - 1.0) / (c * eps_t)
    return float(c * (inner ** (1.0 / p) - 1.0))


def temporal_norm(c, p, eps_t=None, T_max=None):
    """Renormalization constant ``G_norm`` for the temporal kernel.

    Either ``eps_t`` (from which ``T_max`` is derived) or ``T_max`` directly
    may be supplied.  Returns 1.0 when there is no truncation.

    The closed form is ``G_norm = 1 - (1 + T_max/c)^(1 - p)``.
    """
    if T_max is None:
        T_max = temporal_cutoff(c, p, eps_t)
    if not np.isfinite(T_max):
        return 1.0
    base = 1.0 + T_max / c
    return float(1.0 - base ** (1.0 - p))


def temporal_norm_grad(c, p, eps_t=None, T_max=None):
    """Total derivatives of ``G_norm`` w.r.t. ``c`` and ``p``.

    Because ``T_max`` is itself defined by the constraint ``g(T_max) = eps_t``
    (and therefore depends on ``c`` and ``p``), the *total* derivative seen by
    the likelihood — which recomputes ``T_max`` after each parameter perturbation
    — is *not* the partial of ``1 - (1 + T_max/c)^(1-p)`` holding ``T_max`` fixed.

    Rewriting the closed form as ``G_norm = 1 - u^(1-p)`` with
    ``u = ((p-1)/(c*eps_t))^(1/p)`` makes the dependence explicit and yields
    clean closed forms (see MODEL.md §3):

        ln u     = (1/p) * ln((p-1)/(c*eps_t))
        du/dc    = -u / (p*c)
        du/dp    = u * [ 1/(p*(p-1)) - ln u / p ]
        dG/dc    = -(p-1) * u^(1-p) / (p*c)
        dG/dp    = u^(1-p)*ln u - (1-p) * u^(-p) * du/dp

    Returns
    -------
    tuple
        ``(G_norm, dG/dc, dG/dp)``.  When un-truncated all three are
        ``(1.0, 0.0, 0.0)``.
    """
    if T_max is None:
        T_max = temporal_cutoff(c, p, eps_t)
    if not np.isfinite(T_max) or p <= 1.0:
        return 1.0, 0.0, 0.0
    if eps_t is None or eps_t <= 0.0:
        # Defensive: finite T_max with no eps_t only happens if caller passed
        # T_max directly without a corresponding eps_t; fall back to partials.
        base = 1.0 + T_max / c
        Gn = 1.0 - base ** (1.0 - p)
        dGn_dc = (1.0 - p) * (base ** (-p)) * T_max / (c * c)
        dGn_dp = (base ** (1.0 - p)) * np.log(base)
        return float(Gn), float(dGn_dc), float(dGn_dp)

    # u = ((p-1)/(c*eps_t))^(1/p)  ==  1 + T_max/c
    u = ((p - 1.0) / (c * eps_t)) ** (1.0 / p)
    log_u = np.log(u)
    Gn = 1.0 - u ** (1.0 - p)
    # Total derivatives through the cutoff constraint.
    dGn_dc = -(p - 1.0) * (u ** (1.0 - p)) / (p * c)
    du_dp = u * (1.0 / (p * (p - 1.0)) - log_u / p)
    dGn_dp = (u ** (1.0 - p)) * log_u - (1.0 - p) * (u ** (-p)) * du_dp
    return float(Gn), float(dGn_dc), float(dGn_dp)


# ---------------------------------------------------------------------------
# Spatial power-law kernel
# ---------------------------------------------------------------------------
#   f(r | m_j) = (q - 1)/(pi*sigma_j) * (1 + r^2/sigma_j)^(-q),   sigma_j = D*exp(gamma*m_j)
#
# Solve f(R_max) = eps_s  for  R_max:
#   (q-1)/(pi*sigma_j) * (1 + R_max^2/sigma_j)^(-q) = eps_s
#   1 + R_max^2/sigma_j = ((q-1)/(pi*sigma_j*eps_s))^(1/q)
#   R_max = sqrt( sigma_j * [ ((q-1)/(pi*sigma_j*eps_s))^(1/q) - 1 ] )
#
# Integral over the disk of radius R_max (area element r dr dtheta):
#   F_norm(m_j) = 1 - (1 + R_max^2/sigma_j)^(1 - q)
# ---------------------------------------------------------------------------

def _sigma(D, gamma, m_j):
    """Spatial scale ``sigma_j = D * exp(gamma * m_j)``.  Accepts scalar or array."""
    return D * np.exp(gamma * np.asarray(m_j, dtype=np.float64))


def spatial_cutoff(D, gamma, q, eps_s, m_j):
    """Cutoff radius ``R_max(m_j)`` where ``f(R_max|m_j) = eps_s``.

    Parameters
    ----------
    D, gamma, q : float
        Spatial parameters (``D > 0``, ``q > 1``).
    eps_s : float
        Kernel-value threshold.
    m_j : scalar or array
        Parent magnitudes (offset by ``m_0``).

    Returns
    -------
    np.ndarray (or float)
        ``R_max`` per event.  ``+inf`` where un-truncated.
    """
    m_j = np.asarray(m_j, dtype=np.float64)
    scalar_in = (m_j.ndim == 0)
    m_j = np.atleast_1d(m_j)

    if eps_s is None or not np.isfinite(eps_s) or eps_s <= 0.0:
        out = np.full(m_j.shape, np.inf)
        return float(out[0]) if scalar_in else out
    if q <= 1.0:
        # Invalid kernel; return +inf (no truncation), letting the optimizer
        # back off from the invalid parameter region.
        out = np.full(m_j.shape, np.inf)
        return float(out[0]) if scalar_in else out

    sig = _sigma(D, gamma, m_j)
    inner = (q - 1.0) / (np.pi * sig * eps_s)
    R2 = sig * (inner ** (1.0 / q) - 1.0)
    R2 = np.where(R2 > 0.0, R2, 0.0)
    out = np.sqrt(R2)
    return float(out[0]) if scalar_in else out


def spatial_norm(D, gamma, q, m_j, eps_s=None, R_max=None):
    """Renormalization constant ``F_norm(m_j)`` for the spatial kernel.

    Returns a vector the same shape as ``m_j`` (or a scalar if ``m_j`` is
    scalar).  Equal to 1 where un-truncated.

    Closed form: ``F_norm(m_j) = 1 - (1 + R_max^2/sigma_j)^(1 - q)``.
    """
    m_j = np.asarray(m_j, dtype=np.float64)
    scalar_in = (m_j.ndim == 0)
    m_j = np.atleast_1d(m_j)

    if R_max is None:
        R_max = spatial_cutoff(D, gamma, q, eps_s, m_j)
    R_max = np.asarray(R_max, dtype=np.float64)

    finite = np.isfinite(R_max)
    sig = _sigma(D, gamma, m_j)
    out = np.ones(m_j.shape, dtype=np.float64)
    if np.any(finite):
        R2 = R_max[finite] ** 2
        base = 1.0 + R2 / sig[finite]
        out[finite] = 1.0 - base ** (1.0 - q)
    return float(out[0]) if scalar_in else out


def spatial_norm_grad(D, gamma, q, m_j, eps_s=None, R_max=None):
    """Total derivatives of ``F_norm(m_j)`` w.r.t. ``D``, ``q``, ``gamma``.

    As in :func:`temporal_norm_grad`, the likelihood recomputes ``R_max`` (and
    hence ``sigma_j``) after each parameter perturbation, so we need the *total*
    derivative, not the partial of ``1 - (1 + R^2/sigma)^{1-q}`` holding ``R``
    fixed.  Rewriting ``F_norm = 1 - v^(1-q)`` with
    ``v = ((q-1)/(pi*sigma*eps_s))^(1/q)`` and ``sigma = D*exp(gamma*m)``
    yields clean closed forms (see MODEL.md §3):

        ln v     = (1/q)[ ln(q-1) - ln(pi*eps_s) - ln(sigma) ]
        dv/dD    = -v / (q*D)
        dv/dgamma= -v * m / q
        dv/dq    = v * [ 1/(q*(q-1)) - ln v / q ]
        dF/dD    = -(q-1) * v^(1-q) / (q*D)
        dF/dgamma= -(q-1) * m * v^(1-q) / q
        dF/dq    = -[(1-q)*v^(-q)*dv/dq - v^(1-q)*ln v]

    Returns
    -------
    tuple
        ``(F_norm, dF/dD, dF/dq, dF/dgamma)`` each with the same shape as
        ``m_j``.  Zeros where un-truncated.
    """
    m_j = np.asarray(m_j, dtype=np.float64)
    scalar_in = (m_j.ndim == 0)
    m_j = np.atleast_1d(m_j)

    if R_max is None:
        R_max = spatial_cutoff(D, gamma, q, eps_s, m_j)
    R_max = np.asarray(R_max, dtype=np.float64)
    if q <= 1.0:
        # Invalid kernel boundary; return identity with zero gradients.
        N = len(m_j)
        if scalar_in:
            return 1.0, 0.0, 0.0, 0.0
        return (np.ones(N), np.zeros(N), np.zeros(N), np.zeros(N))

    finite = np.isfinite(R_max)

    F = np.ones(m_j.shape, dtype=np.float64)
    dF_dD = np.zeros(m_j.shape, dtype=np.float64)
    dF_dq = np.zeros(m_j.shape, dtype=np.float64)
    dF_dg = np.zeros(m_j.shape, dtype=np.float64)

    if np.any(finite) and eps_s is not None and eps_s > 0.0:
        m_f = m_j[finite]
        sig_f = _sigma(D, gamma, m_f)
        # v = ((q-1)/(pi*sig*eps))^(1/q)  ==  1 + R^2/sig
        v = ((q - 1.0) / (np.pi * sig_f * eps_s)) ** (1.0 / q)
        log_v = np.log(v)
        v_1mq = v ** (1.0 - q)          # v^(1-q)
        F_f = 1.0 - v_1mq

        # Total derivatives.
        dF_dD_f = -(q - 1.0) * v_1mq / (q * D)
        dF_dg_f = -(q - 1.0) * m_f * v_1mq / q
        dv_dq = v * (1.0 / (q * (q - 1.0)) - log_v / q)
        dF_dq_f = -((1.0 - q) * (v ** (-q)) * dv_dq - v_1mq * log_v)

        F[finite] = F_f
        dF_dD[finite] = dF_dD_f
        dF_dq[finite] = dF_dq_f
        dF_dg[finite] = dF_dg_f

    if scalar_in:
        return (float(F[0]), float(dF_dD[0]), float(dF_dq[0]), float(dF_dg[0]))
    return F, dF_dD, dF_dq, dF_dg


# ---------------------------------------------------------------------------
# Depth kernel  h(u; v)
# ---------------------------------------------------------------------------
# With u = z/Z_max, v = z'/Z_max the depth kernel is the Beta-density form
#   h(u; v) = u^{eta*v} (1-u)^{eta*(1-v)} / (Z_max * B(eta*v+1, eta*(1-v)+1))
#
# By construction h integrates to 1 over [0, Z_max], so the default H_norm is 1.
# When an independent depth threshold eps_z is requested (decoupled truncation
# along z), H_norm is the integral of h over the surviving depth window — kept
# here as a hook but returning 1 unless eps_z is supplied by the caller.
# ---------------------------------------------------------------------------

def depth_norm(Z_max, eta, eps_z=None):
    """Renormalization constant ``H_norm`` for the depth kernel.

    The depth kernel ``h(u;v)`` is already normalized over ``[0, Z_max]`` by
    construction, so ``H_norm = 1`` unless the caller requests an independent
    depth threshold ``eps_z`` (in which case a windowed renormalization would
    be applied; currently a no-op returning 1).

    Returns
    -------
    float
        ``H_norm``.
    """
    return 1.0


def depth_norm_grad(Z_max, eta, eps_z=None):
    """``H_norm`` and its derivative w.r.t. ``eta``.

    With no depth truncation ``dH/deta = 0``.  Returns ``(1.0, 0.0)``.
    """
    return 1.0, 0.0


# ---------------------------------------------------------------------------
# Combined helper for the likelihood loop
# ---------------------------------------------------------------------------

def compute_all_norms(theta_param, m, mver, eps_t=None, eps_s=None,
                      eps_z=None, Z_max=None):
    """One-shot precomputation of every renormalization constant for a fit.

    Parameters
    ----------
    theta_param : array-like
        **Natural** (un-squared) ETAS parameters ordered
        ``[mu, A, c, alpha, p, D, q, gamma]`` for ``mver=1`` (add ``eta``
        last for 3D), or ``[mu, A, c, alpha, p, D, gamma]`` for ``mver=2``
        (add ``eta`` last for 3D).
    m : array-like
        Parent magnitudes (offset by ``m_0``), shape ``(N,)``.
    mver : {1, 2}
        Model version.
    eps_t, eps_s : float, optional
        Temporal / spatial truncation thresholds.  ``None`` disables.
    eps_z : float, optional
        Depth truncation threshold (currently a no-op; reserved).
    Z_max : float, optional
        Seismogenic-layer thickness; required when ``eps_z`` is used.

    Returns
    -------
    dict with keys
        ``T_max``      — scalar, temporal cutoff
        ``R_max``      — ndarray (N,), spatial cutoff per event
        ``G_norm``     — scalar
        ``F_norm``     — ndarray (N,)
        ``H_norm``     — scalar
        ``G_grad``     — (dG/dc, dG/dp)
        ``F_grad``     — (dF/dD, dF/dq, dF/dgamma), each ndarray (N,)
        ``H_grad``     — dH/deta
        ``R2_over_sig``— precomputed ``R_max^2/sigma_j`` (handy, ndarray (N,))
    """
    tp = np.asarray(theta_param, dtype=np.float64)
    c = tp[2]
    p = tp[4]
    D = tp[5]
    if mver == 1:
        q = tp[6]
        gamma = tp[7]
    else:
        gamma = tp[6]
        q = None  # Gaussian spatial kernel; not renormalized here

    # Temporal
    T_max = temporal_cutoff(c, p, eps_t)
    Gn, dGn_dc, dGn_dp = temporal_norm_grad(c, p, eps_t=eps_t, T_max=T_max)

    # Spatial
    if mver == 1 and q is not None:
        R_max = spatial_cutoff(D, gamma, q, eps_s, m)
        Fn, dFn_dD, dFn_dq, dFn_dg = spatial_norm_grad(D, gamma, q, m,
                                                        eps_s=eps_s, R_max=R_max)
        sig = _sigma(D, gamma, m)
        R2_over_sig = np.where(np.isfinite(R_max), R_max ** 2 / sig, 0.0)
    else:
        # mver=2 (Gaussian): renormalization is handled by the Gaussian CDF
        # inside the polygon integral, not here.  Return identity constants.
        N = len(m)
        R_max = np.full(N, np.inf)
        Fn = np.ones(N)
        dFn_dD = np.zeros(N)
        dFn_dq = np.zeros(N)
        dFn_dg = np.zeros(N)
        R2_over_sig = np.zeros(N)

    # Depth
    Hn, dHn_deta = 1.0, 0.0
    if eps_z is not None:
        Hn, dHn_deta = depth_norm_grad(Z_max, tp[-1] if len(tp) > (7 if mver == 2 else 8) else 1.0,
                                       eps_z=eps_z)

    return {
        'T_max': T_max,
        'R_max': R_max,
        'G_norm': Gn,
        'F_norm': Fn,
        'H_norm': Hn,
        'G_grad': (dGn_dc, dGn_dp),
        'F_grad': (dFn_dD, dFn_dq, dFn_dg),
        'H_grad': dHn_deta,
        'R2_over_sig': R2_over_sig,
    }
