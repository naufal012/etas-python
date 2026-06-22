"""
ETAS model fitting via iterative stochastic declustering.

Equivalent of etas.R from the R ETAS package. Implements the main
EM-type iterative algorithm: alternate between declustering (E-step)
and parameter estimation (M-step) until convergence.
"""

import numpy as np
import time
from dataclasses import dataclass
from typing import Optional

from .catalog import Catalog
from ..src.decluster import decluster
from ..src.optimizer import etasfit
from ..src.geometry import nn_dist
from ..src.backend import set_engine, get_engine, get_xp as _get_xp
from ..src.renorm import (temporal_cutoff, spatial_cutoff,
                          compute_all_norms)
from ..src.neighbors import NeighborIndex


def _xp():
    return _get_xp()


@dataclass
class ETASResult:
    """Result of ETAS model fitting.

    Attributes:
        param: np.ndarray — MLE parameter estimates
        bk: np.ndarray — background seismicity rates
        pb: np.ndarray — probability of being a background event
        opt: dict — optimizer output (estimate, loglik, gradient, aic, etc.)
        catalog: Catalog — the input catalog (with updated revents)
        bwd: np.ndarray — bandwidths
        thetar: np.ndarray — parameter estimates at each iteration
        loglikfv: np.ndarray — log-likelihood at each iteration
        asd: np.ndarray — asymptotic standard deviations at each iteration
        integ0: float — integral of background rate
        ndiv: int — number of integration subdivisions
        mver: int — model version
        itr: int — number of iterations completed
        exectime: float — execution time in seconds
        par_names: list — parameter names
    """
    param: np.ndarray
    bk: np.ndarray
    pb: np.ndarray
    opt: dict
    catalog: Catalog
    bwd: np.ndarray
    thetar: np.ndarray
    loglikfv: np.ndarray
    asd: np.ndarray
    integ0: float
    ndiv: int
    mver: int
    itr: int
    exectime: float
    par_names: list
    eps_t: Optional[float] = None
    eps_s: Optional[float] = None
    eps_z: Optional[float] = None
    Z_max: Optional[float] = None
    is_3d: bool = False


def etas(catalog_obj, param0=None, bwd=None, nnp=5, bwm=0.05,
         verbose=True, ndiv=1000, no_itr=11, rel_tol=1e-3,
         eps=1e-6, mver=1, epsilon=None, tau_cut=None, r_cut=None,
         engine='cpu', is_3d=False, Z_max=None,
         eps_t=None, eps_s=None, eps_z=None):
    """Fit the ETAS model using iterative stochastic declustering.

    Parameters
    ----------
    catalog_obj : Catalog
        An earthquake catalog object from catalog().
    param0 : np.ndarray, optional
        Initial parameter values [mu, A, c, alpha, p, D, q, gamma].
        If None, non-informative defaults are used.
    bwd : np.ndarray, optional
        Bandwidths for kernel smoothing. If None, computed from k-NN distances.
    nnp : int
        Number of nearest neighbors for bandwidth estimation.
    bwm : float
        Minimum bandwidth value.
    verbose : bool
        Print progress information.
    ndiv : int
        Number of subdivisions for polygon integration.
    no_itr : int
        Maximum number of EM iterations.
    rel_tol : float
        Relative tolerance for convergence.
    eps : float
        Tolerance for the DFP optimizer.
    mver : int
        Model version (1 = power-law spatial kernel, 2 = Gaussian spatial kernel).
    epsilon : float, optional
        Legacy convenience threshold.  If provided and ``eps_t``/``eps_s`` are
        not, sets both to ``epsilon`` (identical temporal/spatial thresholds).
    tau_cut : float, optional
        Hard temporal cutoff (overrides any threshold-derived cutoff).
    r_cut : float, optional
        Hard spatial cutoff (overrides any threshold-derived cutoff).
    engine : str, optional
        Computation backend to use: 'cpu' (default, NumPy) or 'gpu' (CuPy).
    is_3d : bool, optional
        If True, use the 3D-hypocentral ETAS model.
    Z_max : float, optional
        Thickness of the seismogenic layer. If None, inferred from max depth.
    eps_t : float, optional
        Temporal truncation threshold.  When set, the Omori kernel is
        truncated at ``T_max`` (analytic) and renormalized by ``G_norm``.
    eps_s : float, optional
        Spatial truncation threshold.  When set, the power-law kernel is
        truncated at ``R_max(m_j)`` (analytic, magnitude-dependent) and
        renormalized by ``F_norm(m_j)``.
    eps_z : float, optional
        Depth truncation threshold (reserved; currently H_norm = 1).

    Returns
    -------
    ETASResult
        Fitted ETAS model result.

    Notes
    -----
    With ``eps_t = eps_s = None`` (and no hard cutoffs) the model reduces
    exactly to the un-truncated ETAS model: every renormalization constant
    equals 1 and KDTree pruning is skipped.
    """
    # Legacy: epsilon sets both thresholds when the finer-grained ones are absent.
    if epsilon is not None:
        if eps_t is None:
            eps_t = epsilon
        if eps_s is None:
            eps_s = epsilon
    set_engine(engine)

    xp = _xp()
    
    ptm = time.time()

    revents = catalog_obj.revents.copy()
    rpoly = catalog_obj.rpoly
    rtperiod = catalog_obj.rtperiod
    m0 = catalog_obj.mag_threshold
    region_win = catalog_obj.region_win

    # Initial parameter values
    par_names = ['mu', 'A', 'c', 'alpha', 'p', 'D', 'q', 'gamma']
    if is_3d:
        par_names.append('eta')
        if Z_max is None:
            Z_max = revents[:, 8].max()
            if Z_max == 0:
                Z_max = 1.0  # fallback if all depths are 0
        # We append Z_max to the end of rtperiod or pass it separately.
        # It's cleaner to pass it through a global or a new array.
        # Actually, let's append Z_max to rtperiod so optimizer gets it!
        rtperiod = np.array([rtperiod[0], rtperiod[1], Z_max])

    if param0 is None:
        # Compute region area
        if hasattr(region_win, 'area'):
            win_area = region_win.area
        else:
            win_area = 1.0
        T = rtperiod[1] - rtperiod[0]
        N = revents.shape[0]
        mu0 = N / (4.0 * T * win_area)
        param0 = np.array([mu0, 0.01, 0.01, 1.0, 1.3, 0.01, 2.0, 1.0])

        if catalog_obj.dist_unit == 'km':
            param0[5] = 111.0 ** 2 * param0[5]  # D adjustment
            
        if is_3d:
            param0 = np.append(param0, 1.5)  # initial eta = 1.5

        if verbose:
            print("using non-informative initial parameter values:")
            for i, name in enumerate(par_names):
                print(f"  {name} = {param0[i]:.6f}")
        import warnings
        warnings.warn(
            "The algorithm is very sensitive to the choice of starting point")

    # Bandwidths
    if bwd is None:
        if catalog_obj.dist_unit == 'km':
            bwm = 6371.3 * np.pi / 180.0 * bwm
        rbwd = nn_dist(revents[:, 1], revents[:, 2], k=nnp)
        rbwd = np.maximum(rbwd, bwm)
    else:
        rbwd = np.asarray(bwd, dtype=np.float64)

    # Check initial parameter values
    if mver not in (1, 2):
        raise ValueError("mver must be 1 or 2")

    param1 = np.array(param0, dtype=np.float64)

    if mver == 2:
        # Remove q parameter (index 6) for mver=2
        param1 = np.delete(param1, 6)
        if is_3d:
            par_names = ['mu', 'A', 'c', 'alpha', 'p', 'D', 'gamma', 'eta']
        else:
            par_names = ['mu', 'A', 'c', 'alpha', 'p', 'D', 'gamma']

    dimparam = len(param1)

    if not np.all(param1 > 0):
        raise ValueError(
            f"param0 must have all positive components, "
            f"got: {param1}")

    # Storage for iteration history
    thetar = np.full((no_itr, dimparam), np.nan)
    asd = np.full((no_itr, dimparam), np.nan)
    loglikfv = np.zeros(no_itr)

    ihess = np.eye(dimparam)
    bk = np.zeros(revents.shape[0])

    itr = 0
    for itr_idx in range(no_itr):
        itr = itr_idx + 1

        # --- Compute analytic cutoffs from current parameters --------------
        # ``eps_t``/``eps_s`` yield T_max and R_max(m_j) via the closed forms
        # in renorm.py.  Explicit tau_cut / r_cut override the threshold path.
        current_tau = tau_cut if tau_cut is not None else np.inf
        current_r = r_cut if r_cut is not None else np.inf

        c_val = param1[2]
        p_val = param1[4]
        if mver == 1:
            D_val = param1[5]
            gamma_val = param1[7]
            q_val = param1[6]

        if eps_t is not None and tau_cut is None:
            current_tau = temporal_cutoff(c_val, p_val, eps_t)
        if eps_s is not None and r_cut is None and mver == 1:
            # Spatial cutoff is magnitude-dependent; for the KDTree query we
            # use the cutoff at the threshold magnitude m0 (a conservative
            # outer bound that contains every event-specific R_max(m_j)).
            current_r = float(spatial_cutoff(D_val, gamma_val, q_val, eps_s, m0))

        # Build the shared NeighborIndex once for this iteration.
        if (np.isfinite(current_tau) or np.isfinite(current_r)):
            nbr_index = NeighborIndex(revents[:, 1], revents[:, 2],
                                      revents[:, 0])
            nbr_index.set_cutoffs(current_tau, current_r)
        else:
            nbr_index = None

        # Renormalization constants from current parameters.
        Z_max_iter = rtperiod[2] if (is_3d and len(rtperiod) > 2) else None
        norms_iter = compute_all_norms(param1, revents[:, 3], mver,
                                       eps_t=eps_t, eps_s=eps_s,
                                       eps_z=eps_z, Z_max=Z_max_iter)

        # Declustering step
        if verbose:
            print("declustering:")

        for l in range(no_itr + 1 - itr):
            bkg = decluster(param1, rbwd, revents, rpoly, rtperiod,
                            ndiv, mver, current_tau, current_r, is_3d=is_3d,
                            norms=norms_iter, nbr_index=nbr_index)
            revents = bkg['revents']
            if verbose:
                pct = (l + 1) / (no_itr + 1 - itr) * 100
                print(f"\r  progress: {pct:.0f}%", end='', flush=True)

        if verbose:
            print()

        integ0 = bkg['integ0']
        dbk = bk - revents[:, 6]
        bk = revents[:, 6].copy()
        pb = revents[:, 6].copy()  # prob column

        if verbose:
            print(f"iteration: {itr}")
            print("=" * 54)
            print(f"background seismicity rate:")
            print(f"  min={revents[:, 5].min():.6f}  "
                  f"median={np.median(revents[:, 5]):.6f}  "
                  f"max={revents[:, 5].max():.6f}")
            print(f"probability of being a background event:")
            print(f"  min={pb.min():.6f}  "
                  f"median={np.median(pb):.6f}  "
                  f"max={pb.max():.6f}")
            print(f"integral of background seismicity rate: {integ0:.6f}")
            print("=" * 54)

        # Parameter estimation step
        if verbose:
            print("estimating:")

        opt = etasfit(param1, revents, rpoly, rtperiod, integ0, ihess,
                      verbose, ndiv, eps, mver, current_tau, current_r,
                      is_3d=is_3d, eps_t=eps_t, eps_s=eps_s, eps_z=eps_z)

        thetar[itr_idx, :] = opt['estimate']
        loglikfv[itr_idx] = opt['loglik']
        asd[itr_idx, :] = np.sqrt(np.diag(opt['avcov']))
        ihess = opt['ihessian']
        param1 = thetar[itr_idx, :].copy()

        if verbose:
            print("=" * 54)
            print("MLE:")
            for i, name in enumerate(par_names):
                print(f"  {name} = {param1[i]:.8f}")
            print("=" * 54)

        # Convergence check
        if itr > 1:
            dtht = np.max(
                (thetar[itr_idx, :] - thetar[itr_idx - 1, :]) /
                thetar[itr_idx - 1, :])
            dlrv = abs(loglikfv[itr_idx] / loglikfv[itr_idx - 1] - 1.0)
            dbkv = np.max(np.abs(dbk / (bk + 1e-30)))
            if verbose:
                print(f"convergence: dtht={dtht:.6f}  "
                      f"dlrv={dlrv:.6f}  dbkv={dbkv:.6f}")
            if dtht < rel_tol and dlrv < rel_tol and dbkv < rel_tol:
                break
        else:
            if itr == no_itr:
                import warnings
                warnings.warn("Reached maximum number of iterations")

    exectime = time.time() - ptm

    if verbose:
        print(f"Execution time: {exectime:.1f} seconds "
              f"({exectime/60:.2f} minutes)")

    # Update catalog
    catalog_obj.revents = revents

    return ETASResult(
        param=param1,
        bk=revents[:, 5].copy(),
        pb=revents[:, 6].copy(),
        opt=opt,
        catalog=catalog_obj,
        bwd=rbwd,
        thetar=thetar,
        loglikfv=loglikfv,
        asd=asd,
        integ0=integ0,
        ndiv=ndiv,
        mver=mver,
        itr=itr,
        exectime=exectime,
        par_names=par_names,
        eps_t=eps_t,
        eps_s=eps_s,
        eps_z=eps_z,
        Z_max=Z_max,
        is_3d=is_3d,
    )


def print_etas(result):
    """Print summary of a fitted ETAS model."""
    print("ETAS model: fitted using iterative stochastic declustering method")
    print(f"converged after {result.itr} iterations: "
          f"elapsed execution time {result.exectime/60:.2f} minutes\n")

    # Beta estimate from magnitudes
    revents = result.catalog.revents
    mm = revents[revents[:, 4] == 1, 3]
    if len(mm) > 0:
        bt = 1.0 / np.mean(mm)
        asd_bt = bt ** 2 / len(mm)
    else:
        bt = np.nan
        asd_bt = np.nan

    print("ML estimates of model parameters:")
    names = ['beta'] + result.par_names
    ests = np.concatenate([[bt], result.param])
    stds = np.concatenate([[asd_bt], result.asd[result.itr - 1, :]])

    print(f"{'':>12s} " +
          " ".join(f"{n:>10s}" for n in names))
    print(f"{'Estimate':>12s} " +
          " ".join(f"{e:>10.4f}" for e in ests))
    print(f"{'StdErr':>12s} " +
          " ".join(f"{s:>10.4f}" for s in stds))

    print(f"\nDeclustering probabilities:")
    print(f"  min={result.pb.min():.4f}  "
          f"1st Qu.={np.percentile(result.pb, 25):.4f}  "
          f"median={np.median(result.pb):.4f}  "
          f"mean={np.mean(result.pb):.4f}  "
          f"3rd Qu.={np.percentile(result.pb, 75):.4f}  "
          f"max={result.pb.max():.4f}")

    print(f"\nlog-likelihood: {result.opt['loglik']:.4f}\t"
          f"AIC: {result.opt['aic']:.4f}")
