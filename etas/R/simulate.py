"""
ETAS earthquake simulation.

Equivalent of simetas.R from the R ETAS package. Generates synthetic
earthquake catalogs from a fitted ETAS model using thinning/branching.

Supports both the 2D and the 3D (hypocentral) model.  For 3D the depth of
each offspring is drawn from the Beta-density depth kernel
``h(u; v)`` via inverse-CDF sampling on ``u in [0, 1]``.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from scipy.special import betaincinv, gammaln

from .catalog import catalog as make_catalog
from ..src.geometry import longlat2xy, xy2longlat


def _sample_depth_beta(z_parent, Z_max, eta, n, rng):
    """Sample ``n`` offspring depths from the depth kernel ``h(u; v)``.

    The normalized depth kernel in ``u = z/Z_max`` is a Beta density with
    shape parameters ``a = eta*v + 1`` and ``b = eta*(1-v) + 1`` where
    ``v = z_parent/Z_max``.  We sample via the inverse CDF of the Beta
    distribution (``scipy.special.betaincinv``), then rescale to ``[0, Z_max]``.

    When ``eta`` is small the distribution is nearly uniform; when ``eta`` is
    large offspring concentrate tightly around the parent depth.
    """
    v = np.clip(z_parent / Z_max, 1e-6, 1.0 - 1e-6)
    a = eta * v + 1.0
    b = eta * (1.0 - v) + 1.0
    u = betaincinv(a, b, rng.uniform(1e-9, 1.0 - 1e-9, size=n))
    return np.clip(u * Z_max, 0.0, Z_max)


def simetas(param, bkgd, sim_start, sim_end=None, sim_length=None,
            lat_range=None, long_range=None, region_poly=None,
            mag_threshold=None, flatmap=True, dist_unit="degree",
            roundoff=False, tz="UTC", is_3d=False, Z_max=None, eta=None,
            seed=None):
    """Simulate an earthquake catalog from an ETAS model.

    Parameters
    ----------
    param : dict or np.ndarray
        Model parameters. If dict, must contain keys:
        'beta', 'mu', 'A', 'c', 'alpha', 'p', 'D', 'q', 'gamma'.
        If np.ndarray, values in order
        [beta, mu, A, c, alpha, p, D, q, gamma].
    bkgd : dict
        Background rate grid with keys 'x', 'y', 'bkgd' from rates().
    sim_start : str or datetime
        Start of simulation period.
    sim_end : str or datetime, optional
        End of simulation period.
    sim_length : float, optional
        Length of simulation in decimal days.
    lat_range : tuple, optional
    long_range : tuple, optional
    region_poly : dict, optional
        Polygon with keys 'long', 'lat'.
    mag_threshold : float, optional
    flatmap : bool
    dist_unit : str
    roundoff : bool
    tz : str
    is_3d : bool, optional
        If True, also simulate hypocentral depths via the Beta depth kernel.
    Z_max : float, optional
        Seismogenic-layer thickness (required when ``is_3d``).
    eta : float, optional
        Depth-kernel concentration parameter (required when ``is_3d``); may
        also be supplied inside ``param`` as the last element.
    seed : int, optional
        RNG seed for reproducibility.

    Returns
    -------
    Catalog
        Simulated earthquake catalog.
    """
    rng = np.random.default_rng(seed)

    # Parse parameters
    if isinstance(param, dict):
        beta = param['beta']
        A = param['A']
        c = param['c']
        alpha = param['alpha']
        p = param['p']
        D = param['D']
        q = param['q']
        gamma = param['gamma']
        if is_3d and eta is None:
            eta = param.get('eta', 1.5)
    else:
        param = np.asarray(param)
        beta = param[0]
        A = param[1] if len(param) > 1 else 0.01
        c = param[2] if len(param) > 2 else 0.01
        alpha = param[3] if len(param) > 3 else 1.0
        p = param[4] if len(param) > 4 else 1.3
        D = param[5] if len(param) > 5 else 0.01
        q = param[6] if len(param) > 6 else 2.0
        gamma = param[7] if len(param) > 7 else 1.0
        if is_3d:
            eta = param[8] if (eta is None and len(param) > 8) else (eta or 1.5)

    if is_3d and Z_max is None:
        raise ValueError("is_3d=True requires Z_max (seismogenic-layer thickness).")

    # Parse times
    sim_start_dt = pd.Timestamp(sim_start)
    if sim_length is not None:
        if sim_end is not None:
            raise ValueError(
                "Either sim_end or sim_length, not both")
        sim_end_dt = sim_start_dt + pd.Timedelta(days=sim_length)
    else:
        sim_end_dt = pd.Timestamp(sim_end)
        if sim_end_dt < sim_start_dt:
            raise ValueError(
                f"sim_end ({sim_end_dt}) before sim_start ({sim_start_dt})")

    simtt = np.array([0.0,
                      (sim_end_dt - sim_start_dt).total_seconds() / 86400.0])

    # Set up region
    if region_poly is None:
        if lat_range is None or long_range is None:
            raise ValueError(
                "Must provide region_poly or both lat_range and long_range")
        region_poly = {
            'long': np.array([long_range[0], long_range[1],
                              long_range[1], long_range[0]]),
            'lat': np.array([lat_range[0], lat_range[0],
                             lat_range[1], lat_range[1]])
        }

    # Background rate grid
    bx = np.asarray(bkgd['x'])
    by = np.asarray(bkgd['y'])
    bz = np.asarray(bkgd['bkgd'])

    # Generate background events via inhomogeneous Poisson
    # Integrate rate over time
    total_rate = bz * np.diff(simtt)
    max_rate = total_rate.max()

    # Discretize into cells
    if bz.ndim == 2:
        nx, ny = bz.shape
    else:
        nx = len(bx)
        ny = len(by)
        bz = bz.reshape(nx, ny)
        total_rate = total_rate.reshape(nx, ny)

    dx = (bx[-1] - bx[0]) / (nx - 1) if nx > 1 else 1.0
    dy = (by[-1] - by[0]) / (ny - 1) if ny > 1 else 1.0
    cell_area = dx * dy

    # Expected number per cell
    expected = total_rate * cell_area
    # Generate Poisson counts per cell
    events_list = []
    for i in range(nx):
        for j in range(ny):
            n_ij = rng.poisson(max(0, expected[i, j]))
            if n_ij > 0:
                # Uniform within cell
                ex = bx[i] + rng.uniform(-dx/2, dx/2, n_ij)
                ey = by[j] + rng.uniform(-dy/2, dy/2, n_ij)
                et = rng.uniform(simtt[0], simtt[1], n_ij)
                em = rng.exponential(1.0 / beta, n_ij)
                # Background depths: uniform on [0, Z_max] for 3D, else None.
                ez = rng.uniform(0.0, Z_max, n_ij) if is_3d else None
                for k in range(n_ij):
                    row = (et[k], ex[k], ey[k], em[k])
                    events_list.append(row + ((ez[k],) if is_3d else ()))

    # Project background events if flatmap
    if events_list:
        bg_events = np.array(events_list, dtype=object)
        bg_tt = np.array([r[0] for r in events_list], dtype=float)
        bg_xx = np.array([r[1] for r in events_list], dtype=float)
        bg_yy = np.array([r[2] for r in events_list], dtype=float)
        bg_mm = np.array([r[3] for r in events_list], dtype=float)
        bg_zz = (np.array([r[4] for r in events_list], dtype=float)
                 if is_3d else None)

        if flatmap:
            proj = longlat2xy(bg_xx, bg_yy, region_poly, dist_unit)
            bg_xx_proj = proj['x']
            bg_yy_proj = proj['y']
        else:
            bg_xx_proj = bg_xx
            bg_yy_proj = bg_yy

        revents_df = pd.DataFrame({
            'tt': bg_tt, 'xx': bg_xx_proj,
            'yy': bg_yy_proj, 'mm': bg_mm
        })
        if is_3d:
            revents_df['zz'] = bg_zz
    else:
        revents_df = pd.DataFrame(columns=['tt', 'xx', 'yy', 'mm']
                                  + (['zz'] if is_3d else []))

    # Branching: generate triggered events generation by generation
    out = [revents_df]
    gen = 1

    while True:
        gen += 1
        parent = out[-1]
        if len(parent) == 0:
            break

        children = []
        for idx in range(len(parent)):
            # Number of triggered events
            nl = rng.poisson(
                A * np.exp(alpha * parent['mm'].iloc[idx]))
            if nl > 0:
                # Triggered times (Omori law inverse sampling).
                # g(dt) CDF: 1 - (1 + dt/c)^(1-p).  Inverse-CDF sample:
                u = rng.uniform(0, 1, nl)
                tl = (parent['tt'].iloc[idx] +
                      c + c / ((1.0 - u) ** (p - 1)))

                # Triggered locations (power-law spatial kernel).
                sig = D * np.exp(gamma * parent['mm'].iloc[idx])
                u2 = rng.uniform(0, 1, nl)
                rl = np.sqrt(sig + sig / ((1.0 - u2) ** (q - 1)))
                th = rng.uniform(0, 2 * np.pi, nl)
                xl = parent['xx'].iloc[idx] + rl * np.cos(th)
                yl = parent['yy'].iloc[idx] + rl * np.sin(th)

                # Magnitudes
                ml = rng.exponential(1.0 / beta, nl)

                row = {'tt': tl, 'xx': xl, 'yy': yl, 'mm': ml}
                if is_3d:
                    # Offspring depth from the Beta depth kernel h(u; v).
                    z_parent = parent['zz'].iloc[idx] if 'zz' in parent else \
                        (parent['zz'].iloc[idx] if 'zz' in parent.columns else 0.0)
                    row['zz'] = _sample_depth_beta(z_parent, Z_max, eta, nl, rng)
                children.append(pd.DataFrame(row))

        if children:
            out.append(pd.concat(children, ignore_index=True))
        else:
            break

    # Combine all generations
    all_events = pd.concat(out, ignore_index=True)

    # Sort by time
    all_events = all_events.sort_values('tt').reset_index(drop=True)

    # Convert back to lon/lat and datetime
    all_events['tt_dt'] = (
        sim_start_dt + pd.to_timedelta(all_events['tt'], unit='D'))

    if flatmap:
        proj = xy2longlat(all_events['xx'].values,
                          all_events['yy'].values,
                          region_poly, dist_unit)
        all_events['long'] = proj['long']
        all_events['lat'] = proj['lat']
    else:
        all_events['long'] = all_events['xx']
        all_events['lat'] = all_events['yy']

    if mag_threshold is None:
        mag_threshold = 0.0

    all_events['mag'] = all_events['mm'] + mag_threshold
    all_events['date'] = all_events['tt_dt'].dt.strftime('%Y-%m-%d')
    all_events['time'] = all_events['tt_dt'].dt.strftime('%H:%M:%S')

    # Build catalog
    data = pd.DataFrame({
        'date': all_events['date'],
        'time': all_events['time'],
        'long': all_events['long'],
        'lat': all_events['lat'],
        'mag': all_events['mag']
    })
    if is_3d:
        # ``catalog()`` accepts either 'depth' or 'z' as the depth column.
        data['depth'] = all_events['zz']

    simcat = make_catalog(
        data, study_start=None, study_end=str(sim_end_dt),
        lat_range=lat_range, long_range=long_range,
        region_poly=region_poly, mag_threshold=mag_threshold,
        flatmap=flatmap, dist_unit=dist_unit,
        roundoff=roundoff, tz=tz)

    return simcat
