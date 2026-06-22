"""
Earthquake catalog construction and preprocessing.

Equivalent of catalog.R from the R ETAS package.
Handles data ingestion, validation, temporal/spatial filtering,
flat-map projection, and construction of the revents array.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple, Union

from ..src.geometry import (
    longlat2xy, xy2longlat, roundoff_err, date2day,
    inside_polygon, polygon_area, polygon_centroid, nn_dist
)
from ..src.backend import get_xp as _get_xp


def _xp():
    return _get_xp()

try:
    from shapely.geometry import Polygon
except ImportError:
    Polygon = None


@dataclass
class Catalog:
    """Earthquake catalog object, equivalent to R's 'catalog' class.

    Attributes:
        revents: np.ndarray of shape (N, 8) — columns:
            [tt, xx, yy, mm, flag, bkgd, prob, lambd]
        rpoly: np.ndarray of shape (np+1, 2) — closed polygon [px, py]
        rtperiod: np.ndarray of shape (2,) — [study_start_day, study_end_day]
        region_poly: dict with keys 'long', 'lat'
        region_win: shapely Polygon (projected coordinates)
        time_begin: datetime
        study_start: datetime
        study_end: datetime
        study_length: float or None (in decimal days)
        mag_threshold: float
        longlat_coord: pd.DataFrame with columns [long, lat, flag, dt]
        dist_unit: str ('degree' or 'km')
    """
    revents: np.ndarray
    rpoly: np.ndarray
    rtperiod: np.ndarray
    region_poly: Dict
    region_win: object  # shapely Polygon
    time_begin: datetime
    study_start: datetime
    study_end: datetime
    study_length: Optional[float]
    mag_threshold: float
    longlat_coord: pd.DataFrame
    dist_unit: str


def catalog(data, time_begin=None, study_start=None,
            study_end=None, study_length=None,
            lat_range=None, long_range=None,
            region_poly=None, mag_threshold=None,
            flatmap=True, dist_unit="degree",
            roundoff=True, tz="UTC"):
    """Construct an earthquake catalog object.

    Parameters
    ----------
    data : pd.DataFrame
        Must contain columns: 'date', 'time', 'long', 'lat', 'mag'.
    time_begin : str or datetime, optional
        Start of the temporal window. If None, uses min(dates).
    study_start : str or datetime, optional
        Start of the study period. If None, equals time_begin.
    study_end : str or datetime, optional
        End of the study period. If None, uses max(dates).
    study_length : float, optional
        Length of the study period in decimal days.
    lat_range : tuple of (lat_min, lat_max), optional
    long_range : tuple of (long_min, long_max), optional
    region_poly : dict with keys 'long' and 'lat', optional
        Polygon boundary vertices.
    mag_threshold : float, optional
        Minimum magnitude. If None, uses min(mag).
    flatmap : bool
        If True, project to flat-map coordinates.
    dist_unit : str
        'degree' or 'km'.
    roundoff : bool
        If True, add small uniform noise to coordinates.
    tz : str
        Timezone string, default 'UTC'.

    Returns
    -------
    Catalog
        The constructed catalog object.
    """
    # Validate input
    data = data.copy()
    data.columns = [c.lower() for c in data.columns]
    required = ['date', 'time', 'long', 'lat', 'mag']
    for v in required:
        if v not in data.columns:
            raise ValueError(
                f"data must contain column '{v}'. "
                f"Required columns: {required}")
        if data[v].isna().any():
            raise ValueError(f"Column '{v}' must not contain NA values")

    if not np.issubdtype(data['lat'].dtype, np.number):
        data['lat'] = pd.to_numeric(data['lat'])
    if not np.issubdtype(data['long'].dtype, np.number):
        data['long'] = pd.to_numeric(data['long'])
    if not np.issubdtype(data['mag'].dtype, np.number):
        data['mag'] = pd.to_numeric(data['mag'])

    # Extract coordinates, depth and magnitude
    xx = data['long'].values.astype(np.float64).copy()
    yy = data['lat'].values.astype(np.float64).copy()

    # Accept either 'depth' or 'z' as the hypocentral-depth column.
    if 'depth' in data.columns:
        depth_col = 'depth'
    elif 'z' in data.columns:
        depth_col = 'z'
    else:
        depth_col = None

    if depth_col is not None:
        if not np.issubdtype(data[depth_col].dtype, np.number):
            data[depth_col] = pd.to_numeric(data[depth_col])
        zz = data[depth_col].values.astype(np.float64).copy()
    else:
        zz = np.zeros(len(xx), dtype=np.float64)

    mm = data['mag'].values.astype(np.float64).copy()

    # Account for round-off error
    if roundoff:
        xx = roundoff_err(xx)
        yy = roundoff_err(yy)

    # Parse date and time
    dt_strings = data['date'].astype(str) + ' ' + data['time'].astype(str)
    try:
        dt = pd.to_datetime(dt_strings, utc=(tz == 'UTC'))
    except Exception:
        dt = pd.to_datetime(dt_strings, utc=(tz == 'UTC'), format='mixed')

    dt = dt.values  # numpy datetime64

    # Handle duplicate timestamps
    dt_series = pd.Series(dt)
    diffs = dt_series.diff()
    dup_idx = np.where(diffs == np.timedelta64(0))[0]
    if len(dup_idx) > 0:
        dt = dt.copy()
        for i in dup_idx:
            dt[i] = dt[i] + np.timedelta64(1, 's')
        import warnings
        warnings.warn(
            f"More than one event occurred simultaneously! "
            f"Check events {dup_idx.tolist()}. "
            f"Duplicated times have been altered by one second.")

    # Sort chronologically
    if not np.all(dt[:-1] <= dt[1:]):
        sort_idx = np.argsort(dt)
        dt = dt[sort_idx]
        xx = xx[sort_idx]
        yy = yy[sort_idx]
        zz = zz[sort_idx]
        mm = mm[sort_idx]
        data = data.iloc[sort_idx].reset_index(drop=True)
        import warnings
        warnings.warn(
            "Events were not chronologically sorted: "
            "they have been sorted in ascending order")

    # Time begin
    if time_begin is None:
        time_begin_dt = pd.Timestamp(dt.min())
    else:
        time_begin_dt = pd.Timestamp(time_begin)
        if np.all(dt < time_begin_dt.to_numpy()):
            raise ValueError(
                f"No event after time_begin={time_begin_dt}")

    # Study start
    if study_start is None:
        study_start_dt = time_begin_dt
    else:
        study_start_dt = pd.Timestamp(study_start)
        if study_start_dt < time_begin_dt:
            raise ValueError(
                f"study_start ({study_start_dt}) cannot be before "
                f"time_begin ({time_begin_dt})")

    # Study end
    if study_length is not None:
        if study_end is not None:
            raise ValueError(
                "Either study_end or study_length needs to be specified, "
                "not both")
        study_end_dt = study_start_dt + pd.Timedelta(days=study_length)
    elif study_end is None:
        study_end_dt = pd.Timestamp(dt.max())
    else:
        study_end_dt = pd.Timestamp(study_end)
        if study_end_dt < study_start_dt:
            raise ValueError(
                f"study_end ({study_end_dt}) cannot be before "
                f"study_start ({study_start_dt})")

    # Convert times to decimal days
    tt = date2day(dt, time_begin_dt.to_numpy())

    # Spatial region
    if lat_range is None:
        lat_rng = yy.min(), yy.max()
        margin = 0.01 * (lat_rng[1] - lat_rng[0])
        lat_range = (lat_rng[0] - margin, lat_rng[1] + margin)
    if long_range is None:
        long_rng = xx.min(), xx.max()
        margin = 0.01 * (long_rng[1] - long_rng[0])
        long_range = (long_rng[0] - margin, long_rng[1] + margin)

    if region_poly is None:
        region_poly = {
            'long': np.array([long_range[0], long_range[1],
                              long_range[1], long_range[0]]),
            'lat': np.array([lat_range[0], lat_range[0],
                             lat_range[1], lat_range[1]])
        }
        if Polygon is not None:
            region_win = Polygon([
                (long_range[0], lat_range[0]),
                (long_range[1], lat_range[0]),
                (long_range[1], lat_range[1]),
                (long_range[0], lat_range[1])
            ])
        else:
            region_win = None
    else:
        if isinstance(region_poly, pd.DataFrame):
            region_poly = {
                'long': region_poly['long'].values,
                'lat': region_poly['lat'].values
            }
        if not isinstance(region_poly, dict):
            raise ValueError(
                "region_poly must be a dict with keys 'lat' and 'long'")
        if 'lat' not in region_poly or 'long' not in region_poly:
            raise ValueError(
                "region_poly must have keys 'lat' and 'long'")
        rp_lat = np.asarray(region_poly['lat'], dtype=np.float64)
        rp_long = np.asarray(region_poly['long'], dtype=np.float64)
        if np.any(np.isnan(rp_lat)) or np.any(np.isnan(rp_long)):
            raise ValueError(
                "lat and long coordinates must not contain NA values")
        if len(rp_lat) != len(rp_long):
            raise ValueError(
                "lat and long must be numeric arrays of equal length")
        if len(rp_lat) < 3:
            raise ValueError("region_poly needs at least 3 vertices")

        region_poly = {'long': rp_long, 'lat': rp_lat}

        if Polygon is not None:
            coords = list(zip(rp_long, rp_lat))
            region_win = Polygon(coords)
            if region_win.area < 0:
                raise ValueError(
                    "Area of polygon is negative - "
                    "maybe traversed in wrong direction?")
        else:
            region_win = None

    # Magnitude threshold
    if mag_threshold is None:
        mag_threshold = mm.min()

    # Project to flat-map coordinates
    longlat_coord = pd.DataFrame({
        'long': xx.copy(), 'lat': yy.copy()
    })
    if flatmap:
        proj = longlat2xy(xx, yy, region_poly, dist_unit)
        xx = proj['x']
        yy = proj['y']
        region_win = proj['region_win']

    # Filter events
    dt_ts = pd.DatetimeIndex(dt)
    ok = ((dt_ts <= study_end_dt) &
          (dt_ts >= time_begin_dt) &
          (mm >= mag_threshold))

    xx = xx[ok]
    yy = yy[ok]
    zz = zz[ok]
    tt = tt[ok]
    mm = mm[ok] - mag_threshold
    dt_filtered = dt[ok]

    # Flag: 1 = inside study region, 0 = outside spatial,
    # -2 = outside temporal
    flag = inside_polygon(xx, yy, region_win).astype(np.int32)

    dt_filtered_ts = pd.DatetimeIndex(dt_filtered)
    flag[dt_filtered_ts < study_start_dt] = -2

    # Build revents array (N x 9):
    # [tt, xx, yy, mm, flag, bkgd, prob, lambd, zz]
    xp = _xp()
    N = len(tt)
    revents = xp.column_stack([
        xp.asarray(tt), xp.asarray(xx), xp.asarray(yy), xp.asarray(mm),
        xp.asarray(flag, dtype=np.float64),
        xp.zeros(N),   # bkgd
        xp.ones(N),    # prob
        xp.zeros(N),   # lambd
        xp.asarray(zz)             # depth
    ])

    # Update longlat_coord
    longlat_coord = longlat_coord.iloc[np.where(ok)[0]].reset_index(drop=True)
    longlat_coord['flag'] = flag
    longlat_coord['dt'] = dt_filtered

    # Build polygon boundary (closed polygon)
    if region_win is not None and Polygon is not None:
        if hasattr(region_win, 'exterior'):
            # Shapely polygon
            coords = np.array(region_win.exterior.coords)
            rpoly = coords  # already closed
        else:
            # Fallback for rectangular windows
            rpoly = np.column_stack([
                np.array([long_range[0], long_range[1],
                          long_range[1], long_range[0], long_range[0]]),
                np.array([lat_range[0], lat_range[0],
                          lat_range[1], lat_range[1], lat_range[0]])
            ])
    else:
        px = np.array([long_range[0], long_range[1],
                       long_range[1], long_range[0], long_range[0]])
        py = np.array([lat_range[0], lat_range[0],
                       lat_range[1], lat_range[1], lat_range[0]])
        rpoly = np.column_stack([px, py])

    # Study period in decimal days
    rtperiod = np.array([
        date2day(np.array([study_start_dt.to_numpy()]),
                 time_begin_dt.to_numpy())[0],
        date2day(np.array([study_end_dt.to_numpy()]),
                 time_begin_dt.to_numpy())[0]
    ])

    return Catalog(
        revents=revents,
        rpoly=rpoly,
        rtperiod=rtperiod,
        region_poly=region_poly,
        region_win=region_win,
        time_begin=time_begin_dt.to_pydatetime(),
        study_start=study_start_dt.to_pydatetime(),
        study_end=study_end_dt.to_pydatetime(),
        study_length=study_length,
        mag_threshold=mag_threshold,
        longlat_coord=longlat_coord,
        dist_unit=dist_unit
    )


def print_catalog(cat):
    """Print summary of an earthquake catalog."""
    print(f"earthquake catalog:")
    print(f"  time begin {cat.time_begin}")
    print(f"  study period: {cat.study_start} to {cat.study_end} "
          f"(T = {cat.rtperiod[1] - cat.rtperiod[0]:.1f} days)")
    print(f"threshold magnitude: {cat.mag_threshold}")
    n_total = cat.revents.shape[0]
    n_target = np.sum(cat.revents[:, 4] == 1)
    n_outside_spatial = np.sum(cat.revents[:, 4] == 0)
    n_outside_temporal = np.sum(cat.revents[:, 4] == -2)
    print(f"number of events:")
    print(f"  total events {n_total}: {n_target} target events, "
          f"{n_total - n_target} complementary events")
    print(f"  ({n_outside_spatial} events outside geographical region, "
          f"{n_outside_temporal} events outside study period)")
