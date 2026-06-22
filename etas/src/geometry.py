"""
geometry.py – Geometric and spatial utilities for the ETAS model.

Pure-Python translation of the R *ETAS* package's spatial helpers,
using numpy, scipy, and shapely.
"""

import numpy as np
from shapely.geometry import Polygon, Point


def _make_kdtree(coords):
    """Build a KDTree on *coords* using the active backend (CPU or GPU).

    Shared factory used by both this module and ``neighbors.py`` so that
    the tree backend is always consistent with ``set_engine()``.
    """
    from .backend import get_engine

    engine = get_engine()
    if engine == 'gpu':
        import cupy as cp
        from cupyx.scipy.spatial import KDTree
        if not isinstance(coords, cp.ndarray):
            coords = cp.asarray(coords)
        return KDTree(coords)
    else:
        from scipy.spatial import KDTree as _KDTree
        return _KDTree(np.asarray(coords, dtype=np.float64))


def _get_xp():
    """Get the active array module via the backend dispatcher."""
    from .backend import get_xp
    return get_xp()


# ---------------------------------------------------------------------------
# 1. decimal_places
# ---------------------------------------------------------------------------

def decimal_places(x):
    """Return the number of decimal places for each element of *x*.

    Equivalent of R's ``decimalplaces``.

    Parameters
    ----------
    x : array_like
        Numeric values (will be cast to a 1-D numpy array).

    Returns
    -------
    np.ndarray of int
        Number of decimal digits for every element.
    """
    xp = _get_xp()
    x = xp.asarray(x, dtype=float).ravel()
    result = np.zeros(len(x), dtype=int)
    for i, val in enumerate(x):
        s = f"{float(val):.15g}"          # full-precision string, no trailing zeros
        if '.' in s:
            result[i] = len(s.split('.')[1])
        else:
            result[i] = 0
    return result


# ---------------------------------------------------------------------------
# 2. roundoff_err
# ---------------------------------------------------------------------------

def roundoff_err(x):
    """Add tiny uniform noise scaled by the precision of each coordinate.

    ``x + U(-0.5, 0.5) * 10^(-decimal_places(x))``

    Equivalent of R's ``roundoffErr``.

    Parameters
    ----------
    x : array_like
        Coordinate values.

    Returns
    -------
    np.ndarray
        Jittered coordinates.
    """
    xp = _get_xp()
    x = xp.asarray(x, dtype=float).ravel()
    dp = decimal_places(x)
    noise = np.random.uniform(-0.5, 0.5, size=len(x)) * (10.0 ** (-dp))
    return xp.asarray(x) + xp.asarray(noise)


# ---------------------------------------------------------------------------
# 3. date2day
# ---------------------------------------------------------------------------

def date2day(dates, start=None):
    """Convert datetime objects to decimal days since *start*.

    Equivalent of R's ``date2day``.

    Parameters
    ----------
    dates : list of datetime.datetime or pandas.Timestamp
        The dates to convert.
    start : datetime.datetime or pandas.Timestamp, optional
        Reference origin.  If *None*, ``min(dates)`` is used.

    Returns
    -------
    np.ndarray of float64
        Fractional days elapsed since *start*.
    """
    import pandas as pd

    dates = pd.to_datetime(dates)
    if start is None:
        start = dates.min()
    else:
        start = pd.to_datetime(start)

    deltas = dates - start
    # Total seconds → days
    return np.array([td.total_seconds() / 86400.0 for td in deltas],
                    dtype=np.float64)


# ---------------------------------------------------------------------------
# 4. longlat2xy
# ---------------------------------------------------------------------------

def longlat2xy(long, lat, region_poly, dist_unit='degree'):
    """Equirectangular projection from longitude/latitude to flat (x, y).

    Parameters
    ----------
    long, lat : array_like
        Geographic coordinates (degrees).
    region_poly : shapely.geometry.Polygon
        Study-region boundary in geographic coordinates.
    dist_unit : {'degree', 'km'}
        Output distance unit.

    Returns
    -------
    dict
        ``'x'``  – projected x coordinates (np.ndarray)
        ``'y'``  – projected y coordinates (np.ndarray)
        ``'region_win'`` – projected boundary (shapely Polygon)
    """
    xp = _get_xp()
    long = xp.asarray(long, dtype=float)
    lat = xp.asarray(lat, dtype=float)
    
    # Handle both dict and shapely Polygon for region_poly
    if isinstance(region_poly, dict):
        poly_long = xp.asarray(region_poly['long'])
        poly_lat = xp.asarray(region_poly['lat'])
        from shapely.geometry import Polygon
        poly_obj = Polygon(np.column_stack([np.asarray(poly_long), np.asarray(poly_lat)]))
    else:
        poly_obj = region_poly
        poly_coords = np.array(poly_obj.exterior.coords)
        poly_long = poly_coords[:, 0]
        poly_lat = poly_coords[:, 1]

    if dist_unit == 'degree':
        centroid = poly_obj.centroid
        cx, cy = centroid.x, centroid.y
        cos_cy = xp.cos(cy * xp.pi / 180.0)

        x = cos_cy * (long - cx)
        y = lat - cy

        px = cos_cy * (poly_long - cx)
        py = poly_lat - cy

    elif dist_unit == 'km':
        x = 111.320 * xp.cos(lat / 180.0 * xp.pi) * long
        y = 110.574 * lat

        px = 111.320 * xp.cos(poly_lat / 180.0 * xp.pi) * poly_long
        py = 110.574 * poly_lat

    else:
        raise ValueError(f"dist_unit must be 'degree' or 'km', got '{dist_unit}'")

    region_win = Polygon(np.column_stack([np.asarray(px), np.asarray(py)]))

    return {'x': x, 'y': y, 'region_win': region_win}


# ---------------------------------------------------------------------------
# 5. xy2longlat
# ---------------------------------------------------------------------------

def xy2longlat(x, y, region_poly, dist_unit='degree'):
    """Inverse of :func:`longlat2xy`.

    Parameters
    ----------
    x, y : array_like
        Projected coordinates.
    region_poly : shapely.geometry.Polygon
        Study-region boundary in *geographic* coordinates.
    dist_unit : {'degree', 'km'}
        Unit that was used in the forward projection.

    Returns
    -------
    dict
        ``'long'`` – longitude (np.ndarray)
        ``'lat'``  – latitude  (np.ndarray)
    """
    xp = _get_xp()
    x = xp.asarray(x, dtype=float)
    y = xp.asarray(y, dtype=float)

    if isinstance(region_poly, dict):
        poly_long = xp.asarray(region_poly['long'])
        poly_lat = xp.asarray(region_poly['lat'])
        from shapely.geometry import Polygon
        poly_obj = Polygon(np.column_stack([np.asarray(poly_long), np.asarray(poly_lat)]))
    else:
        poly_obj = region_poly

    if dist_unit == 'degree':
        centroid = poly_obj.centroid
        cx, cy = centroid.x, centroid.y
        cos_cy = xp.cos(cy * xp.pi / 180.0)

        lat = y + cy
        long = x / cos_cy + cx

    elif dist_unit == 'km':
        lat = y / 110.574
        long = x / (111.320 * xp.cos(lat / 180.0 * xp.pi))

    else:
        raise ValueError(f"dist_unit must be 'degree' or 'km', got '{dist_unit}'")

    return {'long': long, 'lat': lat}


# ---------------------------------------------------------------------------
# 6. polygon_area  (shoelace formula)
# ---------------------------------------------------------------------------

def polygon_area(px, py):
    """Compute the area of a simple polygon via the shoelace formula.

    Parameters
    ----------
    px, py : array_like
        Vertex coordinates (need *not* be closed; the function closes them).

    Returns
    -------
    float
        Unsigned area.
    """
    px = np.asarray(px, dtype=float)
    py = np.asarray(py, dtype=float)
    n = len(px)
    # Shoelace
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += px[i] * py[j]
        area -= px[j] * py[i]
    return abs(area) / 2.0


# ---------------------------------------------------------------------------
# 7. polygon_centroid
# ---------------------------------------------------------------------------

def polygon_centroid(px, py):
    """Return the centroid (cx, cy) of a simple polygon.

    Parameters
    ----------
    px, py : array_like
        Vertex coordinates (need *not* be closed).

    Returns
    -------
    tuple of float
        ``(cx, cy)``
    """
    xp = _get_xp()
    px = xp.asarray(px, dtype=float)
    py = xp.asarray(py, dtype=float)
    n = len(px)
    A = 0.0
    cx = 0.0
    cy = 0.0
    for i in range(n):
        j = (i + 1) % n
        cross = float(px[i]) * float(py[j]) - float(px[j]) * float(py[i])
        A += cross
        cx += (float(px[i]) + float(px[j])) * cross
        cy += (float(py[i]) + float(py[j])) * cross
    A *= 0.5
    if A == 0:
        return (float(xp.mean(px)), float(xp.mean(py)))
    cx /= (6.0 * A)
    cy /= (6.0 * A)
    return (cx, cy)


# ---------------------------------------------------------------------------
# 8. inside_polygon
# ---------------------------------------------------------------------------

def inside_polygon(x, y, polygon):
    """Point-in-polygon test using shapely.

    Parameters
    ----------
    x, y : array_like
        Query point coordinates.
    polygon : shapely.geometry.Polygon
        The polygon to test against.

    Returns
    -------
    np.ndarray of bool
        True where (x[i], y[i]) is inside the polygon.
    """
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    from shapely import vectorized
    try:
        # shapely ≥ 2.0 vectorised contains
        from shapely import contains_xy
        return contains_xy(polygon, x, y)
    except ImportError:
        # Fallback for older shapely
        return np.array([polygon.contains(Point(xi, yi))
                         for xi, yi in zip(x, y)], dtype=bool)


# ---------------------------------------------------------------------------
# 9. nn_dist
# ---------------------------------------------------------------------------

def nn_dist(x, y, k=5):
    """k-th nearest-neighbour distance for each point.

    Parameters
    ----------
    x, y : array_like
        Point coordinates.
    k : int, optional
        Neighbour rank (default 5).

    Returns
    -------
    np.ndarray of float, shape (N,)
        Distance to the k-th nearest neighbour for every point.
    """
    xp = _get_xp()
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    coords = np.column_stack([x, y])

    # Clamp k to n-1 (can't have more neighbours than points minus self)
    n = len(coords)
    k_eff = min(k, n - 1)
    if k_eff < 1:
        return np.zeros(n)

    tree = _make_kdtree(coords)
    # On GPU the query input must also be a CuPy array.
    from .backend import get_engine
    if get_engine() == 'gpu':
        import cupy as cp
        query_coords = cp.asarray(coords)
    else:
        query_coords = coords
    # query k_eff+1 because the nearest neighbour of a point is itself
    dists, _ = tree.query(query_coords, k=k_eff + 1)
    # dists[:, 0] ≈ 0 (self), dists[:, k_eff] is the k-th neighbour
    return np.asarray(dists)[:, k_eff]
