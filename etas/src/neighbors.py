"""
neighbors.py — Temporal + spatial neighbor pruning for the truncated ETAS model.

The hot loops in the ETAS likelihood sum, for every event ``j``, over all
earlier events ``i < j``.  Naively that is O(N^2).  When temporal and spatial
cutoffs (``tau_cut``, ``r_cut``) are finite we only need the parents that
satisfy

    dt_ij = t_j - t_i <= tau_cut       (temporal: contiguous slice, O(log N))
    r_ij  = ||x_j - x_i|| <= r_cut     (spatial: KDTree ball query)

This module exposes :class:`NeighborIndex`, which builds a single
KDTree over the (x, y) catalog coordinates and provides vectorized
lookups of the parent-index intersection for each target event.

When ``tau_cut`` or ``r_cut`` is infinite the corresponding constraint is
skipped (returning the full earlier-events slice), so the un-truncated path
remains bit-for-bit identical to the original implementation.

The KDTree backend dispatches to ``scipy.spatial.cKDTree`` (CPU) or
``cupyx.scipy.spatial.KDTree`` (GPU) depending on the active engine set via
``etas.src.backend.set_engine()``.
"""

import numpy as np


def _make_kdtree(coords):
    """Build a KDTree on *coords* using the active backend (CPU or GPU).

    Parameters
    ----------
    coords : array-like, shape (N, 2)
        Point coordinates (must be a NumPy array on CPU or CuPy array on GPU).

    Returns
    -------
    tree : scipy.spatial.cKDTree or cupyx.scipy.spatial.KDTree
    """
    from .backend import get_engine

    engine = get_engine()
    if engine == 'gpu':
        import cupy as cp
        from cupyx.scipy.spatial import KDTree
        # cupyx KDTree expects CuPy arrays
        if not isinstance(coords, cp.ndarray):
            coords = cp.asarray(coords)
        return KDTree(coords)
    else:
        from scipy.spatial import cKDTree
        return cKDTree(np.asarray(coords, dtype=np.float64))


class NeighborIndex:
    """Precomputed temporal + spatial neighbor structure for an event catalog.

    Parameters
    ----------
    x, y : array-like, shape (N,)
        Spatial coordinates of all events (in projected units).
    t : array-like, shape (N,)
        Event times, **sorted ascending**.  The catalog() builder guarantees
        this; we do not re-sort here.

    Attributes
    ----------
    tree : scipy.spatial.cKDTree or cupyx.scipy.spatial.KDTree
        KDTree over the 2D (x, y) coordinates, used for spatial ball queries.
    t : np.ndarray
        Sorted event times (reference).
    n : int
        Number of events.
    """

    def __init__(self, x, y, t):
        self.x = np.asarray(x, dtype=np.float64)
        self.y = np.asarray(y, dtype=np.float64)
        self.t = np.asarray(t, dtype=np.float64)
        self.n = self.t.shape[0]

        # KDTree over the (x, y) coordinates.  balanced_tree=True is robust to
        # degenerate (collinear) input; compact_nodes keeps memory tight.
        self.tree = _make_kdtree(np.column_stack([self.x, self.y]))

        # Sentinel "no spatial cutoff" flag.
        self._r_cut = None
        self._tau_cut = None
        # Cache spatial-ball query results when r_cut is fixed across all j.
        self._spatial_cache = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_cutoffs(self, tau_cut=None, r_cut=None):
        """Configure the cutoffs used by :meth:`query`.

        Calling this resets the internal caches; safe to re-call between
        optimizer iterations as the cutoffs evolve with the parameters.

        Parameters
        ----------
        tau_cut : float or None
            Temporal cutoff (days).  ``None`` or ``+inf`` disables temporal
            pruning.
        r_cut : float or None
            Spatial cutoff (projected units).  ``None`` or ``+inf`` disables
            spatial pruning.
        """
        self._tau_cut = None if (tau_cut is None or not np.isfinite(tau_cut)) else float(tau_cut)
        self._r_cut = None if (r_cut is None or not np.isfinite(r_cut)) else float(r_cut)
        # (Re)build the spatial-ball cache only when r_cut is finite and we
        # actually want query-time pruning; otherwise lazily return full slices.
        self._spatial_cache = None

    @property
    def tau_cut(self):
        return self._tau_cut

    @property
    def r_cut(self):
        return self._r_cut

    def query(self, j, tau_cut=None, r_cut=None):
        """Return the sorted parent indices ``i < j`` surviving the cutoffs.

        Parameters
        ----------
        j : int
            Target event index (``0 <= j < n``).
        tau_cut, r_cut : float or None, optional
            Per-call overrides; default to the values set via
            :meth:`set_cutoffs`.

        Returns
        -------
        np.ndarray of int
            Ascending parent indices ``i`` with ``i < j``, ``t_j - t_i <= tau_cut``
            (if finite) and ``||x_j - x_i|| <= r_cut`` (if finite).

        Notes
        -----
        For ``j == 0`` an empty array is returned (no earlier events).  The
        temporal constraint is a *contiguous* slice of ``[0, j)`` because the
        catalog is time-sorted, so it is applied as an ``O(log N)``
        ``searchsorted`` rather than a mask.
        """
        if j <= 0:
            return np.empty(0, dtype=np.intp)

        tc = self._tau_cut if tau_cut is None else tau_cut
        rc = self._r_cut if r_cut is None else r_cut

        # --- temporal slice (contiguous because t is sorted ascending) ------
        # All i in [0, j) have t_i <= t_j by sort.  Keep those with
        # t_j - t_i <= tau_cut, i.e. t_i >= t_j - tau_cut.
        if tc is None or not np.isfinite(tc):
            t_lo_idx = 0
        else:
            t_thresh = self.t[j] - tc
            # searchsorted on t[:j] gives the first index whose time >= t_thresh
            t_lo_idx = int(np.searchsorted(self.t[:j], t_thresh, side='left'))

        candidate = np.arange(t_lo_idx, j, dtype=np.intp)

        # --- spatial ball ----------------------------------------------------
        if rc is None or not np.isfinite(rc):
            return candidate

        # Query the KDTree for all points within r_cut of event j (including
        # j itself and later events, which we filter out below).
        nbrs = self.tree.query_ball_point([self.x[j], self.y[j]], rc)
        nbrs = np.asarray(nbrs, dtype=np.intp)
        if nbrs.size == 0:
            return np.empty(0, dtype=np.intp)

        # Keep only earlier events within the temporal window.
        # Both arrays are sorted ascending (KDTree returns sorted indices).
        return np.intersect1d(candidate, nbrs, assume_unique=True)

    def query_all(self, tau_cut=None, r_cut=None):
        """Return parent-index lists for every event as a list of arrays.

        Convenience wrapper around :meth:`query` for the declustering loop,
        which needs all targets at once.  Each element is the result of
        ``query(j, tau_cut, r_cut)``.
        """
        return [self.query(j, tau_cut, r_cut) for j in range(self.n)]
