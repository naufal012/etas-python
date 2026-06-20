"""
test_neighbors.py — KDTree neighbor lookup vs brute-force masks.
"""

import numpy as np
import pytest


def test_kdtree_matches_brute_force():
    """NeighborIndex.query results equal brute-force for random catalogs."""
    from etas.src.neighbors import NeighborIndex

    rng = np.random.default_rng(42)
    N = 300
    x = rng.uniform(0, 1, N)
    y = rng.uniform(0, 1, N)
    t = np.sort(rng.uniform(0, 100, N))

    nbr = NeighborIndex(x, y, t)

    tau_cut, r_cut = 5.0, 0.3
    nbr.set_cutoffs(tau_cut, r_cut)

    for j in [5, 50, 100, 200, 299]:
        kd_idx = nbr.query(j, tau_cut, r_cut)

        # Brute force
        bf_idx = []
        for i in range(j):
            dt = t[j] - t[i]
            r = np.sqrt((x[j] - x[i])**2 + (y[j] - y[i])**2)
            if dt <= tau_cut and r <= r_cut:
                bf_idx.append(i)
        bf_idx = np.sort(bf_idx)

        assert np.array_equal(np.sort(kd_idx), bf_idx), (
            f"Mismatch at j={j}: KDTree has {len(kd_idx)}, BF has {len(bf_idx)}")


def test_no_cutoffs_returns_all_earlier():
    """With infinite cutoffs, query(j) should return [0, 1, ..., j-1]."""
    from etas.src.neighbors import NeighborIndex

    N = 50
    x = np.random.uniform(0, 1, N)
    y = np.random.uniform(0, 1, N)
    t = np.sort(np.random.uniform(0, 100, N))

    nbr = NeighborIndex(x, y, t)

    for j in [5, 25, 49]:
        idx = nbr.query(j, tau_cut=None, r_cut=None)
        expected = np.arange(j, dtype=np.intp)
        assert np.array_equal(idx, expected)


def test_j_zero_returns_empty():
    """query(0) always returns empty (no earlier events)."""
    from etas.src.neighbors import NeighborIndex

    nbr = NeighborIndex([0.0], [0.0], [0.0])
    idx = nbr.query(0)
    assert len(idx) == 0


def test_spatial_only_cutoff():
    """With only r_cut finite, tau is not applied."""
    from etas.src.neighbors import NeighborIndex

    N = 50
    x = np.random.uniform(0, 1, N)
    y = np.random.uniform(0, 1, N)
    t = np.sort(np.random.uniform(0, 100, N))

    nbr = NeighborIndex(x, y, t)
    r_cut = 0.1

    for j in [10, 30, 49]:
        kd_idx = nbr.query(j, tau_cut=None, r_cut=r_cut)
        bf_idx = []
        for i in range(j):
            r = np.sqrt((x[j] - x[i])**2 + (y[j] - y[i])**2)
            if r <= r_cut:
                bf_idx.append(i)
        assert np.array_equal(np.sort(kd_idx), np.sort(bf_idx))


def test_query_all():
    """query_all returns a list of N arrays."""
    from etas.src.neighbors import NeighborIndex

    N = 20
    x = np.random.uniform(0, 1, N)
    y = np.random.uniform(0, 1, N)
    t = np.sort(np.random.uniform(0, 10, N))

    nbr = NeighborIndex(x, y, t)
    nbr.set_cutoffs(tau_cut=2.0, r_cut=0.5)

    all_nbrs = nbr.query_all()
    assert len(all_nbrs) == N
    assert len(all_nbrs[0]) == 0  # j=0 has no earlier events
