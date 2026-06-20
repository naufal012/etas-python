"""
Benchmark script for comparing Baseline ETAS with Truncated & Renormalized ETAS.

This script demonstrates the performance speedup obtained by pruning
distant space-time interactions using KDTree lookups and analytic
renormalization of the truncated kernels.

The renormalized model is *not* an approximation — the truncated kernels
are divided by closed-form normalizing constants so they remain valid
probability densities over the surviving support.  The only source of
approximation error is the finite pruning threshold ``eps`` itself
(smaller eps → tighter truncation → less error).
"""

import time
import pandas as pd
from etas import search_isc, catalog, etas


def main():
    print("=" * 60)
    print("ETAS Benchmark: Baseline vs. Truncated & Renormalized ETAS")
    print("=" * 60)

    # 1. Fetch a moderate-sized dataset from ISC.
    print("\n[1/4] Fetching benchmark dataset from ISC (Japan, 2010)...")
    df = search_isc(
        start_year=2010, start_month=1, start_day=1,
        end_year=2010, end_month=12, end_day=31,
        searchshape="RECT",
        lat_bot=30.0, lat_top=45.0,
        long_left=130.0, long_right=145.0,
        mag_min=4.5
    )
    print(f"Fetched {len(df)} events from ISC.")

    # 2. Build the ETAS Catalog.
    print("\n[2/4] Building ETAS Catalog...")
    cat = catalog(
        data=df,
        time_begin="2010-01-01",
        study_start="2010-03-01",
        study_length=250,  # days
        mag_threshold=4.5,
        dist_unit="degree"
    )
    num_events = len(cat.revents)
    print(f"Catalog ready! {num_events} events in the target study period.")

    # 3. Fit Baseline ETAS (No pruning — O(N²) pairwise interactions).
    print("\n[3/4] Fitting Baseline ETAS (No Pruning)...")
    fit_baseline = etas(
        cat,
        no_itr=5,
        epsilon=None,  # no truncation → full O(N²)
        verbose=False
    )
    time_baseline = fit_baseline.exectime
    loglik_baseline = fit_baseline.opt['loglik']
    print(f"  Baseline ETAS finished in {time_baseline:.2f} s.")
    print(f"  Baseline Log-Likelihood: {loglik_baseline:.4f}")

    # 4. Fit Truncated & Renormalized ETAS (KDTree pruning + analytic norms).
    print("\n[4/4] Fitting Truncated & Renormalized ETAS (eps=1e-4)...")
    print("  Kernel values below eps are pruned; surviving kernel is renormalized.")
    fit_pruned = etas(
        cat,
        no_itr=5,
        eps_t=1e-4,       # temporal truncation threshold
        eps_s=1e-4,       # spatial truncation threshold
        verbose=False
    )
    time_pruned = fit_pruned.exectime
    loglik_pruned = fit_pruned.opt['loglik']
    print(f"  Renormalized ETAS finished in {time_pruned:.2f} s.")
    print(f"  Renormalized Log-Likelihood: {loglik_pruned:.4f}")

    # 5. Results.
    print("\n" + "=" * 60)
    print("BENCHMARK RESULTS")
    print("=" * 60)
    speedup = time_baseline / time_pruned if time_pruned > 0 else 0
    ll_diff = abs(loglik_baseline - loglik_pruned)

    print(f"  Catalog size        : {num_events} events")
    print(f"  Baseline time       : {time_baseline:.2f} s")
    print(f"  Renormalized time   : {time_pruned:.2f} s")
    print(f"  Speedup             : {speedup:.2f}x faster")
    print(f"  |Δ log-likelihood|  : {ll_diff:.6f}")
    print("=" * 60)
    print("Notes:")
    print("  - The truncated kernels are divided by analytic renormalization")
    print("    constants (G_norm, F_norm) so they remain valid densities.")
    print("  - The only source of difference from the full model is the finite")
    print("    pruning threshold eps.  Smaller eps → tighter cutoffs → less")
    print("    approximation error but less speedup.")
    print("  - The KDTree + searchsorted neighbour lookups replace the O(N²)")
    print("    pairwise-mask overhead, giving order-of-magnitude speedups for")
    print("    large catalogs (N > 10,000).")


if __name__ == "__main__":
    main()
