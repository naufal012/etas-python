import pandas as pd
import numpy as np
from etas import catalog, etas

# 1. Create a dummy catalog with 50 events in a tight space
np.random.seed(42)
N = 10
df = pd.DataFrame({
    'date': pd.date_range('2020-01-01', periods=N, freq='D').strftime('%Y-%m-%d'),
    'time': ['12:00:00'] * N,
    'lat': np.random.uniform(30.0, 31.0, N),
    'long': np.random.uniform(130.0, 131.0, N),
    'mag': np.random.uniform(4.5, 6.0, N),
    'z': np.random.uniform(5.0, 25.0, N) # Depth column for 3D!
})

# 2. Process catalog
print("Processing catalog...")
cat = catalog(
    data=df,
    time_begin="2020-01-01",
    study_start="2020-01-02",
    study_length=9, # days
    mag_threshold=4.5,
    dist_unit="degree"
)

print("\nRunning 3D ETAS Fit... (mver=1, is_3d=True)")
# 3. Run ETAS optimization
# Since it's a dummy catalog, it might not converge well, 
# but it will show us the structure of the print_etas output.
res = etas(
    cat,
    param0=[0.5, 0.5, 0.05, 1.0, 1.2, 0.05, 1.5, 0.5, 2.0], # 9 params for 3D
    mver=1,
    engine='cpu',
    is_3d=True
)

# print_etas is called automatically by etas() if we didn't disable it,
# but we can print the dictionary result as well.
from etas import print_etas
print_etas(res)

from etas.R.rates import rates
import matplotlib.pyplot as plt

print("\nCalculating 3D slices...")
# Calculate intensity at surface (z=0)
rate_surface = rates(res, slice_depth=0.0)

# Calculate intensity deep underground (z=20.0)
rate_deep = rates(res, slice_depth=20.0)

# Plotting
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

extent = [rate_surface['x'].min(), rate_surface['x'].max(), rate_surface['y'].min(), rate_surface['y'].max()]
im1 = axes[0].imshow(rate_surface['lamb'], cmap='jet', extent=extent, origin='lower', aspect='auto')
axes[0].set_title('ETAS Intensity (Depth = 0km)')
axes[0].set_xlabel('Longitude')
axes[0].set_ylabel('Latitude')
fig.colorbar(im1, ax=axes[0], label='Intensity (events/day/deg^2)')

im2 = axes[1].imshow(rate_deep['lamb'], cmap='jet', extent=extent, origin='lower', aspect='auto')
axes[1].set_title('ETAS Intensity (Depth = 20km)')
axes[1].set_xlabel('Longitude')
axes[1].set_ylabel('Latitude')
fig.colorbar(im2, ax=axes[1], label='Intensity (events/day/deg^2)')

plt.tight_layout()
out_path = r'C:\Users\Naufal\.gemini\antigravity\brain\e45151ee-202d-4676-9f2d-91428260c32d\etas_3d_slices.png'
plt.savefig(out_path, dpi=150)
print(f"Saved plot to {out_path}")
print("\nDone!")
