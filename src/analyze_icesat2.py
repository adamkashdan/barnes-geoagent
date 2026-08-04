"""
Analysis script for ICESat-2 land ice height data over the Barnes Ice Cap.
Co-locates laser altimetry tracks with the 2015 MCoRDS baseline
to build a 7-year (2015-2022) glacier surface elevation time series.
"""
from __future__ import annotations
import os
import glob
import numpy as np
import pandas as pd
import geopandas as gpd
from rasterio.warp import transform
from scipy.spatial import cKDTree
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE_DIR = os.path.join(os.path.dirname(__file__), "..")
DATA_DIR = os.path.join(BASE_DIR, "data")
ICESAT2_DIR = os.path.join(DATA_DIR, "ICESat-2")
DEM_DIR = os.path.join(DATA_DIR, "DEM_2015")
SHP_PATH = os.path.join(DEM_DIR, "barnes_glacier_boundary.shp")
CSV_PATH = os.path.join(DATA_DIR, "2015_85564591_v2", "IRMCR2_20150507_07.csv")


def load_mcords_2015() -> pd.DataFrame:
    """Loads 2015 MCoRDS baseline track."""
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f"Missing MCoRDS 2015 baseline data at {CSV_PATH}")
    df = pd.read_csv(CSV_PATH)
    df = df.dropna(subset=["LAT", "LON", "Actual surface"])
    df["z_surf"] = df["Actual surface"]
    return df


def run_icesat2_analysis():
    print("=== Starting ICESat-2 Laser Altimetry Analysis over Barnes Ice Cap ===")
    
    # 1. Load MCoRDS 2015 baseline
    df_15 = load_mcords_2015()
    
    # Project 2015 to Albers (ESRI:102008) for distance metrics
    print("Projecting 2015 baseline coordinates...")
    xs_15, ys_15 = transform('EPSG:4326', 'ESRI:102008', df_15['LON'].tolist(), df_15['LAT'].tolist())
    df_15['x_proj'] = xs_15
    df_15['y_proj'] = ys_15
    tree_15 = cKDTree(np.column_stack((xs_15, ys_15)))
    
    # 2. Search for ICESat-2 CSV files (ignoring files starting with 'elev_')
    is2_files = glob.glob(os.path.join(ICESAT2_DIR, "**/*.csv"), recursive=True)
    is2_files = [f for f in is2_files if not os.path.basename(f).startswith('elev_')]
    print(f"Found {len(is2_files)} ICESat-2 track files.")
    
    is2_data_by_year = {}
    
    for fp in sorted(is2_files):
        filename = os.path.basename(fp)
        # Parse date from filename e.g. 2018-10-17_1681085410510.csv
        date_str = filename.split("_")[0]
        year = int(date_str.split("-")[0])
        
        print(f"Loading ICESat-2 file {filename} (Year {year})...")
        df_is2 = pd.read_csv(fp)
        df_is2.columns = [c.lower() for c in df_is2.columns]
        df_is2 = df_is2.dropna(subset=['latitude', 'longitude', 'elev'])
        df_is2['year'] = year
        df_is2['date'] = date_str
        
        if year not in is2_data_by_year:
            is2_data_by_year[year] = []
        is2_data_by_year[year].append(df_is2)
        
    # Combine entries per year
    for yr in list(is2_data_by_year.keys()):
        is2_data_by_year[yr] = pd.concat(is2_data_by_year[yr], ignore_index=True)
        print(f"Year {yr}: Loaded {len(is2_data_by_year[yr])} track points.")
        
    # 3. Co-locate ICESat-2 tracks with 2015 MCoRDS tracks (100m search radius)
    print("\n--- Co-locating ICESat-2 profiles with 2015 MCoRDS baseline ---")
    results = []
    
    for yr in sorted(is2_data_by_year.keys()):
        df_is2 = is2_data_by_year[yr]
        
        # Project ICESat-2 coordinates
        xs_is2, ys_is2 = transform('EPSG:4326', 'ESRI:102008', df_is2['longitude'].tolist(), df_is2['latitude'].tolist())
        df_is2['x_proj'] = xs_is2
        df_is2['y_proj'] = ys_is2
        
        # Query 2015 KDTree
        dists, indices = tree_15.query(np.column_stack((xs_is2, ys_is2)))
        df_is2['dist_to_15'] = dists
        df_is2['idx_15'] = indices
        
        # Filter points within 100m threshold
        overlap = df_is2[df_is2['dist_to_15'] <= 100.0].copy()
        
        if not overlap.empty:
            corr_15 = df_15.iloc[overlap['idx_15']].copy().reset_index(drop=True)
            overlap = overlap.reset_index(drop=True)
            
            # dz = ICESat-2 (yr) - MCoRDS (2015)
            # Both are referenced directly to the WGS84 ellipsoid in the datasets
            dz = overlap['elev'] - corr_15['z_surf']
            
            # Remove outliers
            clean_mask = (dz > -100) & (dz < 100)
            dz_clean = dz[clean_mask]
            
            mean_dz = float(dz_clean.mean())
            median_dz = float(np.median(dz_clean))
            std_dz = float(dz_clean.std())
            
            print(f"Year {yr} comparison (relative to 2015 MCoRDS):")
            print(f"  Co-located points: {len(dz_clean)}")
            print(f"  Mean dz (IS2 - MCoRDS): {mean_dz:+.3f} meters")
            print(f"  Median dz: {median_dz:+.3f} meters")
            print(f"  Std Dev: {std_dz:.3f} meters")
            
            results.append({
                'year': yr,
                'mean_dz': mean_dz,
                'median_dz': median_dz,
                'std_dz': std_dz,
                'n_points': len(dz_clean)
            })
            
    if not results:
        print("Error: No co-located tracks found.")
        return
        
    df_res = pd.DataFrame(results)
    df_res.to_csv(os.path.join(DATA_DIR, "icesat2_comparison_summary.csv"), index=False)
    
    # 4. Construct time series (2015 baseline = 0.0 m)
    all_years = [2015]
    relative_surf = [0.0]
    
    # Add ICESat-2 years. Since they match the ellipsoid datum directly,
    # the median dz is the surface elevation change relative to 2015.
    # Note: There is no systematic datum shift, unlike the Penny case.
    for r in results:
        all_years.append(r['year'])
        relative_surf.append(r['median_dz'])
        
    print("\n=== Combined Calibrated 2015-2022 Elevation Change Time Series ===")
    for y, v in zip(all_years, relative_surf):
        print(f"  Year {y}: {v:+.3f} meters (relative to 2015 baseline)")
        
    # Plot the 7-year time series
    fig, ax = plt.subplots(figsize=(6, 5))
    
    # Sort by year
    sorted_idx = np.argsort(all_years)
    years_plot = np.array(all_years)[sorted_idx]
    surf_plot = np.array(relative_surf)[sorted_idx]
    
    ax.plot(years_plot, surf_plot, marker='o', color='purple', linewidth=2.5, markersize=8, label="Barnes Dome Track")
    ax.axhline(0, color='gray', linestyle='--', alpha=0.7)
    
    # Add a linear trend line
    slope, intercept = np.polyfit(years_plot, surf_plot, 1)
    ax.plot(years_plot, slope * years_plot + intercept, color='darkorange', linestyle=':', linewidth=1.5, label=f"Trend ({slope:+.3f} m/yr)")
    
    ax.set_title("Barnes Ice Cap: 7-Year Surface Elevation Change (2015-2022)\n(Combined MCoRDS & ICESat-2 Altimetry)", fontsize=10, fontweight="bold")
    ax.set_xlabel("Year", fontsize=9)
    ax.set_ylabel("Elevation Change (meters relative to 2015)", fontsize=9)
    ax.set_xticks(sorted(all_years))
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="lower left")
    
    trend_path = os.path.join(BASE_DIR, "icesat2_12year_trend.png")  # keeps the file name expected by README/manifest
    fig.savefig(trend_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved 7-year trend plot to: {trend_path}")


if __name__ == "__main__":
    run_icesat2_analysis()
