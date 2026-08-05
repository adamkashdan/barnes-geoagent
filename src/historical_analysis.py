"""
Validation analysis script for Barnes Ice Cap comparing MCoRDS 2015
surface elevation with IceBridge ATM L2 2015 surface elevation at overlapping tracks.
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
DEM_DIR = os.path.join(DATA_DIR, "DEM_2015")
SHP_PATH = os.path.join(DEM_DIR, "barnes_glacier_boundary.shp")
CSV_PATH = os.path.join(DATA_DIR, "2015_85564591_v2", "IRMCR2_20150507_07.csv")
ATM_DIR = os.path.join(DATA_DIR, "IceBridge ATM L2 Icessn Elevation, Slope, and Roughness V002")


def run_historical_analysis():
    print("=== Starting Barnes Ice Cap MCoRDS 2015 vs ATM 2015 Comparison ===")
    
    # 1. Load MCoRDS 2015 baseline
    print("Loading MCoRDS 2015 baseline data...")
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f"Missing MCoRDS CSV at {CSV_PATH}")
    df_15 = pd.read_csv(CSV_PATH)
    df_15 = df_15.dropna(subset=["LAT", "LON", "Actual surface"])
    df_15["z_surf"] = df_15["Actual surface"]
    df_15["ice_thickness"] = df_15["THICK"]
    
    # Project to Albers for KDTree queries
    print("Projecting MCoRDS points...")
    xs_15, ys_15 = transform('EPSG:4326', 'ESRI:102008', df_15['LON'].tolist(), df_15['LAT'].tolist())
    df_15['x_proj'] = xs_15
    df_15['y_proj'] = ys_15
    tree_15 = cKDTree(np.column_stack((xs_15, ys_15)))
    
    # 2. Load ATM 2015 files (ignoring smooth files)
    atm_files = [f for f in glob.glob(os.path.join(ATM_DIR, "**/*.csv"), recursive=True) if '_smooth' not in f]
    print(f"Found {len(atm_files)} raw ATM files.")
    
    all_atm = []
    for fp in atm_files:
        try:
            df_atm = pd.read_csv(fp)
            df_atm.columns = [c.strip() for c in df_atm.columns] # Strip spaces
            df_atm = df_atm.dropna(subset=['Longitude(deg)', 'Latitude(deg)', 'WGS84_Ellipsoid_Height(m)'])
            all_atm.append(df_atm)
        except Exception as e:
            print(f"Error parsing {fp}: {e}")
            
    if not all_atm:
        print("Error: No valid ATM data loaded.")
        return
        
    df_atm_all = pd.concat(all_atm, ignore_index=True)
    print(f"Total ATM points: {len(df_atm_all)}")
    
    # Project ATM coordinates
    xs_atm, ys_atm = transform('EPSG:4326', 'ESRI:102008', df_atm_all['Longitude(deg)'].tolist(), df_atm_all['Latitude(deg)'].tolist())
    df_atm_all['x_proj'] = xs_atm
    df_atm_all['y_proj'] = ys_atm
    
    # Filter non-finite projected values
    valid_mask = np.isfinite(df_atm_all['x_proj']) & np.isfinite(df_atm_all['y_proj'])
    df_atm_all = df_atm_all[valid_mask].copy()
    
    # 3. Co-locate ATM and MCoRDS tracks (100m search radius)
    print("Co-locating ATM tracks with MCoRDS baseline...")
    dists, indices = tree_15.query(np.column_stack((df_atm_all['x_proj'], df_atm_all['y_proj'])))
    df_atm_all['dist_to_15'] = dists
    df_atm_all['idx_15'] = indices
    
    # Filter overlap
    overlap = df_atm_all[df_atm_all['dist_to_15'] <= 100.0].copy()
    if overlap.empty:
        print("Error: No overlapping coordinates found between ATM and MCoRDS.")
        return
        
    corr_15 = df_15.iloc[overlap['idx_15']].copy().reset_index(drop=True)
    overlap = overlap.reset_index(drop=True)
    
    # dz_surf = ATM (laser) - MCoRDS (radar)
    overlap['dz'] = overlap['WGS84_Ellipsoid_Height(m)'] - corr_15['z_surf']
    
    # Filter extreme outliers
    clean_mask = (overlap['dz'] >= -100.0) & (overlap['dz'] <= 100.0)
    overlap = overlap[clean_mask].copy()
    corr_15 = corr_15[clean_mask].copy()
    
    mean_dz = overlap['dz'].mean()
    median_dz = overlap['dz'].median()
    std_dz = overlap['dz'].std()
    rmse = np.sqrt(np.mean(overlap['dz']**2))
    
    summary_text = f"""=== ATM 2015 vs MCoRDS 2015 Co-Location Validation ===
Co-Located Overlapping Points: {len(overlap)}
Mean Elevation Difference (dz = z_atm - z_mcoords): {mean_dz:.3f} meters
Median Elevation Difference: {median_dz:.3f} meters
Standard Deviation of dz: {std_dz:.3f} meters
Root Mean Squared Error (RMSE): {rmse:.3f} meters
"""
    print(summary_text)
    
    # Save text summary
    summary_path = os.path.join(BASE_DIR, "historical_analysis_summary.txt")
    with open(summary_path, "w") as f:
        f.write(summary_text)
    print(f"Saved report to: {summary_path}")
    
    # Load boundary Shapefile for overlays
    gdf = gpd.read_file(SHP_PATH)
    
    # Plot 1: historical_glacier_trends.png (Histogram of dz)
    print("Generating dz distribution histogram (historical_glacier_trends.png)...")
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.hist(overlap['dz'], bins=50, color='skyblue', edgecolor='black', alpha=0.8)
    ax.axvline(median_dz, color='red', linestyle='--', linewidth=1.5, label=f"Median dz: {median_dz:+.2f}m")
    ax.set_title("Barnes Ice Cap 2015: ATM L2 vs MCoRDS L2\nElevation Difference Distribution", fontsize=10, fontweight="bold")
    ax.set_xlabel("Elevation Difference (z_atm - z_mcoords, meters)", fontsize=9)
    ax.set_ylabel("Frequency (Count)", fontsize=9)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(fontsize=9)
    
    plot1_path = os.path.join(BASE_DIR, "historical_glacier_trends.png")
    fig.savefig(plot1_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved trends plot to: {plot1_path}")
    
    # Plot 2: historical_elevation_change_map.png (Spatial differences)
    print("Generating difference map (historical_elevation_change_map.png)...")
    fig, ax = plt.subplots(figsize=(6, 5))
    sc = ax.scatter(overlap['x_proj'], overlap['y_proj'], c=overlap['dz'], cmap="RdBu", vmin=-10, vmax=10, s=2, alpha=0.8)
    gdf.boundary.plot(ax=ax, color="black", linewidth=1.5)
    ax.set_title("Barnes Ice Cap 2015: Spatial Elevation Difference\n(z_atm minus z_mcoords along track)", fontsize=9, fontweight="bold")
    ax.set_xlabel("Easting (m, North America Albers)", fontsize=8)
    ax.set_ylabel("Northing (m, North America Albers)", fontsize=8)
    ax.grid(True, linestyle="--", alpha=0.5)
    cbar = fig.colorbar(sc, ax=ax, shrink=0.8)
    cbar.set_label("z_atm - z_mcoords (meters)", fontsize=8)
    
    plot2_path = os.path.join(BASE_DIR, "historical_elevation_change_map.png")
    fig.savefig(plot2_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved difference map to: {plot2_path}")
    
    # Plot 3: historical_thickness_change_map.png (Thickness along overlapping tracks)
    print("Generating thickness map (historical_thickness_change_map.png)...")
    fig, ax = plt.subplots(figsize=(6, 5))
    # Filter out invalid thickness values (-9999) for color mapping
    valid_thick = corr_15[corr_15['ice_thickness'] >= 0]
    sc = ax.scatter(valid_thick['x_proj'], valid_thick['y_proj'], c=valid_thick['ice_thickness'], cmap="Blues", s=2, alpha=0.8)
    gdf.boundary.plot(ax=ax, color="black", linewidth=1.5)
    ax.set_title("Barnes Ice Cap 2015: MCoRDS Ice Thickness\nat Overlapping ATM Tracks", fontsize=9, fontweight="bold")
    ax.set_xlabel("Easting (m, North America Albers)", fontsize=8)
    ax.set_ylabel("Northing (m, North America Albers)", fontsize=8)
    ax.grid(True, linestyle="--", alpha=0.5)
    cbar = fig.colorbar(sc, ax=ax, shrink=0.8)
    cbar.set_label("Ice Thickness (meters)", fontsize=8)
    
    plot3_path = os.path.join(BASE_DIR, "historical_thickness_change_map.png")
    fig.savefig(plot3_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved thickness map to: {plot3_path}")
    print("=== Validation Comparison Complete ===")


if __name__ == "__main__":
    run_historical_analysis()
