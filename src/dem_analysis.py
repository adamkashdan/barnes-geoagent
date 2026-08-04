"""
DEM 2015 vs MCoRDS 2015 surface elevation comparison.
"""
from __future__ import annotations
import os
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.warp import transform
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE_DIR = os.path.join(os.path.dirname(__file__), "..")
DATA_DIR = os.path.join(BASE_DIR, "data")
DEM_DIR = os.path.join(DATA_DIR, "DEM_2015")
TIF_PATH = os.path.join(DEM_DIR, "barnes_dem_2015.tif")
SHP_PATH = os.path.join(DEM_DIR, "barnes_glacier_boundary.shp")
CSV_PATH = os.path.join(DATA_DIR, "2015_85564591_v2", "IRMCR2_20150507_07.csv")


def load_mcords_surface_data() -> pd.DataFrame:
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f"Missing primary data file at {CSV_PATH}")
    df = pd.read_csv(CSV_PATH)
    df = df.dropna(subset=["LAT", "LON", "Actual surface"])
    # Rename column to match expectations
    df["z_2015"] = df["Actual surface"]
    return df


def run_dem_analysis():
    print("=== Launching DEM 2015 vs MCoRDS 2015 Analysis ===")
    
    if not os.path.exists(TIF_PATH) or not os.path.exists(SHP_PATH):
        print(f"Error: Missing DEM TIFF or boundary Shapefile in {DEM_DIR}")
        return
        
    print("Loading glacier boundary shapefile...")
    gdf = gpd.read_file(SHP_PATH)
    
    print("Opening 2015 DEM raster...")
    with rasterio.open(TIF_PATH) as src:
        dem_crs = src.crs.to_string()
        print(f"DEM CRS: {dem_crs}")
        
        # Read downsampled DEM for visualization
        factor = 1
        dem_data = src.read(1)
        dem_data = np.where(dem_data == src.nodata, np.nan, dem_data)
        
        dem_extent = [src.bounds.left, src.bounds.right, src.bounds.bottom, src.bounds.top]
        
        # 1. Plot DEM Topography
        print("Generating DEM topography map...")
        fig, ax = plt.subplots(figsize=(7, 6))
        im = ax.imshow(dem_data, cmap="terrain", extent=dem_extent, origin="upper")
        gdf.boundary.plot(ax=ax, color="black", linewidth=1.5, label="Glacier Boundary")
        ax.set_title("Barnes Ice Cap (2015): Digital Elevation Model (DEM)", fontsize=10, fontweight="bold")
        ax.set_xlabel("Easting (meters, North America Albers)", fontsize=8)
        ax.set_ylabel("Northing (meters, North America Albers)", fontsize=8)
        ax.grid(True, linestyle="--", alpha=0.5)
        fig.colorbar(im, ax=ax, label="Elevation (m above sea level)")
        
        dem_map_path = os.path.join(BASE_DIR, "dem_2015_2016_topography.png")
        fig.savefig(dem_map_path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved DEM map to: {dem_map_path}")
        
        # 2. Compare elevations
        print("Loading MCoRDS 2015 surface elevations...")
        m_df = load_mcords_surface_data()
        
        print("Projecting points to DEM's CRS...")
        xs, ys = transform('EPSG:4326', src.crs, m_df['LON'].tolist(), m_df['LAT'].tolist())
        m_df['x_dem'] = xs
        m_df['y_dem'] = ys
        
        valid_idx = (
            (m_df['x_dem'] >= src.bounds.left) & (m_df['x_dem'] <= src.bounds.right) &
            (m_df['y_dem'] >= src.bounds.bottom) & (m_df['y_dem'] <= src.bounds.top)
        )
        comp_df = m_df[valid_idx].copy()
        
        print(f"Sampling DEM elevations at {len(comp_df)} flight points...")
        coords = [(x, y) for x, y in zip(comp_df['x_dem'], comp_df['y_dem'])]
        comp_df['z_dem'] = [val[0] for val in src.sample(coords)]
        
        # Filter out NoData samples (-9999)
        comp_df = comp_df[comp_df['z_dem'] != src.nodata]
        print(f"Extracted valid comparisons for {len(comp_df)} points.")
        
        if comp_df.empty:
            print("Error: No overlapping coordinates between MCoRDS track and DEM.")
            return
            
        comp_df['dz'] = comp_df['z_dem'] - comp_df['z_2015']
        
        # Filter extreme outliers
        comp_df = comp_df[(comp_df['dz'] >= -100.0) & (comp_df['dz'] <= 100.0)]
        print(f"Cleaned points count: {len(comp_df)}")
        
        mean_dz = comp_df['dz'].mean()
        median_dz = comp_df['dz'].median()
        std_dz = comp_df['dz'].std()
        pearson_r = comp_df['z_2015'].corr(comp_df['z_dem'])
        rmse = np.sqrt(np.mean(comp_df['dz']**2))
        
        summary_text = f"""=== DEM 2015 vs MCoRDS 2015 Surface Comparison ===
Comparison Points: {len(comp_df)}
Mean Elevation Change (dz = z_dem - z_2015): {mean_dz:.3f} meters
Median Elevation Change: {median_dz:.3f} meters
Standard Deviation of dz: {std_dz:.3f} meters
Root Mean Squared Error (RMSE): {rmse:.3f} meters
Pearson Correlation Coefficient: {pearson_r:.5f}
"""
        print(summary_text)
        
        # Save summary report
        txt_out_path = os.path.join(BASE_DIR, "dem_analysis_summary.txt")
        with open(txt_out_path, "w") as f:
            f.write(summary_text)
        print(f"Saved text report to: {txt_out_path}")
        
        # Save matched CSV
        csv_out_path = os.path.join(DATA_DIR, "dem_comparison_points.csv")
        comp_df[['LAT', 'LON', 'z_2015', 'z_dem', 'dz']].to_csv(csv_out_path, index=False)
        print(f"Saved comparison CSV to: {csv_out_path}")
        
        # Plot elevation change map
        print("Generating elevation change map...")
        fig, ax = plt.subplots(figsize=(6, 5))
        sc = ax.scatter(comp_df['LON'], comp_df['LAT'], c=comp_df['dz'], cmap="RdBu", vmin=-20, vmax=20, s=2, alpha=0.8)
        ax.set_title("Barnes Ice Cap: Elevation Difference (dz = z_dem - z_2015)\nOperation IceBridge tracks vs 2015 Contours DEM", fontsize=9, fontweight="bold")
        ax.set_xlabel("Longitude (deg W)", fontsize=8)
        ax.set_ylabel("Latitude (deg N)", fontsize=8)
        ax.grid(True, linestyle="--", alpha=0.5)
        cbar = fig.colorbar(sc, ax=ax, shrink=0.8)
        cbar.set_label("Elevation Difference dz (meters)", fontsize=8)
        
        change_map_path = os.path.join(BASE_DIR, "glacier_elevation_change_map.png")
        fig.savefig(change_map_path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved elevation change map to: {change_map_path}")
        
        # Plot scatter plot comparison
        print("Generating elevation comparison scatter plot...")
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.scatter(comp_df['z_2015'], comp_df['z_dem'], color="blue", s=1, alpha=0.4, label="Track points")
        
        lims = [
            min(comp_df['z_2015'].min(), comp_df['z_dem'].min()),
            max(comp_df['z_2015'].max(), comp_df['z_dem'].max())
        ]
        ax.plot(lims, lims, "r--", linewidth=1.5, label="1:1 line (no change)")
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.set_title("Elevation Comparison: 2015 MCoRDS vs 2015 Contours DEM", fontsize=10, fontweight="bold")
        ax.set_xlabel("2015 Surface Elevation (m above sea level)", fontsize=8)
        ax.set_ylabel("2015 Contours DEM Elevation (m above sea level)", fontsize=8)
        ax.legend(loc="upper left", fontsize=8)
        ax.grid(True, linestyle="--", alpha=0.5)
        
        scatter_path = os.path.join(BASE_DIR, "glacier_elevation_comparison_scatter.png")
        fig.savefig(scatter_path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved scatter plot to: {scatter_path}")


if __name__ == "__main__":
    run_dem_analysis()
