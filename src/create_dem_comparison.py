"""
Creates a 2015 raster DEM from MCoRDS L2 flight track points using linear interpolation,
and compares it as a continuous raster grid with the 2015 Contours DEM.
Saves the interpolated DEM as a GeoTIFF and outputs comparison maps.
"""
from __future__ import annotations
import os
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.transform import from_origin
from rasterio.warp import transform
from scipy.interpolate import griddata
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE_DIR = os.path.join(os.path.dirname(__file__), "..")
DATA_DIR = os.path.join(BASE_DIR, "data")
DEM_DIR = os.path.join(DATA_DIR, "DEM_2015")

TIF_DEM_PATH = os.path.join(DEM_DIR, "barnes_dem_2015.tif")
SHP_PATH = os.path.join(DEM_DIR, "barnes_glacier_boundary.shp")
CSV_PATH = os.path.join(DATA_DIR, "2015_85564591_v2", "IRMCR2_20150507_07.csv")
TIF_2015_PATH = os.path.join(DATA_DIR, "barnes_dem_2015_interpolated.tif")


def load_mcords_points() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Loads 2015 MCoRDS points and returns projected coordinates (x, y) and elevation values."""
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f"Missing primary data file at {CSV_PATH}")
    df = pd.read_csv(CSV_PATH)
    df = df.dropna(subset=["LAT", "LON", "Actual surface"])
    
    # Project to Albers (ESRI:102008)
    print("Projecting MCoRDS 2015 points to Albers (ESRI:102008)...")
    xs, ys = transform('EPSG:4326', 'ESRI:102008', df['LON'].tolist(), df['LAT'].tolist())
    return np.array(xs), np.array(ys), df['Actual surface'].to_numpy()


def run_dem_interpolation():
    print("=== Creating 2015 DEM from MCoRDS Track Points ===")
    
    # Load 2015 points
    xs, ys, zs = load_mcords_points()
    points = np.column_stack((xs, ys))
    
    # Find bounding box of flight lines
    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()
    print(f"Flight line extent UTM (ESRI:102008):")
    print(f"  X: {x_min:.1f} to {x_max:.1f}")
    print(f"  Y: {y_min:.1f} to {y_max:.1f}")
    
    # Define regular grid size (300 x 300 cells)
    grid_size = 300
    x_grid = np.linspace(x_min, x_max, grid_size)
    y_grid = np.linspace(y_max, y_min, grid_size)  # top to bottom for raster rows
    grid_x, grid_y = np.meshgrid(x_grid, y_grid)
    
    # Interpolate using linear method
    print("Interpolating points to grid (linear method)...")
    grid_z15 = griddata(points, zs, (grid_x, grid_y), method="linear")
    
    # Mask out nan values for writing to TIFF
    grid_z15_write = np.where(np.isnan(grid_z15), -9999.0, grid_z15).astype(np.float32)
    
    # Define GeoTIFF transform parameters
    res_x = (x_max - x_min) / (grid_size - 1)
    res_y = (y_max - y_min) / (grid_size - 1)
    transform_new = from_origin(x_min, y_max, res_x, res_y)
    
    print(f"Saving interpolated 2015 DEM to GeoTIFF: {TIF_2015_PATH}")
    with rasterio.open(
        TIF_2015_PATH,
        'w',
        driver='GTiff',
        height=grid_size,
        width=grid_size,
        count=1,
        dtype=np.float32,
        crs='ESRI:102008',
        transform=transform_new,
        nodata=-9999.0
    ) as dst:
        dst.write(grid_z15_write, 1)
        
    print("Interpolated DEM GeoTIFF written successfully.")
    
    # 5. Load 2015 Contours DEM and sample it at the grid locations
    print("Sampling 2015 Contours DEM at the same grid cells...")
    if not os.path.exists(TIF_DEM_PATH):
        print(f"Error: Missing contours DEM raster at {TIF_DEM_PATH}")
        return
        
    with rasterio.open(TIF_DEM_PATH) as src_dem:
        # Sample the DEM at the meshgrid points
        grid_coords = [(x, y) for x, y in zip(grid_x.ravel(), grid_y.ravel())]
        grid_dem_flat = [val[0] for val in src_dem.sample(grid_coords)]
        grid_dem = np.array(grid_dem_flat).reshape((grid_size, grid_size))
        grid_dem = np.where(grid_dem == src_dem.nodata, np.nan, grid_dem)
        
    # Mask out extrapolation zones where MCoRDS interpolation is NaN
    grid_dem[np.isnan(grid_z15)] = np.nan
    
    # Apply geodetic vertical offset correction (+6.81m) to align Contours DEM to WGS84 ellipsoid
    # Or subtract 6.81m from Contours DEM to compare.
    # In dem_analysis, dz = z_dem - z_2015 was +6.81m.
    # Thus, z_dem_corrected = z_dem - 6.81 m
    geoid_offset = 6.81
    grid_dem_corrected = grid_dem - geoid_offset
    
    # Calculate continuous difference (dz = corrected_dem - mcoords_interpolated)
    grid_diff = grid_dem_corrected - grid_z15
    
    # Load boundary Shapefile
    gdf = gpd.read_file(SHP_PATH)
    dem_extent = [x_min, x_max, y_min, y_max]
    
    # 6. Save Plot 1: Interpolated 2015 DEM
    print("Plotting interpolated 2015 MCoRDS DEM...")
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(grid_z15, extent=dem_extent, cmap="terrain", origin="upper")
    gdf.boundary.plot(ax=ax, color="black", linewidth=1.5)
    ax.set_title("Barnes Ice Cap 2015: Interpolated MCoRDS DEM", fontsize=10, fontweight="bold")
    ax.set_xlabel("Easting (meters, North America Albers)", fontsize=8)
    ax.set_ylabel("Northing (meters, North America Albers)", fontsize=8)
    ax.grid(True, linestyle="--", alpha=0.5)
    fig.colorbar(im, ax=ax, label="Elevation (m WGS84)")
    
    img1_path = os.path.join(BASE_DIR, "barnes_dem_2015_interpolated.png")
    fig.savefig(img1_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved interpolated DEM map to: {img1_path}")
    
    # 7. Save Plot 2: Corrected continuous difference (glacier thinning/residuals)
    print("Plotting elevation difference raster...")
    fig, ax = plt.subplots(figsize=(6, 5))
    # We display difference, showing residuals after datum correction
    im = ax.imshow(grid_diff, extent=dem_extent, cmap="RdBu", vmin=-10, vmax=10, origin="upper")
    gdf.boundary.plot(ax=ax, color="black", linewidth=1.5)
    ax.set_title("Barnes Ice Cap 2015: Elevation Difference\n(Corrected Contours DEM minus Interpolated MCoRDS)", fontsize=9, fontweight="bold")
    ax.set_xlabel("Easting (meters, North America Albers)", fontsize=8)
    ax.set_ylabel("Northing (meters, North America Albers)", fontsize=8)
    ax.grid(True, linestyle="--", alpha=0.5)
    fig.colorbar(im, ax=ax, label="Elevation Residual (meters)")
    
    img2_path = os.path.join(BASE_DIR, "glacier_dem_change_raster.png")
    fig.savefig(img2_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved difference raster map to: {img2_path}")
    print("=== Continuous DEM Comparison Complete ===")


if __name__ == "__main__":
    run_dem_interpolation()
