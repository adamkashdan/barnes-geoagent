"""
Analysis script to compare the 2015 contours-derived DEM with the new 2022 DEM.
Performs geodetic reprojection, pixel-by-pixel elevation difference,
statistical analysis over the glaciated region, and saves change maps.
"""
from __future__ import annotations
import os
import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling
import geopandas as gpd
from shapely.geometry import mapping
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE_DIR = os.path.join(os.path.dirname(__file__), "..")
DATA_DIR = os.path.join(BASE_DIR, "data")
DEM_2015_PATH = os.path.join(DATA_DIR, "DEM_2015", "barnes_dem_2015.tif")
DEM_2022_PATH = os.path.join(DATA_DIR, "DEM_2022", "barnes_dem_2022.tif")
SHP_PATH = os.path.join(DATA_DIR, "DEM_2015", "barnes_glacier_boundary.shp")


def run_dem_comparison():
    print("=== Starting Barnes Ice Cap 2015 vs 2022 DEM Analysis ===")
    
    if not os.path.exists(DEM_2015_PATH) or not os.path.exists(DEM_2022_PATH):
        raise FileNotFoundError("Missing one or both DEM files for comparison.")

    # 1. Open both rasters and prepare destination array matching 2015 DEM grid
    with rasterio.open(DEM_2015_PATH) as src_15:
        meta_15 = src_15.meta.copy()
        dem_15 = src_15.read(1)
        transform_15 = src_15.transform
        crs_15 = src_15.crs
        nodata_15 = src_15.nodata or -9999.0
        
    print(f"2015 DEM Grid: shape={dem_15.shape}, CRS={crs_15}")
    
    dem_22_resampled = np.zeros_like(dem_15, dtype=np.float32)
    
    with rasterio.open(DEM_2022_PATH) as src_22:
        print(f"2022 DEM Grid: shape={src_22.shape}, CRS={src_22.crs}")
        
        # Warp/reproject 2022 DEM onto the 2015 grid transformation
        reproject(
            source=rasterio.band(src_22, 1),
            destination=dem_22_resampled,
            src_transform=src_22.transform,
            src_crs=src_22.crs,
            dst_transform=transform_15,
            dst_crs=crs_15,
            resampling=Resampling.bilinear,
            dst_nodata=nodata_15
        )
        
    # Mask invalid/no-data values in both DEMs
    # In 2015 DEM, invalid values are nodata_15 or <= 0
    # In 2022 DEM, we check for values <= 0 or nodata_15
    mask_15 = (dem_15 != nodata_15) & (dem_15 > 0)
    mask_22 = (dem_22_resampled != nodata_15) & (dem_22_resampled > -100) & (dem_22_resampled < 3000)
    valid_mask = mask_15 & mask_22
    
    # 2. Load glacier boundary shapefile to restrict analysis to the ice cap
    if os.path.exists(SHP_PATH):
        gdf = gpd.read_file(SHP_PATH)
        # Ensure projection matches
        if gdf.crs != crs_15:
            gdf = gdf.to_crs(crs_15)
        
        # Rasterize geometry to mask out non-glacier cells
        from rasterio.features import geometry_mask
        glacier_mask = ~geometry_mask(
            gdf.geometry,
            out_shape=dem_15.shape,
            transform=transform_15,
            all_touched=True,
            invert=False
        )
        # Combine valid data mask with glacier boundary mask
        valid_mask = valid_mask & glacier_mask
        
    # Both DEMs are in the CGVD2013 orthometric geoid height system, so no datum shift is needed.
    dz = dem_22_resampled - dem_15
    
    # Filter difference values to glacier boundary valid mask
    dz_ice = dz[valid_mask]
    
    mean_change = float(np.mean(dz_ice))
    median_change = float(np.median(dz_ice))
    std_change = float(np.std(dz_ice))
    min_change = float(np.min(dz_ice))
    max_change = float(np.max(dz_ice))
    
    print("\n=== elevation change results (2022 minus 2015) ===")
    print(f"  Glacier surface pixels compared: {len(dz_ice)}")
    print(f"  Mean elevation change: {mean_change:+.3f} meters")
    print(f"  Median elevation change: {median_change:+.3f} meters")
    print(f"  Std Dev of change: {std_change:.3f} meters")
    print(f"  Max thinning (minimum change): {min_change:+.3f} meters")
    print(f"  Max thickening: {max_change:+.3f} meters")
    
    # 3. Save the difference raster as a new GeoTIFF
    out_meta = meta_15.copy()
    out_meta.update({
        "driver": "GTiff",
        "dtype": "float32",
        "nodata": -9999.0
    })
    
    dz_output = np.where(valid_mask, dz, -9999.0).astype(np.float32)
    out_path = os.path.join(DATA_DIR, "DEM_2022", "barnes_elevation_change_2015_2022.tif")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    
    with rasterio.open(out_path, "w", **out_meta) as dst:
        dst.write(dz_output, 1)
        
    print(f"\nSaved elevation change raster to: {out_path}")
    
    # 4. Generate continuous elevation change map
    fig, ax = plt.subplots(figsize=(6, 5))
    dz_plot = np.where(valid_mask, dz, np.nan)
    
    # Center colormap at 0 using RdBu (Red = thinning, Blue = thickening)
    im = ax.imshow(
        dz_plot,
        cmap="RdBu",
        vmin=-15,
        vmax=15,
        extent=[
            transform_15[2], transform_15[2] + transform_15[0]*dem_15.shape[1],
            transform_15[5] + transform_15[4]*dem_15.shape[0], transform_15[5]
        ]
    )
    
    if os.path.exists(SHP_PATH):
        gdf.boundary.plot(ax=ax, color="black", linewidth=1.5)
        
    ax.set_title("Barnes Ice Cap: DEM Elevation Change (2015-2022)\n(Resampled & Ellipsoidal-Aligned Raster difference)", fontsize=10, fontweight="bold")
    ax.set_xlabel("Easting (m, North America Albers)", fontsize=8)
    ax.set_ylabel("Northing (m, North America Albers)", fontsize=8)
    ax.grid(True, linestyle="--", alpha=0.5)
    
    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Elevation Change dz (meters)", fontsize=8)
    
    map_path = os.path.join(BASE_DIR, "glacier_dem_change_2015_2022.png")
    fig.savefig(map_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved elevation change map to: {map_path}")


if __name__ == "__main__":
    run_dem_comparison()
