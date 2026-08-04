"""
Interpolate Barnes 2015 DEM raster from the 10m contours shapefile
and extract the glacier boundary outline.
"""
from __future__ import annotations
import os
import numpy as np
import geopandas as gpd
import rasterio
from rasterio.transform import from_origin
from scipy.interpolate import griddata

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
SHP_PATH = os.path.join(DATA_DIR, "2015_85564591", "Contours_10m.shp")
DEM_OUT_DIR = os.path.join(DATA_DIR, "DEM_2015")
TIF_OUT_PATH = os.path.join(DEM_OUT_DIR, "barnes_dem_2015.tif")
SHP_OUT_PATH = os.path.join(DEM_OUT_DIR, "barnes_glacier_boundary.shp")

def create_dem():
    print("=== Launching DEM Generation from Contours ===")
    if not os.path.exists(DEM_OUT_DIR):
        os.makedirs(DEM_OUT_DIR)
        print(f"Created output directory: {DEM_OUT_DIR}")

    if not os.path.exists(SHP_PATH):
        raise FileNotFoundError(f"Missing contour shapefile at {SHP_PATH}")

    print("Loading contour shapefile...")
    gdf = gpd.read_file(SHP_PATH)
    print(f"Loaded {len(gdf)} contour features. CRS: {gdf.crs}")

    # 1. Extract vertices from contour geometries
    print("Extracting vertices (sampling every 10th point for speed and accuracy)...")
    x_coords = []
    y_coords = []
    z_coords = []

    for geom, elev in zip(gdf.geometry, gdf['ELEV']):
        if geom is not None:
            # Extract coordinates from LineString / MultiLineString
            if geom.geom_type == 'LineString':
                coords = list(geom.coords)
                for c in coords[::10]:
                    x_coords.append(c[0])
                    y_coords.append(c[1])
                    z_coords.append(elev)
            elif geom.geom_type == 'MultiLineString':
                for line in geom.geoms:
                    coords = list(line.coords)
                    for c in coords[::10]:
                        x_coords.append(c[0])
                        y_coords.append(c[1])
                        z_coords.append(elev)

    x_arr = np.array(x_coords)
    y_arr = np.array(y_coords)
    z_arr = np.array(z_coords)
    print(f"Extracted {len(x_arr)} points.")

    # 2. Set up regular grid (300 x 300) covering the contour extent
    x_min, x_max = x_arr.min(), x_arr.max()
    y_min, y_max = y_arr.min(), y_arr.max()
    print(f"Spatial extent of contours:")
    print(f"  X: {x_min:.1f} to {x_max:.1f}")
    print(f"  Y: {y_min:.1f} to {y_max:.1f}")

    grid_size = 300
    x_grid = np.linspace(x_min, x_max, grid_size)
    y_grid = np.linspace(y_max, y_min, grid_size)  # top to bottom for raster orientation
    grid_x, grid_y = np.meshgrid(x_grid, y_grid)

    # 3. Interpolate onto the grid
    print("Interpolating points using linear Delaunay triangulation...")
    grid_z = griddata((x_arr, y_arr), z_arr, (grid_x, grid_y), method="linear")

    # Set NaN values to raster nodata (-9999.0)
    grid_z_write = np.where(np.isnan(grid_z), -9999.0, grid_z).astype(np.float32)

    # Calculate transform
    res_x = (x_max - x_min) / (grid_size - 1)
    res_y = (y_max - y_min) / (grid_size - 1)
    transform = from_origin(x_min, y_max, res_x, res_y)

    # 4. Write to GeoTIFF
    print(f"Writing interpolated DEM to GeoTIFF: {TIF_OUT_PATH}")
    with rasterio.open(
        TIF_OUT_PATH,
        'w',
        driver='GTiff',
        height=grid_size,
        width=grid_size,
        count=1,
        dtype=np.float32,
        crs=gdf.crs,
        transform=transform,
        nodata=-9999.0
    ) as dst:
        dst.write(grid_z_write, 1)
    print("GeoTIFF written successfully.")

    # 5. Extract glacier boundary shapefile (convex hull of all contours)
    print("Generating boundary outline (convex hull of all contour lines)...")
    union_geom = gdf.geometry.unary_union
    hull = union_geom.convex_hull
    boundary_gdf = gpd.GeoDataFrame(geometry=[hull], crs=gdf.crs)
    
    print(f"Saving glacier boundary to Shapefile: {SHP_OUT_PATH}")
    boundary_gdf.to_file(SHP_OUT_PATH)
    print("Glacier boundary written successfully.")
    print("=== DEM Generation Complete ===")

if __name__ == "__main__":
    create_dem()
