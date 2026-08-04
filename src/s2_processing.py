"""
Sentinel-2 satellite image processing module for the Barnes Ice Cap.
Provides tools to compute NDSI (Normalized Difference Snow Index)
and extract True Color RGB images from Sentinel-2 SAFE folders.
"""
from __future__ import annotations
import os
import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import transform
from rasterio.windows import from_bounds
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE_DIR = os.path.join(os.path.dirname(__file__), "..")
DATA_DIR = os.path.join(BASE_DIR, "data")


def find_band_file(safe_folder: str, band_suffix: str) -> str | None:
    """Recursively search for a band file (e.g., '_B03.jp2', '_B03_10m.jp2') inside a Sentinel-2 folder.
    Prefers higher resolution (10m > 20m > 60m)."""
    folder_path = os.path.join(DATA_DIR, safe_folder)
    if not os.path.exists(folder_path):
        return None
    
    matches = []
    for root, _, files in os.walk(folder_path):
        for f in files:
            if f.endswith(".jp2") and band_suffix in f:
                matches.append(os.path.join(root, f))
                
    if not matches:
        return None
        
    # Sort matches to prioritize highest spatial resolution
    def sort_key(path):
        fname = os.path.basename(path)
        if "_10m" in fname:
            return 0
        elif "_20m" in fname:
            return 1
        elif "_60m" in fname:
            return 2
        else:
            return 3
            
    matches.sort(key=sort_key)
    return matches[0]


def get_window_and_transform(src: rasterio.DatasetReader, bbox: list) -> tuple[rasterio.windows.Window, int, int]:
    """Converts lat/lon bbox [west, south, east, north] to UTM window.
    Caps the window to the raster boundaries and computes output shape."""
    west, south, east, north = bbox
    # Transform coordinates from WGS84 to UTM
    xs, ys = transform('EPSG:4326', src.crs, [west, east], [south, north])
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    
    # Create window
    window = from_bounds(x_min, y_min, x_max, y_max, src.transform)
    
    # Intersect with the raster bounds to avoid reading outside the file
    raster_window = rasterio.windows.Window(0, 0, src.width, src.height)
    window = window.intersection(raster_window)
    
    # Round to integers
    window = window.round()
    
    # Determine downsampled output shape (max dimension 1000px to prevent memory overflow)
    max_dim = 1000
    w_width = int(window.width)
    w_height = int(window.height)
    
    if w_width <= 0 or w_height <= 0:
        raise ValueError("Bounding box does not intersect with this Sentinel-2 tile.")
        
    if w_width > max_dim or w_height > max_dim:
        if w_width > w_height:
            out_w = max_dim
            out_h = int(w_height * max_dim / w_width)
        else:
            out_h = max_dim
            out_w = int(w_width * max_dim / w_height)
    else:
        out_w = w_width
        out_h = w_height
        
    return window, out_h, out_w


def process_ndsi(safe_folder: str, bbox: list | None = None) -> tuple[np.ndarray, list] | dict:
    """Computes NDSI = (Green - SWIR) / (Green + SWIR) = (B03 - B11) / (B03 + B11)
    bbox: [west, south, east, north]
    Returns:
        - ndsi array, [west, south, east, north] of actual cropped extent
    """
    b03_path = find_band_file(safe_folder, "_B03")
    b11_path = find_band_file(safe_folder, "_B11")
    
    if not b03_path or not b11_path:
        return {"error": f"Missing required bands (B03 or B11) in {safe_folder}"}
        
    try:
        with rasterio.open(b03_path) as src_b03:
            # Default bbox to entire tile bounds if none is provided
            if bbox is None:
                xs = [src_b03.bounds.left, src_b03.bounds.right]
                ys = [src_b03.bounds.bottom, src_b03.bounds.top]
                lons, lats = transform(src_b03.crs, 'EPSG:4326', xs, ys)
                bbox = [min(lons), min(lats), max(lons), max(lats)]
            
            try:
                window, out_h, out_w = get_window_and_transform(src_b03, bbox)
            except ValueError as e:
                return {"error": str(e)}
                
            # Read B03 (Green)
            green = src_b03.read(1, window=window, out_shape=(out_h, out_w), resampling=Resampling.bilinear).astype(np.float32)
            
            # Compute actual geographic extent of the cropped window
            win_bounds = rasterio.windows.bounds(window, src_b03.transform)
            xs_win = [win_bounds[0], win_bounds[2]]
            ys_win = [win_bounds[1], win_bounds[3]]
            lons_win, lats_win = transform(src_b03.crs, 'EPSG:4326', xs_win, ys_win)
            actual_bbox = [min(lons_win), min(lats_win), max(lons_win), max(lats_win)]
            
        with rasterio.open(b11_path) as src_b11:
            # Read B11 (SWIR) into the same output shape
            swir = src_b11.read(1, window=window, out_shape=(out_h, out_w), resampling=Resampling.bilinear).astype(np.float32)
            
        # Compute NDSI
        mask = (green == 0) | (swir == 0)
        ndsi = (green - swir) / (green + swir + 1e-10)
        ndsi[mask] = np.nan
        
        return ndsi, actual_bbox
    except Exception as e:
        return {"error": f"Failed to process NDSI: {e}"}


def process_rgb(safe_folder: str, bbox: list | None = None) -> tuple[np.ndarray, list] | dict:
    """Generates True Color RGB image using B04 (Red), B03 (Green), and B02 (Blue) bands."""
    b04_path = find_band_file(safe_folder, "_B04")
    b03_path = find_band_file(safe_folder, "_B03")
    b02_path = find_band_file(safe_folder, "_B02")
    
    if not b04_path or not b03_path or not b02_path:
        # Check if pre-computed True Color Image (TCI) is available (often 10m res)
        tci_path = find_band_file(safe_folder, "_TCI")
        if tci_path:
            return process_tci(tci_path, bbox)
        return {"error": f"Missing required RGB bands in {safe_folder}"}
        
    try:
        with rasterio.open(b04_path) as src_b04:
            if bbox is None:
                xs = [src_b04.bounds.left, src_b04.bounds.right]
                ys = [src_b04.bounds.bottom, src_b04.bounds.top]
                lons, lats = transform(src_b04.crs, 'EPSG:4326', xs, ys)
                bbox = [min(lons), min(lats), max(lons), max(lats)]
                
            try:
                window, out_h, out_w = get_window_and_transform(src_b04, bbox)
            except ValueError as e:
                return {"error": str(e)}
                
            red = src_b04.read(1, window=window, out_shape=(out_h, out_w), resampling=Resampling.bilinear)
            
            # Compute actual geographic extent
            win_bounds = rasterio.windows.bounds(window, src_b04.transform)
            xs_win = [win_bounds[0], win_bounds[2]]
            ys_win = [win_bounds[1], win_bounds[3]]
            lons_win, lats_win = transform(src_b04.crs, 'EPSG:4326', xs_win, ys_win)
            actual_bbox = [min(lons_win), min(lats_win), max(lons_win), max(lats_win)]
            
        with rasterio.open(b03_path) as src_b03:
            green = src_b03.read(1, window=window, out_shape=(out_h, out_w), resampling=Resampling.bilinear)
            
        with rasterio.open(b02_path) as src_b02:
            blue = src_b02.read(1, window=window, out_shape=(out_h, out_w), resampling=Resampling.bilinear)
            
        # Normalize bands to 0-255 using percentiles for enhanced contrast
        rgb = np.dstack([red, green, blue]).astype(np.float32)
        
        # Simple contrast enhancement (clip at 2nd and 98th percentiles)
        for i in range(3):
            band = rgb[:, :, i]
            valid = band[band > 0]
            if len(valid) > 0:
                p2, p98 = np.percentile(valid, [2, 98])
                band_clipped = np.clip(band, p2, p98)
                rgb[:, :, i] = (band_clipped - p2) / (p98 - p2 + 1e-10)
            else:
                rgb[:, :, i] = 0.0
                
        # Mask out values that are zero in all bands
        mask = (red == 0) & (green == 0) & (blue == 0)
        rgb[mask] = 1.0  # Set backgrounds to white
        
        return rgb, actual_bbox
    except Exception as e:
        return {"error": f"Failed to process RGB: {e}"}


def process_tci(tci_path: str, bbox: list | None = None) -> tuple[np.ndarray, list] | dict:
    """Alternative fast RGB extraction using pre-built True Color Image (TCI) jp2."""
    try:
        with rasterio.open(tci_path) as src:
            if bbox is None:
                xs = [src.bounds.left, src.bounds.right]
                ys = [src.bounds.bottom, src.bounds.top]
                lons, lats = transform(src.crs, 'EPSG:4326', xs, ys)
                bbox = [min(lons), min(lats), max(lons), max(lats)]
                
            try:
                window, out_h, out_w = get_window_and_transform(src, bbox)
            except ValueError as e:
                return {"error": str(e)}
                
            r = src.read(1, window=window, out_shape=(out_h, out_w), resampling=Resampling.bilinear)
            g = src.read(2, window=window, out_shape=(out_h, out_w), resampling=Resampling.bilinear)
            b = src.read(3, window=window, out_shape=(out_h, out_w), resampling=Resampling.bilinear)
            
            rgb = np.dstack([r, g, b]).astype(np.float32) / 255.0
            
            mask = (r == 0) & (g == 0) & (b == 0)
            rgb[mask] = 1.0  # White background
            
            win_bounds = rasterio.windows.bounds(window, src.transform)
            xs_win = [win_bounds[0], win_bounds[2]]
            ys_win = [win_bounds[1], win_bounds[3]]
            lons_win, lats_win = transform(src.crs, 'EPSG:4326', xs_win, ys_win)
            actual_bbox = [min(lons_win), min(lats_win), max(lons_win), max(lats_win)]
            
            return rgb, actual_bbox
    except Exception as e:
        return {"error": f"Failed to process TCI: {e}"}
