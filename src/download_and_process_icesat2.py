"""
Script to download latest ICESat-2 (2023-2024) ATL06 HDF5 data over the Barnes Ice Cap
using Earthdata Login credentials from .env, extract coordinates/elevations,
and export them as CSV files matching the existing project data structure.
"""
from __future__ import annotations
import os
import sys
import h5py
import pandas as pd
import numpy as np
import requests
from requests.auth import HTTPBasicAuth

BASE_DIR = os.path.join(os.path.dirname(__file__), "..")
DATA_DIR = os.path.join(BASE_DIR, "data")
ICESAT2_DIR = os.path.join(DATA_DIR, "ICESat-2")
ENV_PATH = os.path.join(BASE_DIR, ".env")

# Barnes Ice Cap Bounding Box
BBOX = {"west": -74.77, "south": 69.54, "east": -71.80, "north": 70.65}


def load_dotenv():
    if os.path.exists(ENV_PATH):
        with open(ENV_PATH) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    os.environ[k.strip()] = v.strip("'\"")


load_dotenv()


class EarthdataSession(requests.Session):
    """Custom session to preserve basic authentication credentials when redirected."""
    def rebuild_auth(self, prepared_request, response):
        headers = prepared_request.headers
        if 'Authorization' not in headers:
            if self.auth:
                prepared_request.prepare_auth(self.auth)


# Selected high-quality tracks (286, 339, 400) for October 2023 and October 2024
TARGET_GRANULES = [
    # 2023 Track 339
    (
        "ATL06_20231011050059_03392103_007_01.h5",
        "https://data.nsidc.earthdatacloud.nasa.gov/nsidc-cumulus-prod-protected/ATLAS/ATL06/007/2023/10/11/ATL06_20231011050059_03392103_007_01.h5",
        "2023-10-11", "t339"
    ),
    # 2023 Track 400
    (
        "ATL06_20231015045248_04002103_007_01.h5",
        "https://data.nsidc.earthdatacloud.nasa.gov/nsidc-cumulus-prod-protected/ATLAS/ATL06/007/2023/10/15/ATL06_20231015045248_04002103_007_01.h5",
        "2023-10-15", "t400"
    ),
    # 2023 Track 286
    (
        "ATL06_20231007175422_02862105_007_01.h5",
        "https://data.nsidc.earthdatacloud.nasa.gov/nsidc-cumulus-prod-protected/ATLAS/ATL06/007/2023/10/07/ATL06_20231007175422_02862105_007_01.h5",
        "2023-10-07", "t286"
    ),
    # 2024 Track 339
    (
        "ATL06_20241008113931_03392503_007_01.h5",
        "https://data.nsidc.earthdatacloud.nasa.gov/nsidc-cumulus-prod-protected/ATLAS/ATL06/007/2024/10/08/ATL06_20241008113931_03392503_007_01.h5",
        "2024-10-08", "t339"
    ),
    # 2024 Track 400
    (
        "ATL06_20241012113125_04002503_007_01.h5",
        "https://data.nsidc.earthdatacloud.nasa.gov/nsidc-cumulus-prod-protected/ATLAS/ATL06/007/2024/10/12/ATL06_20241012113125_04002503_007_01.h5",
        "2024-10-12", "t400"
    ),
    # 2024 Track 286
    (
        "ATL06_20241005003237_02862505_007_01.h5",
        "https://data.nsidc.earthdatacloud.nasa.gov/nsidc-cumulus-prod-protected/ATLAS/ATL06/007/2024/10/05/ATL06_20241005003237_02862505_007_01.h5",
        "2024-10-05", "t286"
    )
]


def download_file(session: EarthdataSession, url: str, filename: str) -> str | None:
    """Downloads HDF5 file from NSIDC with redirects."""
    os.makedirs(ICESAT2_DIR, exist_ok=True)
    dest_path = os.path.join(ICESAT2_DIR, filename)
    
    if os.path.exists(dest_path):
        print(f"File {filename} already exists. Skipping download.")
        return dest_path
        
    print(f"Downloading {filename}...")
    try:
        response = session.get(url, stream=True)
        response.raise_for_status()
        
        # Check for HTML login redirection
        content_type = response.headers.get('Content-Type', '')
        if 'text/html' in content_type:
            print(f"  Error: Received HTML login page instead of data for {filename}. Check Earthdata credentials.")
            return None
            
        with open(dest_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"  Successfully downloaded to {dest_path}")
        return dest_path
    except Exception as e:
        print(f"  Failed to download {filename}: {e}")
        return None


def process_hdf5_to_csv(filepath: str, date_str: str, track_str: str):
    """Extracts coordinates and land ice heights from ATL06 HDF5 and saves as CSV."""
    print(f"Processing HDF5 file {os.path.basename(filepath)}...")
    dfs = []
    
    with h5py.File(filepath, 'r') as f:
        # Loop over all 6 beams (strong and weak)
        for beam in ['gt1l', 'gt1r', 'gt2l', 'gt2r', 'gt3l', 'gt3r']:
            group_path = f"{beam}/land_ice_segments"
            if group_path in f:
                try:
                    lat = f[f"{group_path}/latitude"][:]
                    lon = f[f"{group_path}/longitude"][:]
                    h = f[f"{group_path}/h_li"][:]
                    qual = f[f"{group_path}/atl06_quality_summary"][:]
                    
                    # Filter invalid filler values and select high quality points
                    valid_mask = (h < 1e9) & (qual == 0)
                    
                    if valid_mask.any():
                        df_beam = pd.DataFrame({
                            'latitude': lat[valid_mask],
                            'longitude': lon[valid_mask],
                            'elev': h[valid_mask]
                        })
                        dfs.append(df_beam)
                except Exception:
                    continue
                    
    if not dfs:
        print("  Warning: No valid land ice segments found in HDF5.")
        return
        
    df = pd.concat(dfs, ignore_index=True)
    
    # Filter points to the Barnes Ice Cap bounding box
    df = df[
        (df["latitude"] >= BBOX["south"]) & (df["latitude"] <= BBOX["north"]) &
        (df["longitude"] >= BBOX["west"]) & (df["longitude"] <= BBOX["east"])
    ].copy()
    
    print(f"  Extracted {len(df)} valid track points within Barnes Ice Cap.")
    
    if df.empty:
        return
        
    # Output to the same folder structure
    # Folder name e.g. elev_2023-10-11_t339_processed
    out_dir = os.path.join(ICESAT2_DIR, f"elev_{date_str}_{track_str}_processed")
    os.makedirs(out_dir, exist_ok=True)
    
    # CSV file name e.g. 2023-10-11_processed.csv
    out_csv = os.path.join(out_dir, f"{date_str}_processed.csv")
    df.to_csv(out_csv, index=False)
    print(f"  Saved processed CSV data to: {out_csv}")


def main():
    username = os.environ.get("EARTHDATA_USERNAME")
    password = os.environ.get("EARTHDATA_PASSWORD")
    
    if not username or not password:
        print("Error: NASA Earthdata credentials (EARTHDATA_USERNAME, EARTHDATA_PASSWORD) not found in .env.")
        sys.exit(1)
        
    session = EarthdataSession()
    session.auth = HTTPBasicAuth(username, password)
    
    print(f"Starting process for {len(TARGET_GRANULES)} files...")
    
    for filename, url, date_str, track_str in TARGET_GRANULES:
        local_path = download_file(session, url, filename)
        if local_path:
            process_hdf5_to_csv(local_path, date_str, track_str)
            # Remove HDF5 file after processing to save disk space
            try:
                os.remove(local_path)
                print(f"  Cleaned up HDF5 file: {filename}")
            except Exception as e:
                print(f"  Error cleaning up HDF5 file: {e}")
                
    print("\n=== All ICESat-2 (2023-2024) processing completed! ===")


if __name__ == "__main__":
    main()
