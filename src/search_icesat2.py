"""
Helper script to query the NASA CMR API to find ICESat-2 ATL06 HDF5 granules
intersecting the Barnes Ice Cap region for the years 2023 and 2024.
"""
from __future__ import annotations
import requests

CMR_URL = "https://cmr.earthdata.nasa.gov/search/granules.json"
# Barnes Ice Cap Bounding Box: [west, south, east, north]
# For CMR, bbox parameter is: [longitude_west, latitude_south, longitude_east, latitude_north]
BBOX = "-74.77,69.54,-71.80,70.65"


def search_granules():
    params = {
        "short_name": "ATL06",
        "bounding_box": BBOX,
        "temporal": "2023-01-01T00:00:00Z,2024-12-31T23:59:59Z",
        "page_size": 50
    }
    
    print("Querying NASA CMR API...")
    response = requests.get(CMR_URL, params=params)
    response.raise_for_status()
    
    data = response.json()
    entries = data.get("feed", {}).get("entry", [])
    
    print(f"Found {len(entries)} granules matching search criteria.\n")
    
    results = []
    for entry in entries:
        title = entry.get("title")
        # Extract download link (usually links with rel="http://esipfed.org/ns/fedsearch/1.1/data#")
        links = entry.get("links", [])
        download_url = None
        for link in links:
            if "data#" in link.get("rel", "") and link.get("href", "").endswith(".h5"):
                download_url = link.get("href")
                break
                
        if download_url:
            print(f"Granule: {title}")
            print(f"  URL: {download_url}")
            results.append((title, download_url))
            
    return results


if __name__ == "__main__":
    search_granules()
