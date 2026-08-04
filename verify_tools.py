import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
from tools import list_datasets, query_point_data, compute_zonal_statistics, generate_map_image, compare_variables

print("=== Testing list_datasets ===")
datasets = list_datasets()
print(datasets)

print("\n=== Testing query_point_data (at center of Barnes Ice Cap) ===")
# Latitude ~70.278, Longitude ~-73.400
pt_res = query_point_data(70.278, -73.400)
print(pt_res)

print("\n=== Testing compute_zonal_statistics for 2022 Elevation ===")
bbox = [-73.5, 69.8, -73.0, 70.3]
stats_22 = compute_zonal_statistics("surface_elevation_2022", bbox)
print("2022 surface stats inside bbox:", stats_22)

print("\n=== Testing compare_variables (2015 vs 2022 elevation) ===")
compare_res = compare_variables("surface_elevation", "surface_elevation_2022", bbox)
print("Comparison results:", compare_res)

print("\n=== Testing generate_map_image for 2022 surface elevation ===")
map_res = generate_map_image("surface_elevation_2022", bbox)
if "image_base64" in map_res:
    print("2022 Surface Map generated successfully. Base64 length:", len(map_res["image_base64"]))
    with open("test_surface_2022_map.png", "wb") as f:
        import base64
        f.write(base64.b64decode(map_res["image_base64"]))
    print("Saved test_surface_2022_map.png to workspace root.")
else:
    print("Error generating map:", map_res)

