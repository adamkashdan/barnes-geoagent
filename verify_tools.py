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

print("\n=== Testing compute_zonal_statistics for PIL ===")
bbox = [-73.5, 69.8, -73.0, 70.3]
stats_pil = compute_zonal_statistics("pleistocene_ice_thickness", bbox)
print("PIL stats inside bbox:", stats_pil)

print("\n=== Testing compare_variables ===")
compare_res = compare_variables("surface_elevation", "pleistocene_ice_thickness", bbox)
print("Comparison results:", compare_res)

print("\n=== Testing generate_map_image for PIL ===")
map_res = generate_map_image("pleistocene_ice_thickness", bbox)
if "image_base64" in map_res:
    print("PIL Map generated successfully. Base64 length:", len(map_res["image_base64"]))
    # Save the generated image as a check to workspace
    with open("test_pil_map.png", "wb") as f:
        import base64
        f.write(base64.b64decode(map_res["image_base64"]))
    print("Saved test_pil_map.png to workspace root.")
else:
    print("Error generating map:", map_res)
