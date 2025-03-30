import os
import geopandas as gpd
from tqdm import tqdm


def clean_and_export_land_use(input_root, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    for folder in tqdm(os.listdir(input_root), desc="Processing Cities"):
        base_path = os.path.join(input_root, folder, folder, "Data")
        if not os.path.isdir(base_path):
            continue

        gpkg_files = [f for f in os.listdir(base_path) if f.endswith(".gpkg")]
        if not gpkg_files:
            continue

        gpkg_file = gpkg_files[0]
        gpkg_path = os.path.join(base_path, gpkg_file)

        # Extract base layer name (remove _v013 or similar suffix)
        raw_layer_name = os.path.splitext(gpkg_file)[0]
        layer_name = raw_layer_name.split("_v")[0]

        try:
            gdf = gpd.read_file(gpkg_path, layer=layer_name)
            if "class_2018" not in gdf.columns:
                print(f"Skipping {folder}: 'class_2018' not found.")
                continue

            gdf = gdf[["geometry", "class_2018"]].copy()
            gdf["class_2018"] = gdf["class_2018"].str.strip()

            category_mapping = {
                "Continuous urban fabric (S.L.:>80%)": "High-Density Urban",
                "Discontinuous dense urban fabric (S.L.: 50%-80%)": "High-Density Urban",
                "Discontinuous medium density urban fabric (S.L.: 30%-50%)": "Low-Density Urban",
                "Discontinuous low density urban fabric (S.L.: 10%-30%)": "Low-Density Urban",
                "Discontinuous very low density urban fabric (S.L.:<10%)": "Low-Density Urban",
                "Isolated structures": "Low-Density Urban",
                "Water": "Non-Urban / Green",
                "Sports and leisure facilities": "Non-Urban / Green",
                "Land without current use": "Non-Urban / Green",
                "Forests": "Non-Urban / Green",
                "Green urban areas": "Non-Urban / Green",
                "Pastures": "Non-Urban / Green",
                "Herbaceous vegetation associations (natural grassland, moors...)": "Non-Urban / Green",
                "Open spaces with little or no vegetation (beaches, dunes, bare rocks, glaciers)": "Non-Urban / Green",
                "Wetlands": "Non-Urban / Green",
                "Arable land (annual crops)": "Non-Urban / Green",
                "Industrial, commercial, public, military and private units": "Industrial & Infrastructure",
                "Port areas": "Industrial & Infrastructure",
                "Railways and associated land": "Industrial & Infrastructure",
                "Mineral extraction and dump sites": "Industrial & Infrastructure",
                "Other roads and associated land": "Industrial & Infrastructure",
                "Construction sites": "Industrial & Infrastructure",
                "Fast transit roads and associated land": "Industrial & Infrastructure",
                "Airports": "Industrial & Infrastructure",
                "Permanent crops (vineyards, fruit trees, olive groves)": "Non-Urban / Green"
            }

            gdf["class_2018"] = (
                gdf["class_2018"]
                .str.replace(r"\s*:\s*", ": ", regex=True)
                .str.replace(r"\s*-\s*", "-", regex=True)
                .str.replace(r"\s*>\s*", ">", regex=True)
                .str.replace(r"\s*<\s*", "<", regex=True)
                .str.strip()
            )
            gdf["merged_category"] = gdf["class_2018"].map(category_mapping)
            gdf = gdf.dropna(subset=["merged_category"])

            city_name = folder.split("_")[1].lower()
            output_path = os.path.join(output_dir, f"land_use_{city_name}.shp")
            gdf[["geometry", "merged_category"]].to_file(output_path)

        except Exception as e:
            print(f"Failed to process {folder}: {e}")


# Run
base_dir = r"C:\Users\mike_\Downloads\170584\Results"
output_dir = r"C:\Users\mike_\Downloads\Cleaned_Landuse"

clean_and_export_land_use(base_dir, output_dir)
