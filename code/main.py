import os
import argparse
from categorize_tiles import categorize_tiles

def main(city, iteration, total_samples):
    # Define the base data directory
    base_data_dir = "/mnt/SRGAN/data"

    # Define paths for HR and LR raster data
    hr_raster_path = os.path.join(base_data_dir, f"{city}_hr.tif")
    lr_raster_path = os.path.join(base_data_dir, f"{city}_lr.tif")

    # Define path for the cleaned land use shapefile
    land_use_path = os.path.join(base_data_dir, f"land_use_{city}.shp")

    # Define output directories for the extracted tiles
    output_hr_dir = os.path.join(base_data_dir, f"iteration{iteration}", "train_HR")
    output_lr_dir = os.path.join(base_data_dir, f"iteration{iteration}", "train_LR")

    # Ensure output directories exist
    os.makedirs(output_hr_dir, exist_ok=True)
    os.makedirs(output_lr_dir, exist_ok=True)

    # List of all cities for sample distribution
    all_cities = ["delft", "utrecht", "zwolle", "rotterdam"]

    # Call the categorize_tiles function to process the data
    categorize_tiles(
        hr_raster=hr_raster_path,
        lr_raster=lr_raster_path,
        land_use_path=land_use_path,
        city=city,
        output_hr_dir=output_hr_dir,
        output_lr_dir=output_lr_dir,
        total_samples=total_samples,
        all_cities=all_cities,
        iteration=iteration
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process city data for tile categorization.")
    parser.add_argument("city", type=str, choices=["delft", "utrecht", "zwolle", "rotterdam"],
                        help="The city to process. Choose from: delft, utrecht, zwolle, rotterdam.")
    parser.add_argument("iteration", type=int, choices=[1, 2],
                        help="The iteration number. Choose 1 or 2.")
    parser.add_argument("total_samples", type=int,
                        help="Total number of samples to extract across all cities.")

    args = parser.parse_args()
    main(args.city, args.iteration, args.total_samples)
