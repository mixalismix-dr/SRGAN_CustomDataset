import rasterio
from rasterio.io import MemoryFile
from rasterio import Affine

# Paths to the files
original_tif = r"C:\Users\mike_\OneDrive\Desktop\MSc Geomatics\Master Thesis\Codebases\SRGAN_CustomDataset\custom_dataset\train_HR1\tile_0_0.tif"
non_georef_tif = r"C:\Users\mike_\OneDrive\Desktop\MSc Geomatics\Master Thesis\Codebases\SRGAN_CustomDataset\other scripts\output_rgba_image.tif"
georef_output_tif = r"C:\Users\mike_\OneDrive\Desktop\MSc Geomatics\Master Thesis\Codebases\SRGAN_CustomDataset\other scripts\output_rgba_image_geo.tif"

# Read georeferencing from the original file
with rasterio.open(original_tif) as src:
    transform = src.transform
    print(transform)
    crs = src.crs
    profile = src.profile
    # new_transform = Affine.scale()    # Read pixel values


# Apply georeferencing to the non-georeferenced file
with rasterio.open(non_georef_tif) as src_sr:
    data = src_sr.read()
    t2 = Affine(0.0625, transform[1], transform[2], transform[3], -0.0625, transform[5])
    profile.update({
        "transform": t2,
        "crs": crs,
        "height": src_sr.height,
        "width": src_sr.width,
        "count": 4,
        "nodata": None

    })

    with rasterio.open(georef_output_tif, "w", **profile) as dst:
        dst.write(data)

print(f"Georeferenced TIFF saved to: {georef_output_tif}")
