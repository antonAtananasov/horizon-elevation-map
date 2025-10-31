import rasterio
from rasterio.transform import from_origin
import numpy as np

test_file = 'copernicus\Copernicus_DSM_COG_10_N41_00_E024_00_DEM.tif'

with rasterio.open(test_file) as tiff:

    # Example array (1 band, 512x512)
    data = np.random.rand(1, 512, 512).astype("float32")

    # Define georeferencing
    transform = from_origin(west=10.0, north=45.0, xsize=0.0001, ysize=0.0001)  # top-left corner and pixel size

    meta = {
        "driver": "GTiff",
        "height": data.shape[1],
        "width": data.shape[2],
        "count": data.shape[0],
        "dtype": str(data.dtype),
        "crs": tiff.crs,
        "transform": tiff.transform,
    }
    print(meta)

    # Save GeoTIFF
    with rasterio.open("new_geodata.tif", "w", **meta) as dst:
        dst.write(data)

    print("✅ GeoTIFF saved with new geodata.")
