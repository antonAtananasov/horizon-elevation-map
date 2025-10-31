import PIL
from PIL import Image
import rasterio
import matplotlib.pyplot as plt
from time import time

plot = False
image_path = "bulgaria.tif"
vertical_scale = 1
latitude_scale = 83000
longitude_scale = 83000  # 111000
resolution_rescale = .2
backend = "cuda"  # 'cpu' or 'cuda'

output_folder = "output"

directions = [
    "North-east",
    "North",
    "North-west",
    "West",
    "South-west",
    "South",
    "South-east",
    "East",
]

tiles = [
    "copernicus/Copernicus_DSM_COG_10_N41_00_E022_00_DEM.tif",
    "copernicus/Copernicus_DSM_COG_10_N41_00_E023_00_DEM.tif",
    "copernicus/Copernicus_DSM_COG_10_N41_00_E024_00_DEM.tif",
    "copernicus/Copernicus_DSM_COG_10_N41_00_E025_00_DEM.tif",
    "copernicus/Copernicus_DSM_COG_10_N41_00_E026_00_DEM.tif",
    "copernicus/Copernicus_DSM_COG_10_N41_00_E027_00_DEM.tif",
    "copernicus/Copernicus_DSM_COG_10_N41_00_E028_00_DEM.tif",
    "copernicus/Copernicus_DSM_COG_10_N42_00_E022_00_DEM.tif",
    "copernicus/Copernicus_DSM_COG_10_N42_00_E023_00_DEM.tif",
    "copernicus/Copernicus_DSM_COG_10_N42_00_E024_00_DEM.tif",
    "copernicus/Copernicus_DSM_COG_10_N42_00_E025_00_DEM.tif",
    "copernicus/Copernicus_DSM_COG_10_N42_00_E026_00_DEM.tif",
    "copernicus/Copernicus_DSM_COG_10_N42_00_E027_00_DEM.tif",
    "copernicus/Copernicus_DSM_COG_10_N42_00_E028_00_DEM.tif",
    "copernicus/Copernicus_DSM_COG_10_N43_00_E022_00_DEM.tif",
    "copernicus/Copernicus_DSM_COG_10_N43_00_E023_00_DEM.tif",
    "copernicus/Copernicus_DSM_COG_10_N43_00_E024_00_DEM.tif",
    "copernicus/Copernicus_DSM_COG_10_N43_00_E025_00_DEM.tif",
    "copernicus/Copernicus_DSM_COG_10_N43_00_E026_00_DEM.tif",
    "copernicus/Copernicus_DSM_COG_10_N43_00_E027_00_DEM.tif",
    "copernicus/Copernicus_DSM_COG_10_N43_00_E028_00_DEM.tif",
    "copernicus/Copernicus_DSM_COG_10_N44_00_E022_00_DEM.tif",
    "copernicus/Copernicus_DSM_COG_10_N44_00_E023_00_DEM.tif",
    "copernicus/Copernicus_DSM_COG_10_N44_00_E024_00_DEM.tif",
    "copernicus/Copernicus_DSM_COG_10_N44_00_E025_00_DEM.tif",
    "copernicus/Copernicus_DSM_COG_10_N44_00_E026_00_DEM.tif",
    "copernicus/Copernicus_DSM_COG_10_N44_00_E027_00_DEM.tif",
    "copernicus/Copernicus_DSM_COG_10_N44_00_E028_00_DEM.tif",
]


if backend == "cuda":
    import cupy as np
elif backend == "cpu":
    import numpy as np
else:
    raise Exception("No corresponding backend method")

PIL.Image.MAX_IMAGE_PIXELS = 933120000

def generate_filename(input_file, output_name, ext, output_folder=".", sep="_"):
    return f"{output_folder}/{input_file}{sep}{output_name}.{ext}"


def load_image(
    image_path,
    latitude_scale,
    longitude_scale,
    rescale_factor=1,
    plot=False,
):
    image = Image.open(image_path)
    new_width, new_height = round(image.width * rescale_factor), round(
        image.height * rescale_factor
    )
    resized_image = image.resize((new_width, new_height))
    print(f"Image resolution: {image.width}x{image.height}")
    image_array = np.array(resized_image, dtype=np.float32)
    print("Image array shape:", image_array.shape)

    if plot:
        plt.imshow(image_array.get() if backend != "cpu" else image_array)
        plt.title("Height")
        plt.colorbar()
        plt.show()

    if len(image_array.shape) > 2:
        image_array = image_array[:, :, 0]
    h, w = image_array.shape

    geodata = {
        "driver": "GTiff",
        "height": h,
        "width": w,
        "count": 1,
        "dtype": str(image_array.dtype),
        "compress": "lzw",
    }
    if image_path.endswith(".tif"):
        with rasterio.open(image_path) as tiff:
            geodata["crs"] = tiff.crs
            transform = tiff.transform

            new_transform = rasterio.Affine(
                transform.a / rescale_factor,
                transform.b,
                transform.c,
                transform.d,
                transform.e / rescale_factor,
                transform.f,
            )

            geodata["transform"] = new_transform

    return (latitude_scale / h, longitude_scale / w), image_array, h, w, geodata


def save_image(input_file, output_name, image_data, geodata, output_folder):
    with rasterio.open(
        generate_filename(input_file, output_name, "tif", output_folder),
        "w",
        **geodata,
    ) as tiff:
        tiff.write(image_data[None, :, :])


def calculate_direction_masks(directions, h, w, plot=False):
    y = np.arange(-h, h)[::-1, None]
    x = np.arange(-w, w)[None, :]
    r = np.sqrt(x**2 + y**2)
    sin = y / r
    cos = x / r
    rad = np.arctan2(sin, cos)
    deg = np.rad2deg(rad)
    deg[sin < 0] += 360

    direction_masks = []
    angle = 360 / len(directions)
    for i in range(len(directions)):
        b = i + 1 == len(directions)
        mask = np.where(
            (deg < ((i + 1) * angle + angle / 2)) & (deg >= ((i) * angle + angle / 2))
            | (b & (deg < angle / 2)),
            np.ones(deg.shape),
            np.zeros(deg.shape),
        )
        direction_masks.append(mask)

        if plot:
            plt.imshow(mask.get() if backend != "cpu" else mask)
            plt.title(directions[i])
            plt.colorbar()
            plt.show()

    return direction_masks


def calculate_distance_falloff(h, w, latitude_scale, longitude_scale, plot=False):
    distance = np.sqrt(
        (np.arange(-h, h)[:, None] * latitude_scale) ** 2
        + (np.arange(-w, w)[None, :] * longitude_scale) ** 2
    )  # distance to every point
    if plot:
        plt.imshow(distance.get() if backend != "cpu" else distance)
        plt.title("Distance")
        plt.colorbar()
        plt.show()
    return distance


def calculate_elevation(
    directions,
    scales: tuple[float, float, float],
    image_array,
    h,
    w,
    direction_masks,
    distance,
):
    max_angles_tan = np.zeros(
        (len(directions), image_array.shape[0], image_array.shape[1])
    )
    max_tans = np.zeros(image_array.shape)
    max_angle_directions = np.zeros(image_array.shape)

    start = time()
    for row in range(image_array.shape[0]):
        for col in range(image_array.shape[1]):
            if col == image_array.shape[1] - 1:
                print(
                    round(
                        ((row) * image_array.shape[1] + (col + 1))
                        / (image_array.shape[0] * image_array.shape[1])
                        * 100,
                        1,
                    ),
                    "%",
                )

            height = image_array - image_array[row, col]  # height at every point

            tans = (
                height
                / distance[
                    h - row : h + image_array.shape[0] - row,
                    w - col : w + image_array.shape[1] - col,
                ]
            )  # angle to every point
            tans[row, col] = 0

            max_tan = -1
            max_tan_index = -1
            for k in range(len(directions)):

                max_tan_dir = (
                    tans
                    * direction_masks[k][
                        h - row : h + image_array.shape[0] - row,
                        w - col : w + image_array.shape[1] - col,
                    ]
                ).max()

                max_angles_tan[k, row, col] = max_tan_dir

                if max_tan_dir > max_tan:
                    max_tan = max_tan_dir
                    max_tan_index = k
            max_tans[row, col] = max_tan
            max_angle_directions[row, col] = max_tan_index

    max_angles_degrees_directional = np.rad2deg(np.arctan(max_angles_tan * scales[0]))
    max_angles = np.rad2deg(np.arctan(max_tans * scales[0]))
    print(round(time() - start, 3), "s")

    return max_angles_degrees_directional, max_angles, max_angle_directions


def calculate_aspect(slopes, w, h):
    dz_dy, dz_dx = np.gradient(slopes, h, w)
    aspect_rad = np.arctan2(-dz_dx, dz_dy)
    aspect_deg = np.degrees(aspect_rad)
    aspect_deg = (aspect_deg + 360) % 360
    slope = np.sqrt(dz_dx**2 + dz_dy**2)
    aspect_deg[slope == 0] = -1  # flat areas
    return aspect_deg


def output(
    filename,
    directions,
    cmap,
    elevation,
    max_angles,
    max_dirs,
    aspect,
    geodata,
    output_folder,
    plot=True,
):
    for k, dir in enumerate(directions):
        elevation_image = elevation.get()[k] if backend != "cpu" else elevation[k]
        plt.imshow(elevation_image, cmap=cmap)
        plt.title(directions[k])
        plt.colorbar()
        if plot:
            plt.show()
        if output_folder:
            plt.imsave(
                generate_filename(filename, dir, "png", output_folder),
                elevation_image,
                cmap=cmap,
            )
            save_image(filename, dir, elevation_image, geodata, output_folder)

    max_angles_image = max_angles.get() if backend != "cpu" else max_angles
    plt.imshow(max_angles_image, cmap=cmap)
    plt.title("Worst")
    plt.colorbar()
    if plot:
        plt.show()
    if output_folder:
        plt.imsave(
            generate_filename(filename, "Worst", "png", output_folder),
            max_angles_image,
            cmap=cmap,
        )
        save_image(filename, "Worst", max_angles_image, geodata, output_folder)

    max_dirs_image = max_dirs.get() if backend != "cpu" else max_dirs
    plt.imshow(max_dirs_image)
    plt.title("Quantized Aspect")
    plt.colorbar()
    if plot:
        plt.show()
    if output_folder:
        plt.imsave(
            generate_filename(filename, "Quantized_Aspect", "png", output_folder),
            max_dirs_image,
            cmap=cmap,
        )
        save_image(filename, "Quantized_Aspect", max_dirs_image, geodata, output_folder)

    aspect_image = aspect.get() if backend != "cpu" else aspect
    plt.imshow(aspect_image)
    plt.title("Aspect")
    plt.colorbar()
    if plot:
        plt.show()
    if output_folder:
        plt.imsave(
            generate_filename(filename, "Aspect", "png", output_folder),
            aspect_image,
            cmap=cmap,
        )
        save_image(filename, "Aspect", aspect_image, geodata, output_folder)


def main(
    image_path,
    vertical_scale,
    latitude_scale,
    longitude_scale,
    rescale_factor,
    output_folder,
    directions,
    plot=True,
):
    (y_scale, x_scale), image_array, h, w, geodata = load_image(
        image_path,
        latitude_scale,
        longitude_scale,
        rescale_factor,
    )

    direction_masks = calculate_direction_masks(directions, h, w)

    distance = calculate_distance_falloff(h, w, latitude_scale, longitude_scale)

    cmap = plt.get_cmap("RdYlGn").reversed()

    elevation, max_angles, max_dirs = calculate_elevation(
        directions,
        (vertical_scale, y_scale, x_scale),
        image_array,
        h,
        w,
        direction_masks,
        distance,
    )

    aspect = calculate_aspect(max_angles, w, h)

    output(
        image_path,
        directions,
        cmap,
        elevation,
        max_angles,
        max_dirs,
        aspect,
        geodata,
        output_folder,
        plot,
    )


if __name__ == "__main__":
    if not image_path is None:
        main(
            image_path,
            vertical_scale,
            latitude_scale,
            longitude_scale,
            resolution_rescale,
            output_folder,
            directions,
            plot,
        )
    else:
        for tile in tiles:
            print(tile)
            main(
                tile,
                vertical_scale,
                latitude_scale,
                longitude_scale,
                resolution_rescale,
                output_folder,
                directions,
                plot,
            )
            print()
