from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from time import time

image_path = 's.png'#"Bulgaria_elevation_360dpi.tif"
height_max_meters = 2925
height_max_pixels = 80
distance_max_meters = 500_000
distance_max_pixels = 336
save=True

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


def load_image(
    image_path,
    height_max_meters,
    height_max_pixels,
    distance_max_meters,
    distance_max_pixels,
    plot=False
):
    vscale = height_max_meters / height_max_pixels
    hscale = distance_max_meters / distance_max_pixels

    scale = vscale / hscale

    image = Image.open(image_path)
    image_array = np.array(image, dtype=np.float32)
    print("Image array shape:", image_array.shape)

    if plot:
        plt.imshow(image_array)
        plt.title("Height")
        plt.colorbar()
        plt.show()

    if len(image_array.shape) > 2:
        image_array= image_array[:,:,0]
    h, w = image_array.shape
    return scale, image_array, h, w


scale, image_array, h, w = load_image(
    image_path,
    height_max_meters,
    height_max_pixels,
    distance_max_meters,
    distance_max_pixels,
)


def calculate_direction_masks(directions, h, w, plot=False):
    y = np.arange(-h, h)[::-1, None]
    x = np.arange(-w, w)[None, :]
    r = np.sqrt(x**2 + y**2)
    sin = y / r
    cos = x / r
    rad = np.atan2(sin, cos)
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
            plt.imshow(mask)
            plt.title(directions[i])
            plt.colorbar()
            plt.show()

    return direction_masks


direction_masks = calculate_direction_masks(directions, h, w)


def calculate_distance_falloff(h, w, plot=False):
    distance = np.sqrt(
        np.arange(-h, h)[:, None] ** 2 + np.arange(-w, w)[None, :] ** 2
    )  # distance to every point
    if plot:
        plt.imshow(distance)
        plt.title("Distance")
        plt.colorbar()
        plt.show()
    return distance


distance = calculate_distance_falloff(h, w)

cmap = plt.get_cmap("RdYlGn").reversed()


def calculate_elevation(
    directions, scale, image_array, h, w, direction_masks, distance
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

    max_angles_degrees_directional = np.rad2deg(np.atan(max_angles_tan * scale))
    max_angles = np.rad2deg(np.atan(max_tans * scale))
    print(round(time() - start, 3), "s")

    return max_angles_degrees_directional, max_angles, max_angle_directions


elevation, max_angles, max_dirs = calculate_elevation(
    directions, scale, image_array, h, w, direction_masks, distance
)


def calculate_aspect(slopes, w, h):
    dz_dy, dz_dx = np.gradient(slopes, h, w)
    aspect_rad = np.arctan2(-dz_dx, dz_dy)
    aspect_deg = np.degrees(aspect_rad)
    aspect_deg = (aspect_deg + 360) % 360
    slope = np.sqrt(dz_dx**2 + dz_dy**2)
    aspect_deg[slope == 0] = -1  # flat areas
    return aspect_deg


aspect = calculate_aspect(max_angles, w, h)

def output(directions, cmap, elevation, max_angles, max_dirs, aspect,save=True,plot=True):
    for k, dir in enumerate(directions):
        plt.imshow(elevation[k], cmap=cmap)
        plt.title(directions[k])
        plt.colorbar()
        if plot:
            plt.show()
        if save:
            plt.imsave(dir+'.png',elevation[k], cmap=cmap)

    plt.imshow(max_angles, cmap=cmap)
    plt.title("Worst")
    plt.colorbar()
    if plot:
        plt.show()
    if save:
        plt.imsave('Worst.png',elevation[k], cmap=cmap)

    plt.imshow(max_dirs)
    plt.title("Quantized Aspect")
    plt.colorbar()
    if plot:
        plt.show()
    if save:
        plt.imsave('Quantized Aspect.png',elevation[k], cmap=cmap)

    plt.imshow(aspect)
    plt.title("Aspect")
    plt.colorbar()
    if plot:
        plt.show()
    if save:
        plt.imsave('Aspect.png',elevation[k], cmap=cmap)

output(directions, cmap, elevation, max_angles, max_dirs, aspect,save)

# for i, res in enumerate(result):
#     plt.imshow(res, cmap=cmap)
#     plt.title(dirs[i])
#     plt.colorbar()
#     plt.show()
#     plt.imsave(dirs[i]+'.png',res,cmap=cmap)


# plt.imshow(worst,cmap=cmap)
# plt.title("Total")
# plt.colorbar()
# plt.show()
# plt.imsave('Worst.png',worst,cmap=cmap)
# plt.imsave('Worst_greyscale.png',worst,cmap=plt.get_cmap("Greys").reversed())

# aspect1 = calculate_aspect(worst,image_array.shape[1],image_array.shape[0])
# plt.imshow(max_angles_degrees, cmap=cmap)
# # plt.title("Aspect")
# plt.colorbar()
# plt.show()
# plt.imsave('Aspect.png',aspect1)
