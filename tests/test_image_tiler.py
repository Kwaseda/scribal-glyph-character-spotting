# test pad_image, tile_image on a single small image
import os, cv2
import numpy as np

from scribal_char_spotting.tiling import (
    pad_image,
    tile_image,
    get_tile_coords,
    save_tiles,
)
import scribal_char_spotting.config as cfg

TEST_IMAGE_PATH = os.path.join(cfg.IMAGE_PATH, "WdB_027-0002.jpg")

test_image = cv2.imread(TEST_IMAGE_PATH)

h_test, w_test, c_test = test_image.shape
print(f"height:", h_test, "width:", w_test, "color_chan", c_test)

""" Testing if padding works properly """
n_tiles_x = int(np.ceil(w_test / cfg.STRIDE))  # horizontal
n_tiles_y = int(np.ceil(h_test / cfg.STRIDE))  # vertical

target_width = (n_tiles_x - 1) * cfg.STRIDE + cfg.TILE_SIZE
target_height = (n_tiles_y - 1) * cfg.STRIDE + cfg.TILE_SIZE

"""unpadded = cv2.imshow("Unpadded image", test_image)
cv2.waitKey(0)
cv2.destroyAllWindows()"""

# unpadded = Image.fromarray(test_image).show()

padded_image = pad_image(test_image, target_width, target_height)
# padded = Image.fromarray(padded_image).show()

print(padded_image.shape)


""" Testing to make sure tile coordinate calculation is correct"""

tile_coordinates = get_tile_coords(padded_image, cfg.TILE_SIZE, cfg.OVERLAP)

print(len(tile_coordinates))
print(
    f"First 5 tile coords:",
    tile_coordinates[:5],
)
print(f"Last 5 tile coords:", tile_coordinates[-5:])

for tile_coord in tile_coordinates:
    h, w = tile_coord[:2]
    """ if h > target_height or w > target_width:
        print("Image coordinates > image_dimension")
    else:
        print("All image coordinates within set image dimensions")
 """


""" Tiling the images """

tiled_images = tile_image(padded_image, cfg.TILE_SIZE, cfg.OVERLAP)

print(len(tiled_images))

if len(tiled_images) != len(tile_coordinates):
    print("No, number of tiles don't match the provided no. of coordinates")
else:
    print("Yes, number of tile match the provided no. of coordinates")

# Verify the dimensions of the first tile
first_tile = tiled_images[0].shape
print(first_tile == (cfg.TILE_SIZE, cfg.TILE_SIZE, 3))


""" Save the tiles to disk """

num_images = len(sorted([f for f in os.listdir(cfg.IMAGE_PATH) if f.endswith(".jpg")]))

tiled_images = tile_image(padded_image, cfg.TILE_SIZE, cfg.OVERLAP)

save_tiles(tiled_images, num_images, cfg.TILE_STORAGE_PATH)

# Check number of saved files == tiled_images

lst = os.listdir(cfg.TILE_STORAGE_PATH)  # your directory path
print(len(lst))
