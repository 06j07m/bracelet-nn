import cv2
import os
import numpy as np

from utils import MyError

# IMPORTANT: ONLY WORKS FOR EVEN NUMBER OF STRINGS OR
# IF THE ODD STRING IS ON THE RIGHT OF THE PATTERN
# I.E. FIRST KNOT IN FIRST ROW IS ON THE OUTSIDE LEFT

def process_image(filename, starting_index=0):
    if not os.path.exists(filename):
        raise MyError("file not found")

    # read image
    img = cv2.imread(filename, 0)

    # get rows and cols
    name = os.path.basename(filename)
    number, cols, rows = name.removesuffix(".png").split("-")
    height, width = img.shape

    col_width = 56
    row_height = 56

    col_gap = 20
    row_gap = 20

    odd_start_y = 27
    odd_start_x = 35
    even_start_y = 65
    even_start_x = 73

    # make folder for images
    dirpath = os.path.join("data", "cropped")
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)

    index = starting_index

    for row in range(int(rows)):
        # even iterator -> odd row
        if row % 2 == 0:
            y = odd_start_y + (row // 2) * (row_height + row_gap)
            for col in range(int(cols)):
                x = odd_start_x + col * (col_width + col_gap)
                crop_img = img[y:y+row_height, x:x+col_width]
                filename = os.path.join(dirpath, f"{index}.png")
                cv2.imwrite(filename, crop_img)
                index += 1

        # odd iterator -> even row
        else:
            y = even_start_y + ((row-1) // 2) * (row_height + row_gap)
            for col in range(int(cols)):
                x = even_start_x + col * (col_width + col_gap)
                # check if there is one less knot
                if x + col_width > width:
                    continue
                crop_img = img[y:y+row_height, x:x+col_width]
                filename = os.path.join(dirpath, f"{index}.png")
                cv2.imwrite(filename, crop_img)
                index += 1

    return index

if __name__ == "__main__":
    i = 0
    for file in os.listdir("data/original"):
        if file.endswith(".png"):
            i = process_image(os.path.join("data", "original", file), starting_index=i)