import cv2
import os
import numpy as np

from utils import MyError

def process_image(filename):
    if not os.path.exists(filename):
        raise MyError("file not found")

    # read image
    img = cv2.imread(filename, 0)

    # get rows and cols
    name = os.path.split(filename)[1]
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
    dirpath = os.path.join("data", "cropped", number)
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)

    for row in range(int(rows)):
        # even iterator -> odd row
        if row % 2 == 0:
            y = odd_start_y + (row // 2) * (row_height + row_gap)
            for col in range(int(cols)):
                x = odd_start_x + col * (col_width + col_gap)
                crop_img = img[y:y+row_height, x:x+col_width]
                cv2.imwrite(f"data/cropped/{number}/{row}_{col}.png", crop_img)

        # odd iterator -> even row
        else:
            y = even_start_y + ((row-1) // 2) * (row_height + row_gap)
            for col in range(int(cols)):
                x = even_start_x + col * (col_width + col_gap)
                if x + col_width > width:
                    continue
                crop_img = img[y:y+row_height, x:x+col_width]
                cv2.imwrite(f"data/cropped/{number}/{row}_{col}.png", crop_img)

if __name__ == "__main__":
    process_image('data/original/209936-2-16.png')