import cv2
import os
import pandas as pd

from utils import MyError

DATA_DIR = os.path.join(".", "data")

def label_images(image_dir: str, label_file: str):
    # dictionary to store labels
    labels = {"File": [], "Label": []}

    keys = {
        ord('0'): 0,
        ord('1'): 1,
        ord('2'): 2,
        ord('3'): 3
    }

    # iterate through directory
    for file in os.listdir(image_dir):
        if file.endswith(".png"):
            # get filename without directory
            name = os.path.basename(file)
            img = cv2.imread(os.path.join(image_dir, file))
            cv2.imshow("image", img)
            k = cv2.waitKey(0) & 0xFF

            if k in keys:
                labels["File"].append(name)
                labels["Label"].append(keys[k])
            else:
                pd.DataFrame.from_dict(labels).to_csv(label_file, index=False)
                raise MyError("Invalid key pressed. Please press 0, 1, 2, or 3.")

    # save labels to csv
    pd.DataFrame.from_dict(labels).to_csv(label_file, index=False)

if __name__ == "__main__":
    label_images(image_dir=os.path.join(DATA_DIR, "new"), 
                 label_file=os.path.join(DATA_DIR, "labels_new.csv"))