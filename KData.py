import os
import pandas as pd
from torch.utils.data import Dataset
from torchvision.io import decode_image
import torchvision.transforms.v2 as transforms

class KData(Dataset):
    def __init__(self, labels_file, img_dir):
        self.img_labels = pd.read_csv(labels_file)
        self.img_dir = img_dir
        
    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, index):
        img_path = os.path.join(self.img_dir, str(self.img_labels.iloc[index, 0]))
        image = decode_image(img_path)
        image = transforms.functional.to_dtype(image) # convert to float32 >:|
        label = self.img_labels.iloc[index, 1]
        return image, label

if __name__ == "__main__":
    train_dataset = KData(labels_file=os.path.join("data", "train_labels.csv"), 
                    img_dir=os.path.join("data", "cropped"))
    val_dataset = KData(labels_file=os.path.join("data", "val_labels.csv"), 
                    img_dir=os.path.join("data", "cropped"))
    test_dataset = KData(labels_file=os.path.join("data", "test_labels.csv"), 
                    img_dir=os.path.join("data", "cropped"))

    print(f"Train dataset length: {len(train_dataset)}")
    print(f"Validation dataset length: {len(val_dataset)}")
    print(f"Test dataset length: {len(test_dataset)}")