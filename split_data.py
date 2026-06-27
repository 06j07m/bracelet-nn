import os
import pandas as pd
from sklearn.model_selection import train_test_split

def split_data(all_labels_file, train_percent=0.8):
    labels = pd.read_csv(all_labels_file)

    train, val_and_test = train_test_split(labels, 
                                           train_size=train_percent, 
                                           random_state=42, 
                                           stratify=labels['Label'])
    val, test = train_test_split(val_and_test,
                                    train_size=0.5, 
                                    random_state=42, 
                                    stratify=val_and_test['Label'])

    train.to_csv(os.path.join("data", "train_labels.csv"), index=False)
    val.to_csv(os.path.join("data", "val_labels.csv"), index=False)
    test.to_csv(os.path.join("data", "test_labels.csv"), index=False)


if __name__ == "__main__":
    split_data(os.path.join("data", "labels.csv"))