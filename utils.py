import torch
import os
import pandas as pd

class MyError(Exception):
    pass

def get_device():
    return torch.device(
        "cuda" if torch.cuda.is_available() 
        else "cpu"
    )

def load_checkpoint(checkpoint_dir: str):
    checkpoint = None
    metrics = None

    if os.path.exists(checkpoint_dir) and os.path.isdir(checkpoint_dir):
        # Find latest checkpoint in folder
        checkpoint_file = os.path.join(checkpoint_dir, "latest.pth")
        if os.path.exists(checkpoint_file):
            checkpoint = torch.load(checkpoint_file)

        # Get lastest metrics from csv
        metrics_file = os.path.join(checkpoint_dir, "metrics.csv")
        if os.path.exists(metrics_file):
            metrics_df = pd.read_csv(metrics_file)
            metrics = metrics_df.tail(1).to_dict(orient='index')[0]

    else:
        os.makedirs(checkpoint_dir)

    return checkpoint, metrics

def save_checkpoint(checkpoint_dir: str, checkpoint: dict, metrics: dict|None):
    checkpoint_file = os.path.join(checkpoint_dir, "latest.pth")
    torch.save(checkpoint, checkpoint_file)

    if metrics:
        metrics_file = os.path.join(checkpoint_dir, "metrics.csv")
        new = pd.DataFrame.from_records([metrics])
        if os.path.exists(metrics_file):
            metrics_df = pd.read_csv(metrics_file)
            metrics_df = pd.concat([metrics_df, new], ignore_index=True)
        else:
            metrics_df = new
        metrics_df.to_csv(metrics_file, index=False)