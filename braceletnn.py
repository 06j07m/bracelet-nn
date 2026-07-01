import json
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torcheval.metrics import MulticlassAccuracy, MulticlassRecall, MulticlassPrecision, MulticlassF1Score
from tqdm import tqdm

from utils import get_device, load_checkpoint, save_checkpoint
from KData import KData
from KCModel import KCModel

device = get_device()

def validate(model, val_data, loss_func):
    model.eval()
    running_loss = 0.0

    with torch.no_grad():
        for i, (X, Y) in enumerate(tqdm(val_data), 0):
            X = X.to(device)
            Y = Y.to(device)

            logits = model(X)

            loss = loss_func(logits, Y)
            running_loss += loss.item()

    average_loss = running_loss / len(val_data)

    metrics = {
        "avg_loss": average_loss
    }

    return metrics


def test(model, test_data, loss_func):
    model.eval()
    running_loss = 0.0

    accuracy_metric = MulticlassAccuracy(num_classes=4, average=None, device=device)
    precision_metric = MulticlassPrecision(num_classes=4, average=None, device=device)
    recall_metric = MulticlassRecall(num_classes=4, average=None, device=device)
    f1_metric = MulticlassF1Score(num_classes=4, average=None, device=device)

    with torch.no_grad():
        for i, (X, Y) in enumerate(tqdm(test_data), 0):
            X = X.to(device)
            Y = Y.to(device)

            logits = model(X)

            loss = loss_func(logits, Y)
            running_loss += loss.item()
            
            logits = torch.argmax(logits, dim=1)
            accuracy_metric.update(logits, Y)
            precision_metric.update(logits, Y)
            recall_metric.update(logits, Y)
            f1_metric.update(logits, Y)

    average_loss = running_loss / len(test_data)
    accuracy = accuracy_metric.compute()
    precision = precision_metric.compute()
    recall = recall_metric.compute()
    f1 = f1_metric.compute()

    metrics = {
        "avg_loss": average_loss,
        "accuracy": accuracy.tolist(),
        "precision": precision.tolist(),
        "recall": recall.tolist(),
        "f1": f1.tolist()
    }

    accuracy_metric.reset()
    precision_metric.reset()
    recall_metric.reset()
    f1_metric.reset()

    return metrics


def train(model, train_data, val_data, test_data, 
          loss_func, optimizer, scheduler,
          num_epochs=10, checkpoint_dir="."):

    start_epoch = 0
    best_loss = float('inf')

    # Load checkpoint if exists or create checkpoint directory
    print("Checking for saved checkpoints...")
    checkpoint, metrics = load_checkpoint(checkpoint_dir)

    if checkpoint:
        print("Loading last checkpoint...")
        start_epoch = checkpoint['epoch'] + 1
        model.load_state_dict(checkpoint['model_state'])
        optimizer.load_state_dict(checkpoint['optimizer_state'])
        scheduler.load_state_dict(checkpoint['scheduler_state'])
    if metrics:
        print("Loading last metrics...")
        best_loss = metrics['avg_loss']

    print(f"Starting training from epoch {start_epoch}...")
    for epoch in range(start_epoch, num_epochs):  # loop over the dataset multiple times
        model.train()
        running_loss = 0.0

        print(f"Training epoch {epoch}...")
        for i, data in enumerate(tqdm(train_data), 0):
            X, Y = data[0].to(device), data[1].to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(X)
            loss = loss_func(outputs, Y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        # print statistics
        print(f'Training loss: {running_loss / len(train_data):.3f}')

        # validate
        print(f"Validating after epoch {epoch}...")
        metrics = validate(model, val_data, loss_func)
        print(f'\tValidation loss: {metrics["avg_loss"]:.3f}')

        # save checkpoint
        checkpoint = {
            'epoch': epoch,
            'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'scheduler_state': scheduler.state_dict()
        }
        print("Saving checkpoint...")
        save_checkpoint(checkpoint_dir, checkpoint, metrics)

        # if best model so far, save best model
        if metrics["avg_loss"] < best_loss:
            print("Saving best model...")
            best_loss = metrics["avg_loss"]
            torch.save(checkpoint, os.path.join(checkpoint_dir, "best.pth"))

        running_loss = 0.0

    # test model
    print("Testing model...")
    test_metrics = test(model, test_data, loss_func)

    # print and save test results
    with open(os.path.join(checkpoint_dir, "test_metrics.txt"), "w") as f:
        json.dump(test_metrics, f, indent=4)
    print("Test set results:")
    print(test_metrics)


if __name__ == "__main__":
    # load datasets
    print("Loading datasets...")
    train_dataset = KData(labels_file=os.path.join("data", "train_labels.csv"), 
                    img_dir=os.path.join("data", "cropped"))
    val_dataset = KData(labels_file=os.path.join("data", "val_labels.csv"), 
                    img_dir=os.path.join("data", "cropped"))
    test_dataset = KData(labels_file=os.path.join("data", "test_labels.csv"), 
                    img_dir=os.path.join("data", "cropped"))

    # hyperparameters & train settings
    batch_size = 32
    num_epochs = 10
    checkpoint_dir = os.path.join("runs", "TEST1") # CHANGE TO NEW DIRECTORY FOR EACH TRAINING RUN

    # create data loaders
    print("Creating data loaders...")
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
 
    # load model and move to device
    print(f"Using device {device}...")
    model = KCModel()
    model.to(device)

    # # define loss function, optimizer, and scheduler
    loss_func = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

    # train
    train(model, train_loader, val_loader, test_loader, 
          loss_func, optimizer, scheduler,
          num_epochs=num_epochs, checkpoint_dir=checkpoint_dir)
