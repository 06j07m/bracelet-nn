import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
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
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
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
    checkpoint, metrics = load_checkpoint(checkpoint_dir)

    if checkpoint:
        start_epoch = checkpoint['epoch'] + 1
        model.load_state_dict(checkpoint['model_state'])
        optimizer.load_state_dict(checkpoint['optimizer_state'])
        scheduler.load_state_dict(checkpoint['scheduler_state'])
    if metrics:
        best_loss = metrics['avg_loss'][-1]

    for epoch in range(start_epoch, num_epochs):  # loop over the dataset multiple times
        model.train()
        running_loss = 0.0

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
        print(f'Epoch {epoch}, training loss: {running_loss / len(train_data):.3f}')

        # validate
        metrics = validate(model, val_data, loss_func)
        print(f'\tValidation loss: {metrics["avg_loss"]:.3f}')

        # save checkpoint
        checkpoint = {
            'epoch': epoch,
            'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'scheduler_state': scheduler.state_dict()
        }
        save_checkpoint(checkpoint_dir, checkpoint, metrics)

        # if best model so far, save best model
        if metrics["avg_loss"] < best_loss:
            best_loss = metrics["avg_loss"]
            torch.save(checkpoint, os.path.join(checkpoint_dir, "best.pth"))

        running_loss = 0.0

    # test model
    test_metrics = test(model, test_data, loss_func)

    # print and save test results as csv
    test_df = pd.DataFrame.from_dict(test_metrics, orient='columns')
    test_df.to_csv(os.path.join(checkpoint_dir, "test_metrics.csv"), index=False)
    print("Test set results:")
    print(test_df)


if __name__ == "__main__":
    # # load datasets
    # train_dataset = KData(labels_file=os.path.join("data", "train_labels.csv"), 
    #                 img_dir=os.path.join("data", "cropped"))
    # val_dataset = KData(labels_file=os.path.join("data", "val_labels.csv"), 
    #                 img_dir=os.path.join("data", "cropped"))
    # test_dataset = KData(labels_file=os.path.join("data", "test_labels.csv"), 
    #                 img_dir=os.path.join("data", "cropped"))

    # # hyperparameters & train settings
    # batch_size = 32
    # num_epochs = 10
    # checkpoint_dir = os.path.join("runs")

    # # create data loaders
    # train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    # val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    # test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # # load model and move to device
    # model = KCModel()
    # model.to(device)

    # # define loss function, optimizer, and scheduler
    # loss_func = nn.CrossEntropyLoss()
    # optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    # # TODO: scheuler

    # # TODO TRAIN
    outputs = torch.tensor([[0.1, 0.8, 0.05, 0.05]])
    target = torch.tensor([1])

    _, predicted = torch.max(outputs, 1)

    print(predicted.item())
