import torch
import numpy as np
from tqdm import trange


def train_one_epoch(model, optimizer, loss_function, train_loader, device):
    model.train()
    running_loss = 0.

    for feature, label in train_loader:         
        feature = feature.to(device)
        label = label.to(device)

        optimizer.zero_grad()
        loss = loss_function(model(feature), label)
        running_loss += loss.item()
        loss.backward()
        optimizer.step()

    return running_loss / len(train_loader)


def validate_model(model, loss_function, val_loader, device):
    model.eval()
    running_loss = 0.

    with torch.no_grad():
        for feature, label in val_loader:
            feature = feature.to(device)
            label = label.to(device)

            running_loss += loss_function(model(feature), label).item()

    return running_loss / len(val_loader)


def run_training(model, optimizer, loss_function, train_loader, val_loader, device, epochs=1):
    losses = []
    val_losses = []
    for i in trange(epochs):
        losses.append(train_one_epoch(model, optimizer, loss_function, train_loader, device))
        val_losses.append(validate_model(model, loss_function, val_loader, device))

    return np.array(losses), np.array(val_losses)


def test_model(model, loss_function, test_loader, device):
    model.eval()
    running_loss = 0.

    preds, trues = [], []
    with torch.no_grad():
        for feature, label in test_loader:
            feature = feature.to(device)
            label = label.to(device)

            pred = model(feature)
            preds.append(pred.numpy())
            trues.append(label.numpy())

            running_loss += loss_function(pred, label).item()
    
    preds = np.concatenate(preds)
    trues = np.concatenate(trues)

    return preds, trues, running_loss / len(test_loader)