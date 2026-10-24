import torch 
import numpy as np


def train_step(model, loader, criterion, optimizer):
    model.train()

    total_loss = 0
    total_bsize = 0

    for x, y in loader:
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.size(0)
        total_bsize += x.size(0)
    return total_loss / total_bsize

def train_loop(model, train_loader, val_loader, criterion, optimizer, max_epochs, stopper):
    pass