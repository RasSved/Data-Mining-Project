from sklearn.metrics import classification_report, confusion_matrix
import torch 
import numpy as np
from evaluate import eval_model, scorer
import copy
import os

from model import ECGmodel
from prepare_data import class_weights, load_split, make_dataloaders


def train_step(model, loader, criterion, optimizer, device):
    model.train()

    total_loss = 0
    total_bsize = 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.size(0)
        total_bsize += x.size(0)
    return total_loss / total_bsize

def train_loop(model, train_loader, val_loader, criterion, optimizer, max_epochs, stopper, device):
    prev_macro = -1
    stop_flag = 1
    best_poch = 0
    train_acc = []
    val_acc = []
    val_macro = []

    for epoch in range(1, max_epochs + 1):
        train_loss = train_step(model, train_loader, criterion, optimizer, device)
        train_acc.append(train_loss)
        if epoch % 1 == 0:
            pred, actual = eval_model(model, val_loader, device)
            val_score = scorer(actual, pred)
            val_macro.append(val_score.get("macro_f1"))
            val_acc.append(val_score.get("accuracy"))
            print("Val macro_f1: ", val_score.get("macro_f1"), "Val accuracy: ", val_score.get("accuracy"))
            if val_score.get("macro_f1") > prev_macro:
                prev_macro = val_score.get("macro_f1")  
                best_poch = epoch
                best_weights = copy.deepcopy(model.state_dict())
                stop_flag = 1
            elif stop_flag == stopper:
                break
            else:
                stop_flag += 1
        print("Train Loss: ", train_loss, "Epoch: ", epoch)
    
    model.load_state_dict(best_weights)
    
    return {"train_loss": train_acc, "val_accuracy": val_acc, "val_macro_f1": val_macro, "best_epoch": best_poch, "best_macro_f1": prev_macro}

if __name__ == "__main__":
    import json
    import random

    SEED = 0
    USE_SAMPLER = False
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_dl, val_dl, test_dl, clip = make_dataloaders("data", 64, SEED, USE_SAMPLER)
    print("clip value:", clip)

    model = ECGmodel().to(device)

    if USE_SAMPLER:
        criterion = torch.nn.CrossEntropyLoss()
    else:
        _, train_labels, _ = load_split("data", "train")
        weights = torch.tensor(class_weights(train_labels), dtype=torch.float32).to(device)
        criterion = torch.nn.CrossEntropyLoss(weight=weights)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    history = train_loop(model, train_dl, val_dl, criterion, optimizer, max_epochs=50, stopper=5, device=device)

    y_pred, y_true = eval_model(model, val_dl, device)
    print(classification_report(y_true, y_pred, labels=[0, 1, 2, 3],
                            target_names=["N", "S", "V", "F"], zero_division=0))
    print(confusion_matrix(y_true, y_pred, labels=[0, 1, 2, 3]))

    os.makedirs("results", exist_ok=True)
    torch.save(model.state_dict(), "results/best_fp32.pth")
    with open("results/history_fp32.json", "w") as f:
        json.dump({**history, "clip_value": clip, "seed": SEED,
                   "use_sampler": USE_SAMPLER}, f, indent=2)