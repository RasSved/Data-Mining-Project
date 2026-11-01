import json
import os
import random
import sys
from types import SimpleNamespace
sys.path.append(os.path.expanduser("~/ai8x/ai8x-training"))
import ai8x

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import torch

from evaluate import eval_model
from model import ECGmodel
from prepare_data import class_weights, load_split, make_dataloaders
from train import train_loop
SEED = 1
USE_SAMPLER = False

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
device = "cuda" if torch.cuda.is_available() else "cpu"


# COME GET UR BATCH SIZES HERE!
train_dl, val_dl, test_dl, clip = make_dataloaders("data", 64, SEED, USE_SAMPLER)


model = ECGmodel(num_channels=7).to(device)
model.load_state_dict(torch.load("results/best_fp32.pth", map_location="cpu"))
model.to(device)

QAT_POLICY = {"start_epoch": 0, "weight_bits": 8, "outlier_removal_z_score": 8.0}
ai8x.fuse_bn_layers(model)
args = SimpleNamespace(device=device, act_mode_8bit=False)
ai8x.pre_qat(model, train_dl, args, QAT_POLICY)
ai8x.initiate_qat(model, QAT_POLICY)
model.to(device)


if USE_SAMPLER:
    criterion = torch.nn.CrossEntropyLoss()
else:
    _, train_labels, _ = load_split("data", "train")
    weights = torch.tensor(class_weights(train_labels), dtype=torch.float32).to(device)
    criterion = torch.nn.CrossEntropyLoss(weight=weights)


# COME GET UR LEARNING RATE HERE!
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=1e-2)
history = train_loop(model, train_dl, val_dl, criterion, optimizer, max_epochs=200, stopper=5, device=device)


y_pred, y_true = eval_model(model, test_dl, device, int8_input=False)
print("TEST (QAT, float simulation)")
print(classification_report(y_true, y_pred, labels=[0, 1, 2, 3], target_names=["N", "S", "V", "F"], zero_division=0))
print(confusion_matrix(y_true, y_pred, labels=[0, 1, 2, 3]))
 

os.makedirs("results", exist_ok=True)
torch.save({
    "epoch": history["best_epoch"],
    "arch": "ecgmodel",
    "state_dict": model.state_dict(),
    "extras": {"best_macro_f1": float(history["best_macro_f1"]), "clip_value": float(clip)},
}, "results/qat_best.pth.tar")
with open("results/history_qat.json", "w") as f:
    json.dump(history, f, indent=2, default=float)
print("Saved results/qat_best.pth.tar")
