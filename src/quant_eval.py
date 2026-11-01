import os
import sys
sys.path.append(os.path.expanduser("~/ai8x/ai8x-training"))
import ai8x

import torch
from sklearn.metrics import classification_report, confusion_matrix

from evaluate import eval_model
from model import ECGmodel
from prepare_data import make_dataloaders

device = "cuda" if torch.cuda.is_available() else "cpu"
QAT_POLICY = {"start_epoch": 0, "weight_bits": 8}
SEED = 1

_, _, test_dl, _ = make_dataloaders("data", 64, SEED, False)

ai8x.set_device(87, simulate=True, round_avg=False)   # before building the model

model = ECGmodel(num_channels=7)
ai8x.fuse_bn_layers(model)
ai8x.initiate_qat(model, QAT_POLICY)
ckpt = torch.load("results/ecg-q.pth.tar", map_location="cpu")
missing, unexpected = model.load_state_dict(ckpt["state_dict"], strict=False)
print("missing keys:", missing)
print("unexpected keys:", unexpected)
ai8x.update_model(model)
model.to(device)

y_pred, y_true = eval_model(model, test_dl, device, int8_input=True)
print("TEST (int8, hardware simulation)")
print(classification_report(y_true, y_pred, labels=[0, 1, 2, 3], target_names=["N", "S", "V", "F"], zero_division=0))
print(confusion_matrix(y_true, y_pred, labels=[0, 1, 2, 3]))

import pandas as pd

classes = ["N", "S", "V", "F"]
pd.DataFrame({
    "record": test_dl.dataset.records,
    "true_class": [classes[i] for i in y_true],
    "pred_class": [classes[i] for i in y_pred],
}).to_csv("results/int8_test_predictions.csv", index=False)
print("Saved results/int8_test_predictions.csv")