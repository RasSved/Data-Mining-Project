import os
import sys
sys.path.append(os.path.expanduser("~/ai8x/ai8x-training"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import wfdb
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from torch.utils.data import DataLoader

from dataset import BeatDataSet
from evaluate import eval_model
from model import ECGmodel
from prepare_data import normalize_beat, preprocessing, window_slicing

CLASS_MAP = {"N": 0, "S": 1, "V": 2, "F": 3}
device = "cuda" if torch.cuda.is_available() else "cpu"


# Copy of DS_creat from dataset.py, with the folder as a parameter
def noise_DS_creat(records):
    mapping = {"N": "N", "L": "N", "R": "N", "e": "N", "j": "N",
                "A": "S", "a": "S", "J": "S", "S": "S",
                "V": "V", "E": "V",
                "F": "F"}
    DataSet = {}
    for rec_id in records:
        record = wfdb.rdrecord(f'data/mit-bih-noise-stress-test-database-1.0.0/{rec_id}')
        annotation = wfdb.rdann(f'data/mit-bih-noise-stress-test-database-1.0.0/{rec_id}', 'atr')
        peaks = np.array(annotation.sample)
        labels = np.array(annotation.symbol) # [a, s, y, t, #, !, -, +] # [True, False, True]
        mask = []
        for label in labels:
            if label in mapping:
                mask.append(True)
            else:
                mask.append(False) 

        if "MLII" not in record.sig_name:
            raise ValueError(f"Record {rec_id} has no MLII channel ({record.sig_name})")

        MLII_set = record.sig_name.index("MLII")
        MLII_chanel = (record.p_signal[:, MLII_set])
        DS_peaks = peaks[mask]
        DS_labels = np.array([mapping.get(label) for label in labels[mask]])

        DataSet[rec_id] = MLII_chanel, DS_peaks, DS_labels 
    return DataSet


# Load the stored weights once
model = ECGmodel(num_channels=7)
model.load_state_dict(torch.load("results/best_fp32.pth", map_location="cpu"))
model.to(device)

CONDITIONS = [("12 dB", "data/mit-bih-noise-stress-test-database-1.0.0", "e12"), 
              ("6 dB", "data/mit-bih-noise-stress-test-database-1.0.0", "e06"), 
              ("0 dB", "data/mit-bih-noise-stress-test-database-1.0.0", "e00")]
results = []
for name, folder, suffix in CONDITIONS:
    # Build the test set for this noise level
    DataSet = noise_DS_creat([f"118{suffix}", f"119{suffix}"])
    DataSet = preprocessing(DataSet)
    windows, labels, records, feats = window_slicing(DataSet)
    test_ds = BeatDataSet(windows, labels, records, CLASS_MAP, augment=normalize_beat, feats=feats)
    test_dl = DataLoader(test_ds, batch_size=64, shuffle=False, num_workers=0)

    # Evaluate
    y_pred, y_true = eval_model(model, test_dl, device)
    print("TEST", name)
    print(classification_report(y_true, y_pred, labels=[0, 1, 2, 3], target_names=["N", "S", "V", "F"], zero_division=0))
    print(confusion_matrix(y_true, y_pred, labels=[0, 1, 2, 3]))

    import pandas as pd
    classes = ["N", "S", "V", "F"]
    fname = f"results/noise_predictions_{name.replace(' ', '')}.csv"
    pd.DataFrame({
        "record": test_dl.dataset.records,
        "true_class": [classes[i] for i in y_true],
        "pred_class": [classes[i] for i in y_pred],
    }).to_csv(fname, index=False)
    print("Saved", fname)
