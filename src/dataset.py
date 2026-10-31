import wfdb
import numpy as np
import torch
from torch.utils.data import Dataset

# From list of records return (Record_i: Signal, array[peaks], array[labels])
def DS_creat(records):
    mapping = {"N": "N", "L": "N", "R": "N", "e": "N", "j": "N",
                "A": "S", "a": "S", "J": "S", "S": "S",
                "V": "V", "E": "V",
                "F": "F"}
    DataSet = {}
    for rec_id in records:
        record = wfdb.rdrecord(f'data/mit-bih-arrhythmia-database-1.0.0/{rec_id}')
        annotation = wfdb.rdann(f'data/mit-bih-arrhythmia-database-1.0.0/{rec_id}', 'atr')
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

class BeatDataSet(Dataset):
    def __init__(self, signals, labels, records, label_map, augment=None, feats=None):
        self.signals = np.asarray(signals, dtype=np.float32)
        self.labels = np.array([label_map[i] for i in labels], dtype=np.int64)
        self.records = np.asarray(records)
        self.label_map = label_map
        self.augment = augment
        self.feats = None if feats is None else np.asarray(feats, dtype=np.float32)


    def __len__(self):
        return len(self.signals)

    def __getitem__(self, i):
        signal = self.signals[i]
        if self.augment is not None:
            signal = self.augment(signal.copy())
        sig = torch.as_tensor(signal, dtype=torch.float32).unsqueeze(0)                # (1, 234)
        if self.feats is not None:
            f = torch.as_tensor(self.feats[i]).unsqueeze(1).expand(-1, sig.shape[1])  # (5, 234)
            sig = torch.cat([sig, f], dim=0)                                           # (6, 234)
        lab = torch.tensor(self.labels[i], dtype=torch.long)
        return (sig, lab)