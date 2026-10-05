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
        labels = np.array(annotation.symbol) # [a, s, y, t, #, !, -, +]
        mask = []
        for label in labels:
            if label in mapping:
                mask.append(True)
            else:
                mask.append(False)

        MLII_set = record.sig_name.index("MLII")
        if MLII_set == None:
            ValueError("Record doesnt have the right chanel")
        
        MLII_chanel = (record.p_signal[:, MLII_set])
        DS_peaks = peaks[mask]
        DS_labels = np.array([mapping.get(label) for label in labels[mask]])

        DataSet[rec_id] = MLII_chanel, DS_peaks, DS_labels 
    return DataSet

class BeatDataSet(Dataset):
    def __init__(self, signals, labels, records, label_map, augment=None):
        self.signals = np.asarray(signals, dtype=np.float32)
        self.labels = np.array([label_map[i] for i in labels], dtype=np.int64)
        self.records = np.asarray(records)
        self.label_map = label_map
        self.augment = augment


    def __len__(self):
        return len(self.signals)

    def __getitem__(self, i):
        signal = self.signals[i]
        if self.augment is not None:
            signal = self.augment(signal.copy())
        sig = torch.as_tensor(signal, dtype=torch.float32).unsqueeze(0)
        lab = torch.tensor(self.labels[i], dtype=torch.long)
        return (sig, lab)