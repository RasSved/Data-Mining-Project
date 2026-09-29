import wfdb
import numpy as np

DS1_rec = [101, 106, 108, 109, 112, 114, 115, 116, 118, 119, 122, 124, 
       201, 203, 205, 207, 208, 209, 215, 220, 223, 230 ]
DS2_rec = [100, 103, 105, 111, 113, 117, 121, 123, 200, 202, 210, 212, 
       213, 214, 219, 221, 222, 228, 231, 232, 233, 234 ]

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
        labels = np.array(annotation.symbol)
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

# From dict to (Signal, Label, Record)
def window_slicing(DataSet):
    return None

train_set = DS_creat(DS1_rec)
testing_set = DS_creat(DS2_rec)
print(train_set)