import wfdb
import numpy as np

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

