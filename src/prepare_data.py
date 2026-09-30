import numpy as np
from dataset import DS_creat

DS1_rec = [101]
", 106 , 108, 109, 112, 114, 115, 116, 118, 119, 122, 124, 201, 203, 205, 207, 208, 209, 215, 220, 223, 230 "
DS2_rec = [100, 103, 105, 111, 113, 117, 121, 123, 200, 202, 210, 212, 
       213, 214, 219, 221, 222, 228, 231, 232, 233, 234 ]


def preprocessing(DataSet):
    # Median down the peaks 200ms then 600ms
    for record in DataSet:
        print(record)
        # 0.2s * 360Hz = 72 samples
        twohundred_filter = 73
        signal, peaks, labels = DataSet[record]

        half = twohundred_filter // 2
        padded = np.pad(signal, half, mode="reflect")
        baseline = np.empty(len(signal))

        for i in range(len(signal)):
            baseline[i] = np.median(padded[i : i + twohundred_filter])

        # 0.6s * 360Hz = 216 samples
        sixhundred_filter = 217

        half_base = sixhundred_filter // 2
        padded_base = np.pad(baseline, half_base, mode="reflect")
        final_baseline = np.empty(len(baseline))

        for i in range(len(baseline)):
            final_baseline[i] = np.median(padded_base[i : i + sixhundred_filter])
        
        signal = signal - final_baseline
        # TODO: LowPass filter?


        # Get Z-score for peak normalizations
        mean = signal.mean()
        standard = signal.std()
        signal = (signal - mean) / standard

        DataSet[record] = signal, peaks, labels
    return DataSet

# From dict to (Signal, Label, Record)
# input: (Record: Signal, peaks, labels)
# output: [signal, label, record] 
def window_slicing(DataSet):
    windows = []
    beat_labels = []
    beat_records = []
    before_peak = 36
    after_peak = 144
    for record in DataSet:
        signal, peaks, labels = DataSet[record]
        keep = (peaks - before_peak >= 0) & (peaks + after_peak <= len(signal))
        peaks, labels = peaks[keep], labels[keep]
        for peak, label in zip(peaks, labels):
            windows.append(signal[peak - before_peak : peak + after_peak])
            beat_labels.append(label)
            beat_records.append(record)

    return np.stack(windows), np.array(beat_labels), np.array(beat_records)

        



train_set = DS_creat(DS1_rec)
testing_set = DS_creat(DS2_rec)
pp = preprocessing(train_set)
print(window_slicing(pp))