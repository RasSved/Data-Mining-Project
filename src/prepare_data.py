import numpy as np
from dataset import DS_creat
import os

records = [101, 106, 108, 109, 112, 114, 115, 116, 118, 119, 122, 124, 
           201, 203, 205, 207, 208, 209, 215, 220, 223, 230, 100, 103, 
           105, 111, 113, 117, 121, 123, 200, 202, 210, 212, 213, 214, 
           219, 221, 222, 228, 231, 232, 233, 234]

# -------------------------------------------------------------------- Normalizing ---------------------------------------------------------

# (record: Signal, peaks, labels)
def preprocessing(DataSet):
    # Median down the peaks 200ms then 600ms
    for record in DataSet:
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
        # TODO: Noch filter?


        # Get Z-score for peak normalizations
        # Mean 0 standard diviation 1
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
    before_peak = 90
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

# -------------------------------------------------------------------- Saving/loading ---------------------------------------------------------

# This is to save the fixed sets localy so we dont have to run this every time (prob make a script file for this)
def save_split(dir, train, val, test):
    os.makedirs(dir, exist_ok=True)
    for name, (signals, labels, records) in zip(['train', 'val', 'test'], [train, val, test]):
        np.savez(
            os.path.join(dir, f'{name}.npz'),
            windows=signals,
            labels=labels,
            records=records
        )

def load_split(out_dir, name):
    data = np.load(os.path.join(out_dir, f'{name}.npz'), allow_pickle=True)
    return data['signalss'], data['labels'], data['records']


# -------------------------------------------------------------------- Class imbalance ---------------------------------------------------------

# Now we need to make sure we handle the class imbalance and its done with weights and balanced sampling
# Both used on trainingset
def class_weights(labels): # [a,s,d,f,s,s]
    new_weights = []
    values, counts = np.unique(labels, return_counts=True) # values: s: 3
    for classes in range(len(values)):
        scale = float(len(labels) / (len(values) * counts[classes])) # total / numclass * numofthisclass
        new_weights.append(scale)
    return new_weights

def balanced_sampling(labels): # 1 / numofthisclass [s,s,s,a,a,f] s = 1 / 3 a = 1 / 2 f = 1/1
    rebalanced = []
    class_balance = {}
    values, counts = np.unique(labels, return_counts=True)
    for val in values:
        class_balance[val] = 1 / counts[val]

    for classes in labels:
        rebalanced.append(float(class_balance.get(classes)))
    return rebalanced


# -------------------------------------------------------------------- Augmentation ---------------------------------------------------------

# Cant be bothered to implement this now but i think shifting and some extra noise is all we need 
# input [1,2,3,4,5] shift 2
# output [3,4,5,5,5]

def add_shift(signal, shift):
    sig_copy = signal.copy()
    if shift == 0:
        return signal
    elif shift > 0:
        pad = sig_copy[shift:]
        out = np.pad(pad, (0, shift), mode='edge')
    else:
        pad = sig_copy[:len(signal) - abs(shift)]
        out = np.pad(pad, (abs(shift), 0), mode='edge')
    return np.array(out)

#Input: signal = [1,2,3], sigma = 0.1
#Output: [-1.2037, -0.0410, 1.2447]
#Du får en signal, ett brusvärde sigma och en lista z med färdiga slumptal från standardnormalfördelningen N(0, 1). Skapa den brusiga signalen:


    # y[i] = x[i] + ε[i]

def add_noise(signal, epi):
    random = np.random.default_rng(0)
    noise = random.normal(loc=0.0, scale=epi, size=len(signal)) # ε[i]
    print(noise)

    x = signal.copy() # x[i]
    y = x + noise

    #mean = y.mean()
    #standard = y.std()
    #normalized = (y - mean) / standard

    return y


x = np.array([1.0, 2.0, 3.0])
print(add_noise(x, 0.1))
