import numpy as np
from dataset import DS_creat, BeatDataSet
import os
from scipy.ndimage import median_filter
import torch
from torch.utils.data import WeightedRandomSampler, DataLoader

# This was Recomendation from AI does the exact same thing as our median filter but faster.
# Looked it up because ours was very slow.
# Ill keep ours but comment it out.
def remove_baseline(signal):
    baseline = median_filter(signal, size=73, mode="mirror")
    baseline = median_filter(baseline, size=217, mode="mirror")
    return signal - baseline

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
        # twohundred_filter = 73
        signal, peaks, labels = DataSet[record]
 
        # half = twohundred_filter // 2
        # padded = np.pad(signal, half, mode="reflect")
        # baseline = np.empty(len(signal))
 
        # for i in range(len(signal)):
        #     baseline[i] = np.median(padded[i : i + twohundred_filter])
 
        # # 0.6s * 360Hz = 216 samples
        # sixhundred_filter = 217
 
        # half_base = sixhundred_filter // 2
        # padded_base = np.pad(baseline, half_base, mode="reflect")
        # final_baseline = np.empty(len(baseline))
 
        # for i in range(len(baseline)):
        #     final_baseline[i] = np.median(padded_base[i : i + sixhundred_filter])
        # signal = signal - final_baseline
        signal = remove_baseline(signal)

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
    return data['windows'], data['labels'], data['records']


# -------------------------------------------------------------------- Class imbalance ---------------------------------------------------------

# Now we need to make sure we handle the class imbalance and its done with weights and balanced sampling
# Both used on trainingset
classes = {"N", "S", "F", "V"}
def class_weights(labels):
    labels = np.asarray(labels)
    new_weights = []
    for c in classes:
        count = (labels == c).sum()
        scale = len(labels) / (len(classes) * count)
        new_weights.append(float(scale))
    return new_weights

def balanced_sampling(labels): # 1 / numofthisclass [s,s,s,a,a,f] s = 1 / 3 a = 1 / 2 f = 1/1
    values, counts = np.unique(labels, return_counts=True)
    class_balance = {}
    rebalanced = []
    for val, count in zip(values, counts):
        class_balance[val] = 1 / count

    for label in labels:
        rebalanced.append(float(class_balance[label]))
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

def add_noise(signal, epi, rng=None):
    noise = rng.normal(0.0, epi, size=len(signal))# ε[i]
    x = signal.copy() # x[i]
    y = x + noise

    #mean = y.mean()
    #standard = y.std()
    #normalized = (y - mean) / standard

    return y

# Gatheres our augmentation function under one rooth so we can use it our dataloader
def augment(signal, rng, max_shift=5, epi=0.1):
    n_shift = rng.integers(-max_shift, max_shift, endpoint=True)
    shifted_signal = add_shift(signal, n_shift)
    noised_signal = np.float32(add_noise(shifted_signal, epi, rng))
    return noised_signal

# Applying our balanced sampling function, note we dont do both our class weighing functions since 
# we only need one but we migth swap to the other later if this doesnt preforme
def make_sampler(labels, seed):
    gen = torch.Generator()
    gen.manual_seed(seed)
    balanced = balanced_sampling(labels)
    sample_weights = WeightedRandomSampler(balanced, num_samples=len(labels), replacement=True, generator=gen)
    return sample_weights

# Make our data loaders and we get tuple[dataloader, dataloader, dataloader]
def make_dataloaders(split_dir, batch_size, seed, use_sampler):
    gen = torch.Generator()
    gen.manual_seed(seed)
    rng = np.random.default_rng(gen.initial_seed())
    class_map = {"N": 0, "S": 1, "V": 2, "F": 3}
    
    train_data, val_data, test_data = load_split(split_dir, "train"), load_split(split_dir, "val"), load_split(split_dir, "test")

    def augment_wrap(beat):
            return augment(beat, rng, max_shift=5, epi=0.1)
    
    train_beats, train_labels, train_records = train_data
    val_beats, val_labels, val_records = val_data
    test_beats, test_labels, test_records = test_data

    train_ds = BeatDataSet(train_beats, train_labels, train_records, class_map, augment=augment_wrap)
    val_ds = BeatDataSet(val_beats, val_labels, val_records, class_map, augment=None)
    test_ds = BeatDataSet(test_beats, test_labels, test_records, class_map, augment=None)

    if use_sampler:
        train_loader = DataLoader(train_ds, batch_size, sampler=make_sampler(train_labels, seed), num_workers=0, pin_memory=True)
    else:
        train_loader = DataLoader(train_ds, batch_size, shuffle=True, generator=gen, num_workers=0, pin_memory=True)

    val_loader = DataLoader(val_ds, batch_size, shuffle=False, num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size, shuffle=False, num_workers=0, pin_memory=True)

    return train_loader, val_loader, test_loader
