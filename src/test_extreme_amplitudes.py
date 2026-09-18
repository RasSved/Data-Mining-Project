import wfdb
import numpy as np

record = wfdb.rdrecord(
    "data/mit-bih-arrhythmia-database-1.0.0/100"
)

signal = record.p_signal

print("Shape:", signal.shape)

for channel in range(signal.shape[1]):
    print(f"\nChannel {channel + 1}")
    print("Min:", np.min(signal[:, channel]))
    print("Max:", np.max(signal[:, channel]))
    print("Mean:", np.mean(signal[:, channel]))
    print("Std:", np.std(signal[:, channel]))