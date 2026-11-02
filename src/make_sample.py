import os
import sys
sys.path.append(os.path.expanduser("~/ai8x/ai8x-training"))

import numpy as np
from prepare_data import make_dataloaders

INDEX = 0   # which test beat to use

_, _, test_dl, _ = make_dataloaders("data", 64, 1, False)
x, y = test_dl.dataset[INDEX]                       # float, (7, 234), in [-1, 127/128]
sample = (x.numpy() * 128).round().clip(-128, 127).astype(np.int64)

os.makedirs("results", exist_ok=True)
np.save("results/sample_ecg.npy", sample, allow_pickle=False, fix_imports=False)
print("Saved results/sample_ecg.npy, shape", sample.shape, "label", int(y))