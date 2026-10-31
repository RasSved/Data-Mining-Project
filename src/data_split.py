import os
import numpy as np

from dataset import DS_creat
from prepare_data import preprocessing, window_slicing, save_split, classes

OUT_DIR = "data/"

DS1 = [101, 106, 108, 109, 112, 114, 115, 116, 118, 119, 122, 124,
       201, 203, 205, 207, 208, 209, 215, 220, 223, 230]
DS2 = [100, 103, 105, 111, 113, 117, 121, 123, 200, 202, 210, 212,
       213, 214, 219, 221, 222, 228, 231, 232, 233, 234]

VAL_RECORDS = [108, 114, 201, 207, 223]
TRAIN_RECORDS = [r for r in DS1 if r not in VAL_RECORDS]
TEST_RECORDS = DS2


def split_stats(name, labels, records):
    row = {"split": name, "records": len(np.unique(records))}
    for c in classes:
        row[c] = int((labels == c).sum())
    row["total"] = int(len(labels))
    return row


def main():
    # Leakage check: no record may be in two splits
    assert not set(TRAIN_RECORDS) & set(VAL_RECORDS), "train/val overlap"
    assert not set(DS1) & set(TEST_RECORDS), "DS1/test overlap"

    splits = {"train": TRAIN_RECORDS, "val": VAL_RECORDS, "test": TEST_RECORDS}
    results = {}
    stats = []

    for name, recs in splits.items():
        print(f"Processing {name} ({len(recs)} records)...")
        DataSet = DS_creat(recs)
        DataSet = preprocessing(DataSet)
        windows, labels, beat_records, feats = window_slicing(DataSet)
        results[name] = (windows.astype(np.float32), labels, beat_records, feats)
        stats.append(split_stats(name, labels, beat_records))

    save_split(OUT_DIR, results["train"], results["val"], results["test"])

    # Print and save the class-count table (section 3.8)
    print(f"\n{'Split':<7}{'Recs':>6}" + "".join(f"{c:>8}" for c in classes) + f"{'Total':>9}")
    for row in stats:
        print(f"{row['split']:<7}{row['records']:>6}"
              + "".join(f"{row[c]:>8}" for c in classes) + f"{row['total']:>9}")
        missing = [c for c in classes if row[c] == 0]
        if missing:
            print(f"  WARNING: {row['split']} has no {missing} beats")

    with open(os.path.join(OUT_DIR, "split_stats.csv"), "w") as f:
        f.write("split,records," + ",".join(classes) + ",total\n")
        for row in stats:
            f.write(f"{row['split']},{row['records']},"
                    + ",".join(str(row[c]) for c in classes) + f",{row['total']}\n")

    print(f"\nSaved train.npz, val.npz, test.npz and split_stats.csv to {OUT_DIR}/")


if __name__ == "__main__":
    main()