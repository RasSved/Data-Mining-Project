import wfdb
from collections import Counter
from pathlib import Path

DATA_PATH = Path("data/mit-bih-arrhythmia-database-1.0.0")

# Preliminary mapping
CLASS_MAP = {
    "N": "N",

    "A": "S",
    "a": "S",
    "J": "S",
    "S": "S",

    "V": "V",
    "E": "V",

    "F": "F",
}

total_counts = Counter()

records = sorted(
    path.stem
    for path in DATA_PATH.glob("*.atr")
)

for record_name in records:
    annotation = wfdb.rdann(
        str(DATA_PATH / record_name),
        "atr"
    )

    for symbol in annotation.symbol:
        if symbol in CLASS_MAP:
            target_class = CLASS_MAP[symbol]
            total_counts[target_class] += 1

print("Class distribution:")

total = sum(total_counts.values())

for class_name in ["N", "S", "V", "F"]:
    count = total_counts[class_name]
    percentage = count / total * 100

    print(
        f"{class_name}: "
        f"{count} beats "
        f"({percentage:.2f}%)"
    )

print(f"\nTotal classified beats: {total}")