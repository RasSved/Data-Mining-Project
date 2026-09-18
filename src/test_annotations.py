import wfdb
from collections import Counter
from pathlib import Path

DATA_PATH = Path("data/mit-bih-arrhythmia-database-1.0.0")

records = sorted(
    path.stem
    for path in DATA_PATH.glob("*.atr")
)

total_counts = Counter()

for record_name in records:
    annotation = wfdb.rdann(
        str(DATA_PATH / record_name),
        "atr"
    )

    counts = Counter(annotation.symbol)
    total_counts.update(counts)

    print(f"{record_name}: {counts}")

print("\nTOTAL:")
print(total_counts)
print(f"\nNumber of records: {len(records)}")