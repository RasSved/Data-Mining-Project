import wfdb
from pathlib import Path

DATA_PATH = Path("data/mit-bih-arrhythmia-database-1.0.0")

records = sorted(
    path.stem
    for path in DATA_PATH.glob("*.hea")
)

for record_name in records:
    try:
        record = wfdb.rdrecord(
            str(DATA_PATH / record_name)
        )

        print(
            f"{record_name}: "
            f"OK | "
            f"samples={record.sig_len} | "
            f"channels={record.n_sig}"
        )

    except Exception as e:
        print(
            f"{record_name}: ERROR | {e}"
        )