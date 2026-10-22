# Data-Mining-Project

## Data

This project uses the **MIT-BIH Arrhythmia Database** from PhysioNet.

**The raw data is NOT included in this repository** — per assignment instructions
("Do not submit the complete raw PhysioNet datasets") and to keep the repo size
manageable.

### How to get the data

1. Download the database from PhysioNet:
   https://physionet.org/content/mitdb/1.0.0/
   (Download the ZIP file, ~73.5 MB)

2. Unzip it and place the folder inside `data/`, so the structure looks like:

data/
└── mit-bih-arrhythmia-database-1.0.0/
├── 100.dat
├── 100.hea
├── 100.atr
└── ...


3. The `data/` folder is excluded from version control via `.gitignore`.

4. In order to split the data into Train, Val and Testing with our preprocessing run `py src/data_split.py` from root. This will store the data in data/test.npz, data/train.npz and data/val.npz.

### Environment setup

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

