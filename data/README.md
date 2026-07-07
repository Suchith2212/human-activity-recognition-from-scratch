# Data Directory

This directory is **not tracked by git** (see `.gitignore`).

## How to set up the dataset

1. **Download** the [UCI HAR Dataset](https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones)
2. **Extract** and place it here so the structure looks like:
   ```
   data/
   └── raw/
       └── UCI HAR Dataset/
           ├── train/
           │   ├── Inertial Signals/
           │   ├── subject_train.txt
           │   ├── X_train.txt
           │   └── y_train.txt
           ├── test/
           │   └── ...
           ├── activity_labels.txt
           └── features.txt
   ```
3. **Preprocess** the raw signals into per-subject CSV files:
   ```bash
   python -c "from src.data.preprocessing import combine_inertial_signals; combine_inertial_signals('data/raw/UCI HAR Dataset', 'data/processed/Combined')"
   ```
4. The `data/processed/Combined/` directory will be auto-created with `Train/` and `Test/` subdirectories per activity.
