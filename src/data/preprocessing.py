"""
Data preprocessing pipeline for the UCI HAR Dataset.

Handles the complete data engineering workflow:
    1. Combining raw inertial signals into subject-activity CSV files
    2. Windowed dataset creation with configurable offset and duration
    3. Stratified train/test splitting

This module replaces the original ``Pre_Processing.py`` with:
    - Configurable paths (no hardcoded Kaggle directories)
    - Reusable functions instead of top-level script execution
    - Elimination of duplicated train/test combination logic
    - Proper type hints, docstrings, and logging

Usage:
    >>> from src.data.preprocessing import load_and_split_dataset
    >>> X_train, X_test, y_train, y_test = load_and_split_dataset(config)
"""

import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)

# Default activity label mapping (from the UCI HAR Dataset documentation)
DEFAULT_ACTIVITIES = {
    1: "WALKING",
    2: "WALKING_UPSTAIRS",
    3: "WALKING_DOWNSTAIRS",
    4: "SITTING",
    5: "STANDING",
    6: "LAYING",
}

# Inverse mapping: activity name → numeric class
DEFAULT_CLASSES = {name: idx for idx, name in DEFAULT_ACTIVITIES.items()}


# ============================================================================
# Stage 1: Combine Raw Inertial Signals
# ============================================================================

def _combine_split(
    split_path: str,
    split_name: str,
    output_dir: str,
    activities: Dict[int, str],
    signal_offset: int = 64,
) -> None:
    """Combine inertial signal files for a single data split (train or test).

    Reads the raw ``total_acc_x``, ``total_acc_y``, ``total_acc_z`` files
    and groups them by subject and activity into individual CSV files.

    Parameters
    ----------
    split_path : str
        Path to the split directory (e.g., ``.../UCI HAR Dataset/train``).
    split_name : str
        Name of the split (``"Train"`` or ``"Test"``).
    output_dir : str
        Root output directory for combined files.
    activities : dict
        Mapping from activity ID → activity name.
    signal_offset : int
        Number of initial readings to skip per window (transient removal).
    """
    suffix = split_name.lower()
    inertial_dir = os.path.join(split_path, "Inertial Signals")

    # Load tri-axial accelerometer data
    total_acc_x = pd.read_csv(
        os.path.join(inertial_dir, f"total_acc_x_{suffix}.txt"),
        sep=r"\s+", header=None, engine="python",
    )
    total_acc_y = pd.read_csv(
        os.path.join(inertial_dir, f"total_acc_y_{suffix}.txt"),
        sep=r"\s+", header=None, engine="python",
    )
    total_acc_z = pd.read_csv(
        os.path.join(inertial_dir, f"total_acc_z_{suffix}.txt"),
        sep=r"\s+", header=None, engine="python",
    )

    # Load subject IDs and labels
    subjects = pd.read_csv(
        os.path.join(split_path, f"subject_{suffix}.txt"),
        sep=r"\s+", header=None, engine="python",
    )
    labels = pd.read_csv(
        os.path.join(split_path, f"y_{suffix}.txt"),
        sep=r"\s+", header=None, engine="python",
    )

    for subject in np.unique(subjects.values):
        sub_idxs = np.where(subjects.iloc[:, 0] == subject)[0]
        sub_labels = labels.loc[sub_idxs]

        for label in sub_labels.iloc[:, 0].unique():
            activity_name = activities[label]
            save_dir = os.path.join(output_dir, split_name, activity_name)
            os.makedirs(save_dir, exist_ok=True)

            label_idxs = sub_labels[sub_labels.iloc[:, 0] == label].index

            # Vectorized extraction: select the rows matching label_idxs, skip the offset,
            # and flatten the entire 2D block into a 1D array.
            accx = total_acc_x.iloc[label_idxs, signal_offset:].values.flatten()
            accy = total_acc_y.iloc[label_idxs, signal_offset:].values.flatten()
            accz = total_acc_z.iloc[label_idxs, signal_offset:].values.flatten()

            data = pd.DataFrame({"accx": accx, "accy": accy, "accz": accz})
            save_path = os.path.join(save_dir, f"Subject_{subject}.csv")
            data.to_csv(save_path, index=False)

    logger.info("Combined %s data → %s", split_name, output_dir)


def combine_inertial_signals(
    raw_dir: str,
    output_dir: str,
    activities: Optional[Dict[int, str]] = None,
) -> None:
    """Combine raw UCI HAR inertial signals into per-subject CSV files.

    Processes both train and test splits and saves the combined data
    into a structured directory hierarchy.

    Parameters
    ----------
    raw_dir : str
        Path to the UCI HAR Dataset root (contains ``train/`` and ``test/``).
    output_dir : str
        Path where combined CSV files will be written.
    activities : dict, optional
        Activity ID → name mapping.  Uses UCI HAR defaults if ``None``.
    """
    if activities is None:
        activities = DEFAULT_ACTIVITIES

    train_path = os.path.join(raw_dir, "train")
    test_path = os.path.join(raw_dir, "test")

    _combine_split(train_path, "Train", output_dir, activities)
    _combine_split(test_path, "Test", output_dir, activities)

    logger.info("✅ All inertial signals combined successfully")


# ============================================================================
# Stage 2: Create Windowed Dataset
# ============================================================================

def create_windowed_dataset(
    combined_dir: str,
    split: str,
    window_size: int = 500,
    offset: int = 100,
    activity_classes: Optional[Dict[str, int]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Create a windowed feature matrix from combined CSV files.

    Reads per-subject CSV files, applies offset-based slicing, and
    constructs a 3D tensor suitable for ML models.

    Parameters
    ----------
    combined_dir : str
        Root directory containing ``Train/`` and ``Test/`` subdirectories.
    split : str
        Which split to load: ``"Train"`` or ``"Test"``.
    window_size : int
        Number of time steps per window (``sampling_rate × duration``).
    offset : int
        Number of initial samples to skip (transient removal).
    activity_classes : dict, optional
        Activity name → numeric class mapping.

    Returns
    -------
    X : np.ndarray of shape (n_samples, window_size, 3)
        Windowed accelerometer data (x, y, z axes).
    y : np.ndarray of shape (n_samples,)
        Activity labels.
    """
    if activity_classes is None:
        activity_classes = DEFAULT_CLASSES

    X: List[np.ndarray] = []
    y: List[int] = []
    dataset_dir = os.path.join(combined_dir, split)

    for activity_name, class_label in sorted(activity_classes.items()):
        activity_dir = os.path.join(dataset_dir, activity_name)
        if not os.path.isdir(activity_dir):
            logger.warning("Directory not found, skipping: %s", activity_dir)
            continue

        for filename in sorted(os.listdir(activity_dir)):
            filepath = os.path.join(activity_dir, filename)
            df = pd.read_csv(filepath)
            
            # Explicitly select expected accelerometer columns to ensure consistent layout
            if not all(col in df.columns for col in ['accx', 'accy', 'accz']):
                logger.warning("Skipping %s — missing accx/accy/accz columns", filename)
                continue
            windowed = df[['accx', 'accy', 'accz']].iloc[offset : offset + window_size]

            if len(windowed) < window_size:
                logger.warning(
                    "Skipping %s — only %d samples (need %d)",
                    filename, len(windowed), window_size,
                )
                continue

            X.append(windowed.values)
            y.append(class_label)

    return np.array(X), np.array(y)


# ============================================================================
# Stage 3: Full Pipeline — Load & Split
# ============================================================================

def load_and_split_dataset(
    config: Dict[str, Any],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Execute the full data pipeline: load, window, merge, and split.

    This is the main entry point for downstream training scripts.

    Parameters
    ----------
    config : dict
        Configuration dictionary (loaded from ``config/config.yaml``).

    Returns
    -------
    X_train : np.ndarray
        Training features of shape ``(n_train, window_size, 3)``.
    X_test : np.ndarray
        Test features of shape ``(n_test, window_size, 3)``.
    y_train : np.ndarray
        Training labels.
    y_test : np.ndarray
        Test labels.
    """
    combined_dir = config["data"]["combined_dir"]
    signal_cfg = config["signal"]
    train_cfg = config["training"]

    window_size = signal_cfg["window_size"]
    offset = signal_cfg["offset"]
    test_size = train_cfg["test_size"]
    seed = train_cfg["random_seed"]

    # Load both splits
    X_train_raw, y_train_raw = create_windowed_dataset(
        combined_dir, "Train", window_size, offset,
    )
    X_test_raw, y_test_raw = create_windowed_dataset(
        combined_dir, "Test", window_size, offset,
    )

    logger.info(
        "Loaded %d train + %d test samples",
        len(X_train_raw), len(X_test_raw),
    )

    if len(X_train_raw) == 0 or len(X_test_raw) == 0:
        raise FileNotFoundError(
            f"No windowed samples could be loaded from combined directory: {combined_dir}"
        )

    # Merge and re-split with stratification for balanced evaluation
    X = np.concatenate([X_train_raw, X_test_raw])
    y = np.concatenate([y_train_raw, y_test_raw])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y,
    )

    logger.info(
        "Final split → Train: %s  |  Test: %s",
        X_train.shape, X_test.shape,
    )

    return X_train, X_test, y_train, y_test
