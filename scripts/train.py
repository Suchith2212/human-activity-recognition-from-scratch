"""
CLI training script for the Human Activity Recognition pipeline.

Provides a command-line interface to run the full training pipeline:
    1. Load configuration
    2. Preprocess and window the dataset
    3. Extract features (statistical + FFT spectral)
    4. Train the from-scratch Decision Tree or Random Forest
    5. Evaluate and print a classification report
    6. Save results

Usage:
    python scripts/train.py                          # Use default config
    python scripts/train.py --config config/config.yaml
    python scripts/train.py --max-depth 10 --criterion gini
    python scripts/train.py --augment               # Enable data augmentation
    python scripts/train.py --model forest --augment # Random Forest + augmentation
"""

import argparse
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.preprocessing import StandardScaler

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.decision_tree import DecisionTree
from src.models.random_forest import RandomForest
from src.evaluation.metrics import (
    accuracy,
    classification_report,
    confusion_matrix,
)
from src.utils.helpers import load_config, set_seed, setup_logging

logger = logging.getLogger(__name__)


# ============================================================================
# Feature Extraction
# ============================================================================


def extract_statistical_features(X: np.ndarray) -> np.ndarray:
    """Extract hand-crafted statistical features from windowed data.

    Computes per-channel: mean, std, min, max, median, energy, zero-crossing
    rate, and inter-quartile range — yielding a compact feature vector.
    This implementation is fully vectorized using NumPy.

    Parameters
    ----------
    X : np.ndarray of shape (n_samples, window_size, n_channels)
        Windowed accelerometer data.

    Returns
    -------
    np.ndarray of shape (n_samples, n_channels × 8)
        Statistical feature matrix.
    """
    X = np.asarray(X, dtype=np.float64)
    
    means = np.mean(X, axis=1)
    stds = np.std(X, axis=1)
    mins = np.min(X, axis=1)
    maxs = np.max(X, axis=1)
    medians = np.median(X, axis=1)
    energies = np.sum(X ** 2, axis=1)
    zcrs = np.sum(np.diff(np.sign(X), axis=1) != 0, axis=1)
    iqrs = np.percentile(X, 75, axis=1) - np.percentile(X, 25, axis=1)

    # Stack along a new axis to get (n_samples, 8, n_channels)
    stacked = np.stack([means, stds, mins, maxs, medians, energies, zcrs, iqrs], axis=1)
    
    # Transpose to (n_samples, n_channels, 8) and flatten to (n_samples, n_channels * 8)
    # to preserve the channel-first ordering of features per sample
    features = np.transpose(stacked, (0, 2, 1)).reshape(X.shape[0], -1)
    return features


def extract_fft_features(X: np.ndarray) -> np.ndarray:
    """Extract frequency-domain features from windowed accelerometer data.

    Computes per-channel:
        - **Spectral Energy**: Sum of squared FFT magnitudes, capturing
          the total energy content across all frequency bins.
        - **Dominant Frequency**: The frequency bin index with the highest
          FFT magnitude, indicating the primary oscillation pattern
          (e.g., walking cadence vs. static posture).

    These features help the model distinguish between activities that
    look similar in the time domain but differ in frequency content
    (e.g., WALKING vs. WALKING_UPSTAIRS).

    Parameters
    ----------
    X : np.ndarray of shape (n_samples, window_size, n_channels)
        Windowed accelerometer data.

    Returns
    -------
    np.ndarray of shape (n_samples, n_channels × 2)
        FFT feature matrix (spectral energy + dominant frequency per channel).
    """
    X = np.asarray(X, dtype=np.float64)

    # Compute FFT along the time axis (axis=1), take only positive frequencies
    fft_vals = np.fft.rfft(X, axis=1)
    fft_magnitudes = np.abs(fft_vals)

    # Spectral energy: sum of squared magnitudes per channel
    spectral_energy = np.sum(fft_magnitudes ** 2, axis=1)

    # Dominant frequency: index of the max magnitude per channel
    # Skip DC component (index 0) to find the dominant oscillatory frequency
    dominant_freq = np.argmax(fft_magnitudes[:, 1:, :], axis=1) + 1

    # Stack: (n_samples, 2, n_channels) -> flatten to (n_samples, n_channels * 2)
    stacked = np.stack([spectral_energy, dominant_freq.astype(np.float64)], axis=1)
    features = np.transpose(stacked, (0, 2, 1)).reshape(X.shape[0], -1)
    return features


def extract_all_features(X: np.ndarray) -> np.ndarray:
    """Extract combined statistical + FFT features from windowed data.

    Parameters
    ----------
    X : np.ndarray of shape (n_samples, window_size, n_channels)
        Windowed accelerometer data.

    Returns
    -------
    np.ndarray of shape (n_samples, n_channels × 10)
        Combined feature matrix (8 statistical + 2 FFT features per channel).
    """
    stat_feats = extract_statistical_features(X)
    fft_feats = extract_fft_features(X)
    return np.hstack([stat_feats, fft_feats])


# ============================================================================
# Main Pipeline
# ============================================================================

def main() -> None:
    """Run the full training pipeline."""
    args = parse_args()
    setup_logging()

    logger.info("=" * 70)
    logger.info("  Human Activity Recognition — Training Pipeline")
    logger.info("=" * 70)

    # ---- Load configuration ----
    config = load_config(args.config)

    # Override config with CLI arguments
    if args.max_depth is not None:
        config["decision_tree"]["max_depth"] = args.max_depth
    if args.criterion is not None:
        config["decision_tree"]["criterion"] = args.criterion
    if args.seed is not None:
        config["training"]["random_seed"] = args.seed

    seed = config["training"]["random_seed"]
    set_seed(seed)

    # ---- Load data ----
    logger.info("Loading and preprocessing data...")

    # Try to import and use the preprocessing pipeline
    try:
        from src.data.preprocessing import load_and_split_dataset
        X_train, X_test, y_train, y_test = load_and_split_dataset(config)
    except FileNotFoundError:
        logger.warning(
            "UCI HAR Dataset not found at configured path. "
            "Using sklearn's built-in dataset for demonstration."
        )
        # Fallback: generate synthetic data for demo purposes
        from sklearn.datasets import make_classification
        X_flat, y = make_classification(
            n_samples=300, n_features=24, n_classes=6,
            n_informative=18, n_clusters_per_class=1,
            random_state=seed,
        )
        from sklearn.model_selection import train_test_split
        X_train_flat, X_test_flat, y_train, y_test = train_test_split(
            X_flat, y, test_size=0.3, random_state=seed, stratify=y,
        )
        # Skip windowed feature extraction for synthetic data
        X_train_features = X_train_flat
        X_test_features = X_test_flat
        logger.info("Synthetic data: Train=%s, Test=%s", X_train_features.shape, X_test_features.shape)
        _run_training(config, X_train_features, X_test_features, y_train, y_test, model_type=args.model)
        return

    # ---- Data augmentation (optional) ----
    if args.augment or config.get("augmentation", {}).get("enabled", False):
        from src.data.augmentation import augment_dataset
        logger.info("Applying data augmentation...")
        X_train, y_train = augment_dataset(X_train, y_train, config, seed=seed)

    # ---- Feature extraction (statistical + FFT) ----
    logger.info("Extracting statistical + FFT features...")
    X_train_features = extract_all_features(X_train)
    X_test_features = extract_all_features(X_test)

    logger.info(
        "Feature matrix: Train=%s, Test=%s",
        X_train_features.shape, X_test_features.shape,
    )

    _run_training(config, X_train_features, X_test_features, y_train, y_test, model_type=args.model)


def _run_training(
    config: dict,
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    model_type: str = "tree",
) -> None:
    """Train, evaluate, and report results."""

    # ---- Scaling ----
    if config["training"].get("scaling", True):
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

    # ---- Build model ----
    dt_config = config["decision_tree"]
    seed = config["training"]["random_seed"]

    if model_type == "forest":
        rf_config = config.get("random_forest", {})
        model = RandomForest(
            n_estimators=rf_config.get("n_estimators", 100),
            max_depth=dt_config["max_depth"],
            min_samples_split=dt_config["min_samples_split"],
            criterion=dt_config["criterion"],
            random_state=seed,
        )
        model_label = "From-Scratch Random Forest"
    else:
        model = DecisionTree(
            max_depth=dt_config["max_depth"],
            min_samples_split=dt_config["min_samples_split"],
            criterion=dt_config["criterion"],
        )
        model_label = "From-Scratch Decision Tree"

    logger.info("Training %s...", model)
    start = time.perf_counter()
    model.fit(X_train, y_train)
    train_time = time.perf_counter() - start

    # ---- Predict ----
    start = time.perf_counter()
    y_pred = model.predict(X_test)
    pred_time = time.perf_counter() - start

    # ---- Evaluate ----
    acc = accuracy(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    activity_names = list(config.get("activities", {}).values())
    if not activity_names:
        activity_names = [f"Class {i}" for i in sorted(np.unique(y_test))]

    report = classification_report(y_test, y_pred, target_names=activity_names)

    # ---- Print results ----
    print("\n" + "=" * 70)
    print(f"  RESULTS — {model_label}")
    print("=" * 70)
    print(f"  Model:           {model}")
    if model_type == "tree":
        print(f"  Tree Depth:      {model.get_depth()}")
        print(f"  Leaf Nodes:      {model.get_n_leaves()}")
    else:
        print(f"  Num Trees:       {model.n_estimators}")
        print(f"  Max Depth:       {model.max_depth}")
    print(f"  Training Time:   {train_time:.2f}s")
    print(f"  Prediction Time: {pred_time:.4f}s")
    print(f"  Accuracy:        {acc:.4f} ({acc * 100:.1f}%)")
    print("-" * 70)
    print("  Classification Report:")
    print(report)
    print("-" * 70)
    print("  Confusion Matrix:")
    print(cm)
    print("=" * 70)

    # ---- Save results (optional) ----
    output_dir = config.get("output", {}).get("results_dir", "outputs/results")
    os.makedirs(output_dir, exist_ok=True)

    results_path = os.path.join(output_dir, "training_results.txt")
    with open(results_path, "w", encoding="utf-8") as f:
        f.write(f"Model: {model}\n")
        f.write(f"Accuracy: {acc:.4f}\n")
        f.write(f"Training Time: {train_time:.2f}s\n\n")
        f.write("Classification Report:\n")
        f.write(report + "\n\n")
        f.write("Confusion Matrix:\n")
        f.write(str(cm) + "\n")

    logger.info("Results saved to %s", results_path)


# ============================================================================
# Argument Parsing
# ============================================================================

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train a Decision Tree or Random Forest on the UCI HAR Dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/train.py
  python scripts/train.py --max-depth 10 --criterion gini
  python scripts/train.py --augment --seed 123
  python scripts/train.py --model forest --augment
        """,
    )
    parser.add_argument(
        "--config", type=str, default="config/config.yaml",
        help="Path to YAML configuration file (default: config/config.yaml)",
    )
    parser.add_argument(
        "--model", type=str, choices=["tree", "forest"], default="tree",
        help="Model type: 'tree' for Decision Tree, 'forest' for Random Forest (default: tree)",
    )
    parser.add_argument(
        "--max-depth", type=int, default=None,
        help="Override maximum tree depth",
    )
    parser.add_argument(
        "--criterion", type=str, choices=["entropy", "gini"], default=None,
        help="Split criterion: entropy or gini",
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Override random seed for reproducibility",
    )
    parser.add_argument(
        "--augment", action="store_true",
        help="Enable data augmentation",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
