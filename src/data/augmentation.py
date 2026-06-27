"""
Signal-level data augmentation for time-series accelerometer data.

Provides lightweight, dependency-free augmentation techniques commonly used
in sensor-based human activity recognition to improve model generalization.

Implemented Techniques:
    - **Jittering**: Additive Gaussian noise to simulate sensor noise
    - **Scaling**: Random magnitude scaling to simulate varying sensor gain
    - **Time Warping**: Smooth temporal distortion via cubic interpolation

These methods are based on:
    - Um, T.T. et al. (2017). "Data Augmentation of Wearable Sensor Data
      for Parkinson's Disease Monitoring using Convolutional Neural Networks."
    - Iwana, B.K. & Uchida, S. (2021). "An Empirical Survey of Data
      Augmentation for Time Series Classification with Neural Networks."

Usage:
    >>> from src.data.augmentation import augment_dataset
    >>> X_aug, y_aug = augment_dataset(X_train, y_train, config)
"""

import logging
from typing import Any, Dict, Optional, Tuple

import numpy as np
from scipy.interpolate import CubicSpline

logger = logging.getLogger(__name__)


# ============================================================================
# Individual Augmentation Techniques
# ============================================================================

def jitter(
    X: np.ndarray,
    sigma: float = 0.05,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Add Gaussian noise to simulate sensor measurement noise.

    Parameters
    ----------
    X : np.ndarray of shape (n_samples, seq_len, n_channels)
        Input accelerometer data.
    sigma : float
        Standard deviation of the Gaussian noise.
    rng : np.random.Generator, optional
        Random number generator for reproducibility.

    Returns
    -------
    np.ndarray
        Noisy version of the input data (same shape).
    """
    if rng is None:
        rng = np.random.default_rng()
    noise = rng.normal(loc=0.0, scale=sigma, size=X.shape)
    return X + noise


def scaling(
    X: np.ndarray,
    sigma: float = 0.1,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Apply random scaling to simulate varying sensor sensitivity.

    Each sample is multiplied by a random factor drawn from
    ``N(1.0, sigma)``, applied uniformly across all time steps.

    Parameters
    ----------
    X : np.ndarray of shape (n_samples, seq_len, n_channels)
        Input accelerometer data.
    sigma : float
        Standard deviation of the scaling factor.
    rng : np.random.Generator, optional
        Random number generator for reproducibility.

    Returns
    -------
    np.ndarray
        Scaled version of the input data.
    """
    if rng is None:
        rng = np.random.default_rng()
    # One scaling factor per sample per channel, clipped to be positive
    factors = rng.normal(loc=1.0, scale=sigma, size=(X.shape[0], 1, X.shape[2]))
    factors = np.clip(factors, 0.01, None)
    return X * factors


def time_warp(
    X: np.ndarray,
    sigma: float = 0.2,
    n_knots: int = 4,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Apply smooth temporal distortion via cubic spline warping.

    Creates a random, smooth warping path using cubic spline interpolation on [0, 1],
    then resamples the time series along this warped path scaled back to the original length.

    Parameters
    ----------
    X : np.ndarray of shape (n_samples, seq_len, n_channels)
        Input accelerometer data.
    sigma : float
        Standard deviation of the random knot displacements on the [0, 1] scale.
    n_knots : int
        Number of interior knots for the cubic spline.
    rng : np.random.Generator, optional
        Random number generator for reproducibility.

    Returns
    -------
    np.ndarray
        Time-warped version of the input data.
    """
    if rng is None:
        rng = np.random.default_rng()

    n_samples, seq_len, n_channels = X.shape
    X_warped = np.empty_like(X)

    orig_steps = np.arange(seq_len)
    t = np.linspace(0, 1, seq_len)

    for i in range(n_samples):
        # Generate random warp path via cubic spline on [0, 1]
        knot_positions = np.linspace(0, 1, n_knots + 2)
        knot_values = knot_positions + rng.normal(scale=sigma, size=len(knot_positions))
        knot_values[0], knot_values[-1] = 0.0, 1.0  # Anchor endpoints
        
        # Ensure monotonic increasing to prevent time reversals
        knot_values = np.sort(knot_values)

        spline = CubicSpline(knot_positions, knot_values)
        warped_steps = spline(t) * (seq_len - 1)

        # Clip to valid range and interpolate each channel
        warped_steps = np.clip(warped_steps, 0, seq_len - 1)

        for c in range(n_channels):
            X_warped[i, :, c] = np.interp(warped_steps, orig_steps, X[i, :, c])

    return X_warped


# ============================================================================
# Combined Augmentation Pipeline
# ============================================================================

def augment_dataset(
    X: np.ndarray,
    y: np.ndarray,
    config: Optional[Dict[str, Any]] = None,
    n_copies: int = 2,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Apply a suite of augmentations and combine with original data.

    For each original sample, generates ``n_copies`` augmented versions
    using a random combination of jittering, scaling, and time warping.

    Parameters
    ----------
    X : np.ndarray of shape (n_samples, seq_len, n_channels)
        Original training data.
    y : np.ndarray of shape (n_samples,)
        Original training labels.
    config : dict, optional
        Configuration dictionary with augmentation parameters.
        Expected keys under ``config["augmentation"]``:
        ``jitter_sigma``, ``scaling_sigma``, ``num_augmented_copies``.
    n_copies : int
        Number of augmented copies per original sample (overridden by config).
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    X_combined : np.ndarray
        Concatenation of original and augmented samples.
    y_combined : np.ndarray
        Corresponding labels (augmented samples inherit parent labels).
    """
    rng = np.random.default_rng(seed)

    # Read augmentation parameters from config (with defaults)
    jitter_sigma = 0.05
    scaling_sigma = 0.1
    time_warp_sigma = 0.2

    if config is not None:
        aug_cfg = config.get("augmentation", {})
        jitter_sigma = aug_cfg.get("jitter_sigma", jitter_sigma)
        scaling_sigma = aug_cfg.get("scaling_sigma", scaling_sigma)
        time_warp_sigma = aug_cfg.get("time_warp_sigma", time_warp_sigma)
        n_copies = aug_cfg.get("num_augmented_copies", n_copies)

    augmented_X_list = [X]
    augmented_y_list = [y]

    techniques = [
        ("jitter", lambda data: jitter(data, sigma=jitter_sigma, rng=rng)),
        ("scaling", lambda data: scaling(data, sigma=scaling_sigma, rng=rng)),
        ("time_warp", lambda data: time_warp(data, sigma=time_warp_sigma, rng=rng)),
    ]

    for copy_idx in range(n_copies):
        # Apply a random subset of techniques per copy
        n_techniques = rng.integers(1, len(techniques) + 1)
        selected = rng.choice(len(techniques), size=n_techniques, replace=False)

        X_aug = X.copy()
        applied = []
        for idx in selected:
            name, transform = techniques[idx]
            X_aug = transform(X_aug)
            applied.append(name)

        augmented_X_list.append(X_aug)
        augmented_y_list.append(y.copy())

        logger.info(
            "Augmentation copy %d/%d — applied: %s",
            copy_idx + 1, n_copies, ", ".join(applied),
        )

    X_combined = np.concatenate(augmented_X_list, axis=0)
    y_combined = np.concatenate(augmented_y_list, axis=0)

    logger.info(
        "✅ Augmentation complete: %d → %d samples (%.1f× expansion)",
        len(X), len(X_combined), len(X_combined) / len(X),
    )

    return X_combined, y_combined
