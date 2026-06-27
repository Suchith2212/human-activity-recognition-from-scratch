"""
Utility helpers for reproducibility, configuration, and benchmarking.

This module provides foundational utilities used across the entire pipeline:
- YAML configuration loading with validation
- Random seed management for reproducibility
- Timing decorator for performance benchmarking
- Logging setup for consistent output formatting
"""

import functools
import logging
import os
import random
import time
from pathlib import Path
from typing import Any, Callable, Dict, Optional, TypeVar, cast

import numpy as np
import yaml

F = TypeVar("F", bound=Callable[..., Any])

logger = logging.getLogger(__name__)


# ============================================================================
# Configuration
# ============================================================================

def load_config(config_path: str = "config/config.yaml") -> Dict[str, Any]:
    """Load and validate a YAML configuration file.

    Parameters
    ----------
    config_path : str
        Path to the YAML configuration file.  Supports both absolute paths
        and paths relative to the project root.

    Returns
    -------
    Dict[str, Any]
        Parsed configuration dictionary.

    Raises
    ------
    FileNotFoundError
        If the configuration file does not exist.
    yaml.YAMLError
        If the YAML is malformed.

    Examples
    --------
    >>> config = load_config("config/config.yaml")
    >>> config["training"]["random_seed"]
    42
    """
    path = Path(config_path)
    if not path.is_absolute():
        # Resolve relative to project root (two levels up from src/utils/)
        project_root = Path(__file__).resolve().parents[2]
        path = project_root / path

    if not path.exists():
        raise FileNotFoundError(
            f"Configuration file not found: {path}\n"
            f"Expected location: {path.resolve()}"
        )

    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    logger.info("Loaded configuration from %s", path)
    return config


# ============================================================================
# Reproducibility
# ============================================================================

def set_seed(seed: int = 42) -> None:
    """Set random seeds across all libraries for reproducibility.

    Ensures deterministic behavior for NumPy and Python's built-in
    random module.  If PyTorch is installed, it sets torch seeds as well.

    Parameters
    ----------
    seed : int
        The random seed value.

    Examples
    --------
    >>> set_seed(42)
    >>> np.random.rand()  # Will produce the same value every time
    0.3745401188473625
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    # Optionally set PyTorch seed if available
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass

    logger.info("Random seed set to %d", seed)


# ============================================================================
# Timing / Benchmarking
# ============================================================================

def timer(func: F) -> F:
    """Decorator that measures and logs the execution time of a function.

    Parameters
    ----------
    func : Callable
        The function to be timed.

    Returns
    -------
    Callable
        Wrapped function that logs elapsed time after execution.

    Examples
    --------
    >>> @timer
    ... def slow_function():
    ...     time.sleep(1)
    >>> slow_function()
    [TIMER] slow_function executed in 1.00s
    """

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        logger.info("[TIMER] %s executed in %.2fs", func.__name__, elapsed)
        print(f"[TIMER] {func.__name__} executed in {elapsed:.2f}s")
        return result

    return cast(F, wrapper)


# ============================================================================
# Data Utilities
# ============================================================================

def shuffle_data(
    X: np.ndarray, y: np.ndarray, seed: Optional[int] = None
) -> tuple[np.ndarray, np.ndarray]:
    """Shuffle feature matrix and labels in unison.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix of shape ``(n_samples, ...)``.
    y : np.ndarray
        Label array of shape ``(n_samples,)``.
    seed : int, optional
        Random seed for reproducible shuffling.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Shuffled ``(X, y)`` pair.
    """
    rng = np.random.default_rng(seed)
    idx = np.arange(X.shape[0])
    rng.shuffle(idx)
    return X[idx], y[idx]


# ============================================================================
# Logging Setup
# ============================================================================

def setup_logging(level: int = logging.INFO) -> None:
    """Configure logging with a clean, readable format.

    Parameters
    ----------
    level : int
        Logging level (e.g., ``logging.INFO``, ``logging.DEBUG``).
    """
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(name)-25s | %(levelname)-7s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
