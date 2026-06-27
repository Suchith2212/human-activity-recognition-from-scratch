"""Data loading, preprocessing, and augmentation subpackage."""

__all__ = [
    "combine_inertial_signals",
    "create_windowed_dataset",
    "load_and_split_dataset",
    "jitter",
    "scaling",
    "time_warp",
    "augment_dataset",
]


def __getattr__(name: str):
    """Lazy imports — only load heavy modules when actually accessed."""
    if name in ("combine_inertial_signals", "create_windowed_dataset", "load_and_split_dataset"):
        from src.data.preprocessing import (
            combine_inertial_signals,
            create_windowed_dataset,
            load_and_split_dataset,
        )
        return locals()[name]

    if name in ("jitter", "scaling", "time_warp", "augment_dataset"):
        from src.data.augmentation import jitter, scaling, time_warp, augment_dataset
        return locals()[name]

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
