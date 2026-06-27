"""Evaluation metrics subpackage."""

__all__ = [
    "accuracy",
    "precision",
    "recall",
    "f1_score",
    "confusion_matrix",
    "classification_report",
]


def __getattr__(name: str):
    """Lazy imports — only load modules when actually accessed."""
    if name in __all__:
        from src.evaluation import metrics as _m
        return getattr(_m, name)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
