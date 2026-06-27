"""Model implementations subpackage."""

__all__ = ["DecisionTree"]


def __getattr__(name: str):
    """Lazy imports — only load modules when actually accessed."""
    if name == "DecisionTree":
        from src.models.decision_tree import DecisionTree
        return DecisionTree

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
