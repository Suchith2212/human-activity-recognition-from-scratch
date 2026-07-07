"""Model implementations subpackage."""

__all__ = ["DecisionTree", "RandomForest"]


def __getattr__(name: str):
    """Lazy imports — only load modules when actually accessed."""
    if name == "DecisionTree":
        from src.models.decision_tree import DecisionTree
        return DecisionTree

    if name == "RandomForest":
        from src.models.random_forest import RandomForest
        return RandomForest

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

