"""Utility helpers subpackage."""

__all__ = ["load_config", "set_seed", "timer", "shuffle_data", "setup_logging"]


def __getattr__(name: str):
    """Lazy imports — only load modules when actually accessed."""
    if name in __all__:
        from src.utils import helpers as _h
        return getattr(_h, name)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
