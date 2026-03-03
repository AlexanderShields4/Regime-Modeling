"""Regime-Modeling: Adaptive regime-switching investment strategy using HMM."""

import logging


def setup_logging(level=logging.INFO):
    """Configure logging to produce output identical to print()."""
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(message)s"))

    root = logging.getLogger("regime_modeling")
    if not root.handlers:
        root.addHandler(handler)
    root.setLevel(level)
