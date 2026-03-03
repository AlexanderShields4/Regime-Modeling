"""Backward-compatibility shim. Use regime_modeling.data.fetcher instead."""
import warnings
warnings.warn(
    "Importing from top-level 'pages_utils' is deprecated. "
    "Use 'from regime_modeling.data.fetcher import ...' instead.",
    DeprecationWarning,
    stacklevel=2,
)
from regime_modeling.data.fetcher import *
