"""Stage 07 scientific intelligence, explainability, and comparison services."""

from src.intelligence.comparison import compare_ann01_ann02
from src.intelligence.dataset import build_dataset_intelligence
from src.intelligence.explainability import (
    explain_ann01_local,
    explain_ann02_signal,
    permutation_importance_ann01,
)

__all__ = [
    "build_dataset_intelligence",
    "compare_ann01_ann02",
    "explain_ann01_local",
    "explain_ann02_signal",
    "permutation_importance_ann01",
]
