"""Feature fidelity: compare a test CellProfiler run with a reference run.

Four levels, each calibrated against the data itself:

1. Segmentation: are the same objects found (`run_segmentation`)?
2. Same object, same values: per-object Spearman agreement (`run_agreement`).
3. Per image: distribution distances against a shuffle noise floor and a
   well-to-well yardstick, per feature and multivariate (`run_distances`).
4. Per well: profile identity retrieval and distance geometry (`run_profiles`).

`run_spread` adds the loss of cell-to-cell spread per channel.
"""

from .config import FidelityConfig
from .pipeline import (
    FidelityData,
    prepare,
    run_agreement,
    run_distances,
    run_profiles,
    run_segmentation,
    run_spread,
    write_tables,
)

__all__ = [
    "FidelityConfig",
    "FidelityData",
    "prepare",
    "run_agreement",
    "run_distances",
    "run_profiles",
    "run_segmentation",
    "run_spread",
    "write_tables",
]
