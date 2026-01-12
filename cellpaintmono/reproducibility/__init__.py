"""
Object-level reproducibility analysis for CellProfiler outputs.

This subpackage provides tools for analyzing reproducibility across
multiple experiments using well-aggregated object-level features.
"""

from .data_loading import (
    load_multiple_experiments,
    load_and_aggregate_objects,
    create_sample_id_from_image_csv,
)

from .metrics import (
    calculate_percentage_error,
    calculate_per_feature_mae,
    create_consolidated_metrics_table,
)

from .visualization import create_clean_mirror_violin, create_publication_violin_table

__all__ = [
    "load_multiple_experiments",
    "load_and_aggregate_objects",
    "create_sample_id_from_image_csv",
    "calculate_percentage_error",
    "calculate_per_feature_mae",
    "create_consolidated_metrics_table",
    "create_clean_mirror_violin",
    "create_publication_violin_table",
]
