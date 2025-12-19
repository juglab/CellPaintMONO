"""CellPaintMONO - Tools for analyzing Cell Painting data."""

__version__ = "0.1.0"

# Data loading
from .dataloading import (
    load_image_level_data,
    load_object_level_data,
    create_sample_ids,
)

# Preprocessing
from .preprocessing import (
    identify_metadata_columns,
    identify_numeric_features,
    apply_standard_filtering,
    select_features,
    separate_target_predicted,
    align_samples,
)

# Comparison
from .comparison import (
    calculate_percentage_error_matrix,
    calculate_pe_statistics_per_feature,
    calculate_pe_statistics_per_sample,
    calculate_pearson_per_sample,
    rank_features_by_pe,
)

# Plotting
from .plotting import (
    plot_pe_heatmap,
    plot_best_worst_comparison,
    plot_channel_comparison,
    save_figure,
)

__all__ = [
    "__version__",
    # Data loading
    "load_image_level_data",
    "load_object_level_data",
    "create_sample_ids",
    # Preprocessing
    "identify_metadata_columns",
    "identify_numeric_features",
    "apply_standard_filtering",
    "select_features",
    "separate_target_predicted",
    "align_samples",
    # Comparison
    "calculate_percentage_error_matrix",
    "calculate_pe_statistics_per_feature",
    "calculate_pe_statistics_per_sample",
    "calculate_pearson_per_sample",
    "rank_features_by_pe",
    # Plotting
    "plot_pe_heatmap",
    "plot_best_worst_comparison",
    "plot_channel_comparison",
    "save_figure",
]
