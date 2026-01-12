"""
Mirror violin plot visualization for target vs predicted comparison.

This module creates publication-quality mirror violin plots where:
- Target distribution appears on the LEFT (blue)
- Predicted distribution appears on the RIGHT (magenta)
- Median lines are shown as dashed lines
- Uses KDE (Kernel Density Estimation) for smooth distributions
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import stats
from matplotlib.backends.backend_pdf import PdfPages
from pathlib import Path
from typing import Dict
import pandas as pd


def create_clean_mirror_violin(
    target_vals: np.ndarray,
    pred_vals: np.ndarray,
    width: float = 4,
    height: float = 1.4,
    dpi: int = 300,
) -> np.ndarray:
    """
    Create clean mirror violin plot without mean lines.
    Target on LEFT (blue), Predicted on RIGHT (magenta).

    Args:
        target_vals: Array of target values
        pred_vals: Array of predicted values
        width: Figure width in inches
        height: Figure height in inches
        dpi: Resolution for rendering

    Returns:
        RGB image array of the violin plot

    Example:
        >>> target = np.random.normal(100, 15, 100)
        >>> predicted = np.random.normal(105, 12, 100)
        >>> img = create_clean_mirror_violin(target, predicted)
        >>> plt.imshow(img); plt.axis('off'); plt.show()
    """
    fig, ax = plt.subplots(figsize=(width, height), dpi=dpi)

    # Calculate kernel density estimates with optimal bandwidth
    kde_target = stats.gaussian_kde(target_vals, bw_method="scott")
    kde_pred = stats.gaussian_kde(pred_vals, bw_method="scott")

    # Create smooth evaluation range
    all_vals = np.concatenate([target_vals, pred_vals])
    pad = (all_vals.max() - all_vals.min()) * 0.1
    y_range = np.linspace(all_vals.min() - pad, all_vals.max() + pad, 200)

    # Evaluate KDEs
    density_target = kde_target(y_range)
    density_pred = kde_pred(y_range)

    # Normalize to same scale
    max_density = max(density_target.max(), density_pred.max())
    density_target_norm = density_target / max_density * 0.45
    density_pred_norm = density_pred / max_density * 0.45

    # Plot mirror violin - cleaner colors
    ax.fill_betweenx(
        y_range,
        0,
        -density_target_norm,
        alpha=0.85,
        color="#2E86AB",  # Blue for target
        edgecolor="#1a5276",
        linewidth=0.8,
    )
    ax.fill_betweenx(
        y_range,
        0,
        density_pred_norm,
        alpha=0.85,
        color="#A23B72",  # Magenta for predicted
        edgecolor="#6c1c47",
        linewidth=0.8,
    )

    # Add median lines only (dashed, subtle)
    target_median = np.median(target_vals)
    pred_median = np.median(pred_vals)

    ax.hlines(
        target_median,
        -density_target_norm.max(),
        0,
        colors="#0d3651",
        linewidth=1.2,
        linestyles="dashed",
        alpha=0.7,
    )
    ax.hlines(
        pred_median,
        0,
        density_pred_norm.max(),
        colors="#4a1430",
        linewidth=1.2,
        linestyles="dashed",
        alpha=0.7,
    )

    # Customize axes
    ax.set_xlim(-0.5, 0.5)
    ax.set_xticks([])
    ax.tick_params(axis="y", labelsize=8, length=3, width=0.8)

    # Clean spines
    ax.spines["top"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.8)

    # Center line
    ax.axvline(x=0, color="black", linewidth=1.2, zorder=10, alpha=0.8)

    # Very subtle grid
    ax.grid(axis="y", alpha=0.15, linestyle="-", linewidth=0.4, color="gray")

    # Convert to image
    fig.tight_layout()
    fig.canvas.draw()
    img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)

    return img


def format_unit(unit: str) -> str:
    """
    Format unit for display - return empty string for unitless.

    Args:
        unit: Unit string ('unitless', 'pixels²', 'pixels', 'cells', etc.)

    Returns:
        Formatted unit string

    Example:
        >>> format_unit('pixels²')
        'px²'
        >>> format_unit('unitless')
        ''
    """
    if unit == "unitless":
        return ""
    elif unit == "pixels²":
        return "px²"
    elif unit == "pixels":
        return "px"
    else:
        return unit


def create_publication_violin_table(
    target_data: pd.DataFrame,
    predicted_data: pd.DataFrame,
    target_objects: Dict[str, pd.DataFrame],
    predicted_objects: Dict[str, pd.DataFrame],
    features_config: Dict[str, Dict],
    experiment_name: str,
    output_pdf: Path,
) -> None:
    """
    Create publication-quality table with embedded mirror violin plots.

    This creates a multi-page PDF with one page per experiment, showing:
    - Feature name and unit on the left
    - Summary statistics (mean ± std) in the middle
    - Mirror violin plot on the right

    Args:
        target_data: Target Image.csv dataframe
        predicted_data: Predicted Image.csv dataframe
        target_objects: Dict mapping compartment to target object dataframe
        predicted_objects: Dict mapping compartment to predicted object dataframe
        features_config: Configuration dict for each feature with:
            - 'type': 'image' or 'object'
            - 'unit': Unit string
            - 'compartment': Compartment name (for object features)
            - 'actual_col': Actual column name (for object features)
        experiment_name: Name of experiment for title
        output_pdf: Path to output PDF file

    Example:
        >>> features_config = {
        ...     'Count_Cells': {'type': 'image', 'unit': 'cells'},
        ...     'Nuclei_Area': {
        ...         'type': 'object',
        ...         'compartment': 'Nuclei',
        ...         'actual_col': 'AreaShape_Area',
        ...         'unit': 'pixels²'
        ...     }
        ... }
        >>> create_publication_violin_table(
        ...     target, predicted, target_objs, pred_objs,
        ...     features_config, 'bio-crispr-1', Path('output.pdf')
        ... )
    """
    # Set publication-quality parameters
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Arial", "Helvetica"]
    plt.rcParams["font.size"] = 9
    plt.rcParams["axes.linewidth"] = 0.8

    with PdfPages(output_pdf) as pdf:
        n_features = len(features_config)

        # Calculate layout
        feature_height = 0.12  # Height per feature row
        spacing = 0.02
        top_margin = 0.12
        bottom_margin = 0.08

        fig_height = (
            top_margin + n_features * (feature_height + spacing) + bottom_margin
        )
        fig = plt.figure(figsize=(10, fig_height))

        # Title
        fig.text(
            0.5,
            0.96,
            f"{experiment_name}: Target vs Predicted Comparison",
            ha="center",
            fontsize=14,
            fontweight="bold",
            family="Arial",
        )

        # Column headers
        header_y = 0.92
        fig.text(0.15, header_y, "Feature", fontsize=10, fontweight="bold")
        fig.text(0.35, header_y, "Mean ± Std", fontsize=10, fontweight="bold")
        fig.text(0.70, header_y, "Distribution", fontsize=10, fontweight="bold")

        # Draw each feature row
        for idx, (feature_name, config) in enumerate(features_config.items()):
            # Get data
            if config["type"] == "image":
                target_vals = target_data[feature_name].values
                pred_vals = predicted_data[feature_name].values
            else:  # object
                compartment = config["compartment"]
                actual_col = config["actual_col"]
                target_vals = target_objects[compartment][actual_col].values
                pred_vals = predicted_objects[compartment][actual_col].values

            # Calculate statistics
            target_mean = np.mean(target_vals)
            target_std = np.std(target_vals)
            pred_mean = np.mean(pred_vals)
            pred_std = np.std(pred_vals)

            # Position for this row
            actual_feature_height = feature_height
            y_bottom = 0.88 - idx * (feature_height + spacing)

            # Feature name (left)
            unit_str = format_unit(config["unit"])
            feature_label = (
                f"{feature_name}\n({unit_str})" if unit_str else feature_name
            )
            fig.text(
                0.10,
                y_bottom + actual_feature_height / 2,
                feature_label,
                va="center",
                fontsize=9,
                fontweight="bold",
            )

            # Statistics (middle)
            stats_text = (
                f"Target: {target_mean:.2f} ± {target_std:.2f}\n"
                f"Predicted: {pred_mean:.2f} ± {pred_std:.2f}"
            )
            fig.text(
                0.35,
                y_bottom + actual_feature_height / 2,
                stats_text,
                va="center",
                fontsize=8,
                family="monospace",
            )

            # Violin plot (right)
            violin_img = create_clean_mirror_violin(
                target_vals, pred_vals, width=4, height=1.4
            )
            ax_violin = fig.add_axes(
                [0.52, y_bottom + 0.02, 0.43, actual_feature_height - 0.04]
            )
            ax_violin.imshow(violin_img, aspect="auto")
            ax_violin.axis("off")

            # Separator line
            if idx < n_features - 1:
                sep_y = y_bottom - spacing / 2
                fig.add_artist(
                    plt.Line2D(
                        [0.05, 0.95],
                        [sep_y, sep_y],
                        color="#d5d8dc",
                        linewidth=1.0,
                        transform=fig.transFigure,
                    )
                )

        # Legend at bottom
        legend_y = bottom_margin - 0.02
        target_patch = mpatches.Patch(
            color="#2E86AB", label="Target (left)", alpha=0.85
        )
        pred_patch = mpatches.Patch(
            color="#A23B72", label="Predicted (right)", alpha=0.85
        )

        legend_ax = fig.add_axes([0.1, legend_y, 0.8, 0.02])
        legend_ax.axis("off")
        legend_ax.legend(
            handles=[target_patch, pred_patch],
            loc="center",
            ncol=2,
            frameon=False,
            fontsize=8,
            handlelength=1.5,
        )

        # Caption
        caption = (
            "Mirror violin plots show distribution of values across samples. "
            "Dashed lines indicate medians."
        )
        fig.text(
            0.5,
            legend_y - 0.01,
            caption,
            ha="center",
            fontsize=7,
            style="italic",
            color="#555555",
            family="Arial",
        )

        pdf.savefig(fig, bbox_inches="tight", dpi=300)
        plt.close()


def create_multi_experiment_violin_comparison(
    target_data: pd.DataFrame,
    predicted_data: Dict[str, pd.DataFrame],
    target_objects: Dict[str, pd.DataFrame],
    predicted_objects: Dict[str, Dict[str, pd.DataFrame]],
    features_config: Dict[str, Dict],
    output_pdf: Path,
) -> None:
    """
    Create violin comparison for multiple experiments in a single PDF.

    Args:
        target_data: Target Image.csv dataframe
        predicted_data: Dict of predicted Image.csv dataframes
        target_objects: Dict of target object dataframes
        predicted_objects: Dict of dicts of predicted object dataframes
        features_config: Feature configuration dict
        output_pdf: Path to output PDF

    Example:
        >>> create_multi_experiment_violin_comparison(
        ...     target, predicted_dict, target_objs, pred_objs,
        ...     features_config, Path('all_experiments.pdf')
        ... )
    """
    print("Creating multi-experiment violin comparison PDF...")

    for exp_name in predicted_data.keys():
        print(f"Processing {exp_name}...")

        # Extract data for this experiment
        pred_img = predicted_data[exp_name]
        pred_objs = {
            comp: pred_dict[exp_name] for comp, pred_dict in predicted_objects.items()
        }

        # Create page for this experiment
        create_publication_violin_table(
            target_data,
            pred_img,
            target_objects,
            pred_objs,
            features_config,
            exp_name,
            output_pdf,
        )
