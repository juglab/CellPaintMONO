"""
Plotting functions for visualizing PE matrices and comparisons.
Uses matplotlib and seaborn.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import Optional, Tuple
from matplotlib.figure import Figure
from matplotlib.axes import Axes


def plot_pe_heatmap(
    pe_matrix,
    features,
    samples,
    title="Percentage Error Heatmap",
    fixed_vmin=-100,
    fixed_vmax=100,
):
    """
    Plot percentage error heatmap with fixed color scale and outlier detection.

    Parameters
    ----------
    pe_matrix : np.ndarray
        Percentage error matrix (features × samples)
    features : list
        Feature names (row labels)
    samples : list
        Sample IDs (column labels)
    title : str
        Plot title
    fixed_vmin, fixed_vmax : float
        Fixed color scale limits (default: -100 to +100%)

    Returns
    -------
    fig, ax : matplotlib figure and axes
    """
    fig_height = max(20, len(features) * 0.15)
    fig_width = max(12, len(samples) * 0.6)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    im = ax.imshow(
        pe_matrix,
        aspect="auto",
        cmap="RdBu_r",
        vmin=fixed_vmin,
        vmax=fixed_vmax,
        interpolation="nearest",
    )

    cbar = plt.colorbar(im, ax=ax, pad=0.01)
    cbar.set_label("Percentage Error (%)", fontsize=12, fontweight="bold")
    cbar.ax.tick_params(labelsize=10)

    ax.set_xticks(np.arange(len(samples)))
    ax.set_xticklabels(samples, fontsize=10, rotation=90, ha="right")
    ax.set_yticks(np.arange(len(features)))
    ax.set_yticklabels(features, fontsize=6)

    ax.set_xticks(np.arange(len(samples)) - 0.5, minor=True)
    ax.set_yticks(np.arange(len(features)) - 0.5, minor=True)
    ax.grid(which="minor", color="gray", linestyle="-", linewidth=0.3, alpha=0.3)

    ax.set_xlabel("Sample", fontsize=13, fontweight="bold")
    ax.set_ylabel("Feature", fontsize=13, fontweight="bold")
    ax.set_title(
        f"{title}\n(Color scale: {fixed_vmin}% to {fixed_vmax}%)",
        fontsize=14,
        fontweight="bold",
        pad=15,
    )

    plt.tight_layout()

    outlier_threshold = max(abs(fixed_vmin), abs(fixed_vmax))
    outliers = []

    for i, feature in enumerate(features):
        for j, sample in enumerate(samples):
            value = pe_matrix[i, j]
            if not np.isnan(value) and abs(value) > outlier_threshold:
                outliers.append({"feature": feature, "sample": sample, "PE": value})

    if outliers:
        outliers_sorted = sorted(outliers, key=lambda x: abs(x["PE"]), reverse=True)
        pe_values = [o["PE"] for o in outliers_sorted]
        print(f"OUTLIERS: {len(outliers)} values with |PE| > {outlier_threshold}%")
        print(f"{'Feature':<50} {'Sample':<15} {'PE (%)':<10}")
        for outlier in outliers_sorted:
            print(
                f"{outlier['feature']:<50} {outlier['sample']:<15} {outlier['PE']:>9.2f}"
            )
        print(f"\nOutliers > 100%:  {max(pe_values):>8.2f}%")
        print(f"Outliers < -100%: {min(pe_values):>8.2f}%")
        print(f"Mean |PE| (outliers):    {np.mean(np.abs(pe_values)):>8.2f}%")

    return fig, ax


def plot_best_worst_comparison(
    pe_matrix: np.ndarray,
    feature_names: list,
    sample_names: list,
    pe_stats_df: pd.DataFrame,
    n_features: int = 10,
    title_prefix: str = "",
    figsize: Tuple[float, float] = (18, 9),
    vmin: float = -100.0,
    vmax: float = 100.0,
) -> Tuple[Figure, Tuple[Axes, Axes]]:
    """
    Side-by-side heatmap: Top N best vs worst features.

    Used for PowerPoint-ready slides (16:9 or similar aspect ratio).
    Fixed color scale with outlier tracking.
    """
    # Get best and worst features
    best_features = pe_stats_df.nsmallest(n_features, "Mean_Abs_PE_%")
    worst_features = pe_stats_df.nlargest(n_features, "Mean_Abs_PE_%")

    # Get indices
    best_indices = [feature_names.index(f) for f in best_features["Feature"].values]
    worst_indices = [feature_names.index(f) for f in worst_features["Feature"].values]

    # Extract submatrices
    pe_matrix_best = pe_matrix[best_indices, :]
    pe_matrix_worst = pe_matrix[worst_indices, :]

    best_feature_names = best_features["Feature"].values
    worst_feature_names = worst_features["Feature"].values

    # Create figure
    fig, (ax_best, ax_worst) = plt.subplots(1, 2, figsize=figsize)

    # LEFT: Best features
    im_best = ax_best.imshow(
        pe_matrix_best,
        aspect="auto",
        cmap="RdBu_r",
        vmin=vmin,
        vmax=vmax,
        interpolation="nearest",
    )

    # Annotate best
    for i in range(n_features):
        for j in range(len(sample_names)):
            value = pe_matrix_best[i, j]
            if not np.isnan(value):
                # Highlight outliers
                if abs(value) > 100:
                    text_color = "yellow"
                    text_weight = "bold"
                else:
                    text_color = "white" if abs(value) > abs(vmax) * 0.6 else "black"
                    text_weight = "normal"

                ax_best.text(
                    j,
                    i,
                    f"{value:.0f}",
                    ha="center",
                    va="center",
                    fontsize=7,
                    color=text_color,
                    fontweight=text_weight,
                )

    # Configure best axis
    ax_best.set_xticks(range(len(sample_names)))
    ax_best.set_xticklabels(sample_names, rotation=90, ha="right", fontsize=9)
    ax_best.set_yticks(range(n_features))
    ax_best.set_yticklabels(
        [f.split("_", 2)[-1][:50] for f in best_feature_names], fontsize=8
    )
    ax_best.set_xlabel("Sample", fontsize=11, fontweight="bold")
    ax_best.set_ylabel("Feature", fontsize=11, fontweight="bold")

    mean_abs_pe_best = best_features["Mean_Abs_PE_%"].mean()
    ax_best.set_title(
        f"Top {n_features} Best Features\nMean |PE| = {mean_abs_pe_best:.1f}%",
        fontsize=12,
        fontweight="bold",
        pad=10,
    )

    # Grid for best
    ax_best.set_xticks(np.arange(len(sample_names) + 1) - 0.5, minor=True)
    ax_best.set_yticks(np.arange(n_features + 1) - 0.5, minor=True)
    ax_best.grid(which="minor", color="black", linestyle="-", linewidth=1.2)
    ax_best.tick_params(which="minor", size=0)

    # RIGHT: Worst features
    ax_worst.imshow(
        pe_matrix_worst,
        aspect="auto",
        cmap="RdBu_r",
        vmin=vmin,
        vmax=vmax,
        interpolation="nearest",
    )

    # Annotate worst
    for i in range(n_features):
        for j in range(len(sample_names)):
            value = pe_matrix_worst[i, j]
            if not np.isnan(value):
                # Highlight outliers
                if abs(value) > 100:
                    text_color = "yellow"
                    text_weight = "bold"
                else:
                    text_color = "white" if abs(value) > abs(vmax) * 0.6 else "black"
                    text_weight = "normal"

                ax_worst.text(
                    j,
                    i,
                    f"{value:.0f}",
                    ha="center",
                    va="center",
                    fontsize=7,
                    color=text_color,
                    fontweight=text_weight,
                )

    # Configure worst axis
    ax_worst.set_xticks(range(len(sample_names)))
    ax_worst.set_xticklabels(sample_names, rotation=90, ha="right", fontsize=9)
    ax_worst.set_yticks(range(n_features))
    ax_worst.set_yticklabels(
        [f.split("_", 2)[-1][:50] for f in worst_feature_names], fontsize=8
    )
    ax_worst.set_xlabel("Sample", fontsize=11, fontweight="bold")
    ax_worst.set_ylabel("Feature", fontsize=11, fontweight="bold")

    mean_abs_pe_worst = worst_features["Mean_Abs_PE_%"].mean()
    ax_worst.set_title(
        f"Top {n_features} Worst Features\nMean |PE| = {mean_abs_pe_worst:.1f}%",
        fontsize=12,
        fontweight="bold",
        pad=10,
    )

    # Grid for worst
    ax_worst.set_xticks(np.arange(len(sample_names) + 1) - 0.5, minor=True)
    ax_worst.set_yticks(np.arange(n_features + 1) - 0.5, minor=True)
    ax_worst.grid(which="minor", color="black", linestyle="-", linewidth=1.2)
    ax_worst.tick_params(which="minor", size=0)

    # Shared colorbar
    fig.subplots_adjust(right=0.92, wspace=0.35)
    cbar_ax = fig.add_axes([0.94, 0.15, 0.015, 0.7])
    cbar = fig.colorbar(im_best, cax=cbar_ax)
    cbar.set_label("Percentage Error (%)", fontsize=11, fontweight="bold")
    cbar.ax.tick_params(labelsize=10)

    # Super title
    if title_prefix:
        fig.suptitle(
            f"{title_prefix}\nBest vs Worst Feature Performance",
            fontsize=14,
            fontweight="bold",
            y=0.98,
        )

    return fig, (ax_best, ax_worst)


def plot_channel_comparison(
    comparison_df: pd.DataFrame, figsize: Tuple[float, float] = (16, 12)
) -> Figure:
    """
    Create 4-panel comparison figure for multiple channels.

    Panels:
    1. Mean Absolute PE by channel (bar chart)
    2. Pearson correlation by channel (bar chart)
    3. PE distribution by channel (violin/box plot)
    4. Feature count by compartment (grouped bar chart)
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)

    channels = comparison_df["Channel"].values
    n_channels = len(channels)
    x = np.arange(n_channels)

    # Panel 1: Mean Absolute PE
    ax1 = axes[0, 0]
    bars = ax1.bar(
        x,
        comparison_df["Mean_Abs_PE_%"],
        color="#4472C4",
        alpha=0.8,
        edgecolor="black",
        linewidth=1,
    )
    ax1.set_ylabel("Mean |PE| (%)", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Biological Channel", fontsize=12, fontweight="bold")
    ax1.set_title(
        "Mean Absolute Percentage Error\nby Channel", fontsize=13, fontweight="bold"
    )
    ax1.set_xticks(x)
    ax1.set_xticklabels(channels)
    ax1.grid(axis="y", alpha=0.3)

    # Add value labels on bars
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax1.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{height:.1f}%",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )

    # Panel 2: Pearson correlation
    ax2 = axes[0, 1]
    bars = ax2.bar(
        x,
        comparison_df["Pearson_r"],
        color="#70AD47",
        alpha=0.8,
        edgecolor="black",
        linewidth=1,
    )
    ax2.set_ylabel("Pearson r", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Biological Channel", fontsize=12, fontweight="bold")
    ax2.set_title(
        "Mean Pearson Correlation\nby Channel", fontsize=13, fontweight="bold"
    )
    ax2.set_xticks(x)
    ax2.set_xticklabels(channels)
    ax2.set_ylim([0.9, 1.0])
    ax2.grid(axis="y", alpha=0.3)

    # Add value labels
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax2.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{height:.4f}",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )

    # Panel 3: PE distribution
    ax3 = axes[1, 0]
    data_for_plot = []
    labels_for_plot = []
    for _, row in comparison_df.iterrows():
        data_for_plot.append([row["Mean_PE_%"], row["Median_PE_%"]])
        labels_for_plot.append(row["Channel"])

    bp = ax3.boxplot(
        [comparison_df["Mean_PE_%"].values, comparison_df["Median_PE_%"].values],
        positions=[0, 1],
        labels=["Mean PE", "Median PE"],
        patch_artist=True,
        showmeans=True,
    )

    for patch in bp["boxes"]:
        patch.set_facecolor("#FFC000")
        patch.set_alpha(0.7)

    ax3.axhline(y=0, color="red", linestyle="--", linewidth=1.5, alpha=0.7)
    ax3.set_ylabel("Percentage Error (%)", fontsize=12, fontweight="bold")
    ax3.set_title("PE Distribution Across Channels", fontsize=13, fontweight="bold")
    ax3.grid(axis="y", alpha=0.3)

    # Panel 4: Compartment distribution
    ax4 = axes[1, 1]
    width = 0.25
    ax4.bar(
        x - width,
        comparison_df["Cells"],
        width,
        label="Cells",
        color="#2E8B57",
        alpha=0.8,
        edgecolor="black",
        linewidth=1,
    )
    ax4.bar(
        x,
        comparison_df["Cytoplasm"],
        width,
        label="Cytoplasm",
        color="#4682B4",
        alpha=0.8,
        edgecolor="black",
        linewidth=1,
    )
    ax4.bar(
        x + width,
        comparison_df["Nuclei"],
        width,
        label="Nuclei",
        color="#DC143C",
        alpha=0.8,
        edgecolor="black",
        linewidth=1,
    )

    ax4.set_ylabel("Number of Features", fontsize=12, fontweight="bold")
    ax4.set_xlabel("Biological Channel", fontsize=12, fontweight="bold")
    ax4.set_title("Feature Distribution by Compartment", fontsize=13, fontweight="bold")
    ax4.set_xticks(x)
    ax4.set_xticklabels(channels)
    ax4.legend(fontsize=10)
    ax4.grid(axis="y", alpha=0.3)

    plt.tight_layout()

    return fig


def plot_correlation_scatter(
    target_values: np.ndarray,
    predicted_values: np.ndarray,
    sample_name: str = "",
    figsize: Tuple[float, float] = (8, 8),
) -> Tuple[Figure, Axes]:
    """
    Scatter plot: target vs predicted values with regression line.
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Remove NaN values
    mask = ~(np.isnan(target_values) | np.isnan(predicted_values))
    target_clean = target_values[mask]
    pred_clean = predicted_values[mask]

    # Scatter plot
    ax.scatter(
        target_clean, pred_clean, alpha=0.6, s=50, edgecolors="black", linewidths=0.5
    )

    # Diagonal line (perfect prediction)
    min_val = min(target_clean.min(), pred_clean.min())
    max_val = max(target_clean.max(), pred_clean.max())
    ax.plot(
        [min_val, max_val],
        [min_val, max_val],
        "r--",
        linewidth=2,
        label="Perfect prediction",
        alpha=0.7,
    )

    # Regression line
    z = np.polyfit(target_clean, pred_clean, 1)
    p = np.poly1d(z)
    ax.plot(
        target_clean,
        p(target_clean),
        "b-",
        linewidth=2,
        label=f"Fit: y={z[0]:.2f}x+{z[1]:.2f}",
        alpha=0.7,
    )

    # Calculate Pearson r
    from scipy.stats import pearsonr

    r, p_val = pearsonr(target_clean, pred_clean)

    ax.set_xlabel("Target Values", fontsize=12, fontweight="bold")
    ax.set_ylabel("Predicted Values", fontsize=12, fontweight="bold")

    title = "Target vs Predicted"
    if sample_name:
        title += f"\n{sample_name}"
    title += f"\nPearson r = {r:.4f} (p = {p_val:.2e})"
    ax.set_title(title, fontsize=13, fontweight="bold")

    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)

    plt.tight_layout()

    return fig, ax


def save_figure(
    fig: Figure,
    filepath: str,
    dpi: int = 300,
    bbox_inches: str = "tight",
    facecolor: str = "white",
):
    """
    Save figure with standard settings.
    """
    fig.savefig(filepath, dpi=dpi, bbox_inches=bbox_inches, facecolor=facecolor)


def plot_stacked_raw_values(
    target_matrix: pd.DataFrame,
    predicted_matrix: pd.DataFrame,
    feature_names: list,
    sample_names: list,
    title: str = "Stacked Feature Maps: Target vs Predicted",
    figsize: Optional[Tuple[float, float]] = None,
) -> Figure:
    """
    Stacked heatmap showing raw feature values: Target (top) vs Predicted (bottom).

    Uses viridis colormap for raw values.
    Shared colorbar on right side.
    """
    n_features = len(feature_names)
    n_samples = len(sample_names)

    # Auto-calculate figure size if not provided
    if figsize is None:
        fig_width = max(16, n_samples * 0.6)
        fig_height = 14
        figsize = (fig_width, fig_height)

    # Get data as arrays
    target_vals = target_matrix[feature_names].T.values
    pred_vals = predicted_matrix[feature_names].T.values

    # Shared color scale
    vmin = np.nanpercentile(
        np.concatenate([target_vals.flatten(), pred_vals.flatten()]), 1
    )
    vmax = np.nanpercentile(
        np.concatenate([target_vals.flatten(), pred_vals.flatten()]), 99
    )

    # Create figure
    fig, axes = plt.subplots(2, 1, figsize=figsize, sharex=True, sharey=True)

    # Top: Target
    im1 = axes[0].imshow(
        target_vals,
        aspect="auto",
        cmap="viridis",
        vmin=vmin,
        vmax=vmax,
        interpolation="nearest",
    )

    axes[0].set_yticks(np.arange(n_features))
    axes[0].set_yticklabels(
        [f.split("_", 1)[1][:60] if "_" in f else f[:60] for f in feature_names],
        fontsize=9,
    )
    axes[0].set_title("Target", fontsize=14, fontweight="bold", pad=10)
    axes[0].set_ylabel("Feature", fontsize=12, fontweight="bold")

    # Bottom: Predicted
    axes[1].imshow(
        pred_vals,
        aspect="auto",
        cmap="viridis",
        vmin=vmin,
        vmax=vmax,
        interpolation="nearest",
    )

    axes[1].set_yticks(np.arange(n_features))
    axes[1].set_yticklabels(
        [f.split("_", 1)[1][:60] if "_" in f else f[:60] for f in feature_names],
        fontsize=9,
    )
    axes[1].set_xticks(np.arange(n_samples))
    axes[1].set_xticklabels(sample_names, fontsize=9, rotation=90, ha="right")
    axes[1].set_title("MicroSPLIT-Predicted", fontsize=14, fontweight="bold", pad=10)
    axes[1].set_ylabel("Feature", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Sample", fontsize=12, fontweight="bold")

    # Shared colorbar
    fig.subplots_adjust(right=0.92)
    cbar_ax = fig.add_axes([0.94, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(im1, cax=cbar_ax)
    cbar.set_label("Raw Feature Value", fontsize=12, fontweight="bold")

    # Super title
    plt.suptitle(title, fontsize=16, fontweight="bold", y=0.995)

    return fig


def plot_interleaved_raw_values(
    target_matrix: pd.DataFrame,
    predicted_matrix: pd.DataFrame,
    feature_names: list,
    sample_names: list,
    title: str = "Unified Feature Map: Target vs Predicted",
    figsize: Optional[Tuple[float, float]] = None,
) -> Figure:
    """
    Interleaved heatmap: alternating Target/Predicted columns for each sample.

    Layout: [Target_S1, Pred_S1, Target_S2, Pred_S2, ...]
    White vertical lines separate sample pairs.
    Uses viridis colormap for raw values.
    """
    n_features = len(feature_names)
    n_samples = len(sample_names)

    # Build interleaved matrix
    unified_matrix = []
    unified_labels = []

    for sample in sample_names:
        unified_matrix.append(target_matrix.loc[sample, feature_names].values)
        unified_labels.append(f"{sample}\nTarget")
        unified_matrix.append(predicted_matrix.loc[sample, feature_names].values)
        unified_labels.append(f"{sample}\nPred")

    unified_matrix = np.array(unified_matrix).T  # Shape: (n_features, n_samples*2)

    # Color scale
    vmin = np.nanpercentile(unified_matrix, 1)
    vmax = np.nanpercentile(unified_matrix, 99)

    # Auto-calculate figure size if not provided
    if figsize is None:
        fig_width = max(20, n_samples * 0.8)
        fig_height = 12
        figsize = (fig_width, fig_height)

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    im = ax.imshow(
        unified_matrix,
        aspect="auto",
        cmap="viridis",
        vmin=vmin,
        vmax=vmax,
        interpolation="nearest",
    )

    # Colorbar
    cbar = plt.colorbar(im, ax=ax, pad=0.01)
    cbar.set_label("Raw Feature Value", fontsize=12, fontweight="bold")

    # X-axis
    ax.set_xticks(np.arange(len(unified_labels)))
    ax.set_xticklabels(unified_labels, fontsize=8, ha="center")

    # Y-axis
    ax.set_yticks(np.arange(n_features))
    ax.set_yticklabels(
        [f.split("_", 1)[1][:60] if "_" in f else f[:60] for f in feature_names],
        fontsize=8,
    )

    # Vertical separators between sample pairs
    for i in range(1, n_samples):
        ax.axvline(x=i * 2 - 0.5, color="white", linewidth=2, linestyle="-")

    # Grid
    ax.set_xticks(np.arange(len(unified_labels)) - 0.5, minor=True)
    ax.set_yticks(np.arange(n_features) - 0.5, minor=True)
    ax.grid(which="minor", color="gray", linestyle="-", linewidth=0.3, alpha=0.3)

    # Labels
    ax.set_xlabel("Sample (Target | Predicted pairs)", fontsize=13, fontweight="bold")
    ax.set_ylabel("Feature", fontsize=13, fontweight="bold")
    ax.set_title(title, fontsize=15, fontweight="bold", pad=20)

    plt.setp(ax.get_xticklabels(), rotation=90, ha="center")
    plt.tight_layout()

    return fig
