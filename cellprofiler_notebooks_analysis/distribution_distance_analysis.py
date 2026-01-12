"""
Distribution Distance Analysis: KL Divergence and Wasserstein Distance
For CRISPR Reproducibility Notebook

This module computes KL divergence and Wasserstein (Earth Mover's) distance
between target and predicted distributions, on both absolute and
percentile-normalized values.
"""

import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


def percentile_normalize(data, lower_percentile=10, upper_percentile=90):
    """
    Normalize data so that lower_percentile maps to 0 and upper_percentile maps to 1.

    Parameters:
    -----------
    data : array-like
        Input data
    lower_percentile : float
        Lower percentile (default 10)
    upper_percentile : float
        Upper percentile (default 90)

    Returns:
    --------
    normalized_data : np.ndarray
        Percentile-normalized data
    """
    data = np.array(data)
    p_low = np.percentile(data, lower_percentile)
    p_high = np.percentile(data, upper_percentile)

    # Avoid division by zero
    if p_high - p_low == 0:
        return np.zeros_like(data)

    # Normalize
    normalized = (data - p_low) / (p_high - p_low)

    return normalized


def compute_kl_divergence(target_vals, predicted_vals, n_bins=50):
    """
    Compute KL divergence between two continuous distributions.
    Uses histogram binning to create discrete probability distributions.

    Parameters:
    -----------
    target_vals : array-like
        Target distribution values
    predicted_vals : array-like
        Predicted distribution values
    n_bins : int
        Number of bins for histogram (default 50)

    Returns:
    --------
    kl_div_value : float
        KL divergence D_KL(P||Q) where P=target, Q=predicted
    """
    # Determine common bin range
    all_vals = np.concatenate([target_vals, predicted_vals])
    bin_range = (all_vals.min(), all_vals.max())

    # Create histograms
    target_hist, bin_edges = np.histogram(
        target_vals, bins=n_bins, range=bin_range, density=True
    )
    predicted_hist, _ = np.histogram(
        predicted_vals, bins=n_bins, range=bin_range, density=True
    )

    # Normalize to get probability distributions
    target_prob = target_hist / target_hist.sum()
    predicted_prob = predicted_hist / predicted_hist.sum()

    # Add small epsilon to avoid log(0) and division by zero
    epsilon = 1e-10
    target_prob = target_prob + epsilon
    predicted_prob = predicted_prob + epsilon

    # Re-normalize after adding epsilon
    target_prob = target_prob / target_prob.sum()
    predicted_prob = predicted_prob / predicted_prob.sum()

    # Compute KL divergence: D_KL(P||Q) = sum(P * log(P/Q))
    kl_divergence = np.sum(target_prob * np.log(target_prob / predicted_prob))

    return kl_divergence


def compute_distribution_distances(target_vals, predicted_vals, n_bins=50):
    """
    Compute both KL divergence and Wasserstein distance between two distributions.

    Parameters:
    -----------
    target_vals : array-like
        Target distribution values
    predicted_vals : array-like
        Predicted distribution values
    n_bins : int
        Number of bins for KL divergence histogram

    Returns:
    --------
    results : dict
        Dictionary with 'kl_divergence' and 'wasserstein_distance' keys
    """
    target_vals = np.array(target_vals).flatten()
    predicted_vals = np.array(predicted_vals).flatten()

    # Remove NaN values
    target_vals = target_vals[~np.isnan(target_vals)]
    predicted_vals = predicted_vals[~np.isnan(predicted_vals)]

    # Compute KL divergence
    kl_div_val = compute_kl_divergence(target_vals, predicted_vals, n_bins=n_bins)

    # Compute Wasserstein distance (directly on continuous data)
    wasserstein_dist = wasserstein_distance(target_vals, predicted_vals)

    return {"kl_divergence": kl_div_val, "wasserstein_distance": wasserstein_dist}


def analyze_feature_distributions(
    target_data,
    predicted_data,
    target_objects,
    predicted_objects,
    experiments,
    features_config,
    n_bins=50,
    lower_percentile=10,
    upper_percentile=90,
):
    """
    Analyze distribution distances for all features across all experiments.
    Computes metrics on both absolute and percentile-normalized values.

    Parameters:
    -----------
    target_data : pd.DataFrame
        Target image-level data
    predicted_data : dict
        Dictionary of predicted image-level data by experiment
    target_objects : dict
        Dictionary of target object-level data by compartment
    predicted_objects : dict
        Nested dict of predicted object-level data by compartment and experiment
    experiments : dict
        Dictionary mapping folder names to experiment names
    features_config : dict
        Feature configuration with type, compartment, actual_col, unit
    n_bins : int
        Number of bins for KL divergence
    lower_percentile : float
        Lower percentile for normalization
    upper_percentile : float
        Upper percentile for normalization

    Returns:
    --------
    results_df : pd.DataFrame
        DataFrame with columns: Experiment, Feature, KL_Div_Absolute,
        Wasserstein_Absolute, KL_Div_Normalized, Wasserstein_Normalized
    """
    results = []

    for exp_name in experiments.values():
        for feature_name, config in features_config.items():
            # Get data based on feature type
            if config["type"] == "image":
                target_vals = target_data[feature_name].values
                pred_vals = predicted_data[exp_name][feature_name].values
            else:
                compartment = config["compartment"]
                actual_col = config["actual_col"]
                target_vals = target_objects[compartment][actual_col].values
                pred_vals = predicted_objects[compartment][exp_name][actual_col].values

            # Compute on absolute values
            abs_distances = compute_distribution_distances(
                target_vals, pred_vals, n_bins=n_bins
            )

            # Percentile normalize
            target_norm = percentile_normalize(
                target_vals, lower_percentile, upper_percentile
            )
            pred_norm = percentile_normalize(
                pred_vals, lower_percentile, upper_percentile
            )

            # Compute on normalized values
            norm_distances = compute_distribution_distances(
                target_norm, pred_norm, n_bins=n_bins
            )

            results.append(
                {
                    "Experiment": exp_name,
                    "Feature": feature_name,
                    "KL_Div_Absolute": abs_distances["kl_divergence"],
                    "Wasserstein_Absolute": abs_distances["wasserstein_distance"],
                    "KL_Div_Normalized": norm_distances["kl_divergence"],
                    "Wasserstein_Normalized": norm_distances["wasserstein_distance"],
                    "Unit": config.get("unit", "unitless"),
                }
            )

    return pd.DataFrame(results)


def create_distance_summary_table(results_df, output_path=None):
    """
    Create a summary table of distribution distances by experiment.

    Parameters:
    -----------
    results_df : pd.DataFrame
        Results from analyze_feature_distributions
    output_path : str or Path, optional
        Path to save the table image

    Returns:
    --------
    summary_df : pd.DataFrame
        Summary statistics by experiment
    """
    summary = (
        results_df.groupby("Experiment")
        .agg(
            {
                "KL_Div_Absolute": ["mean", "std"],
                "Wasserstein_Absolute": ["mean", "std"],
                "KL_Div_Normalized": ["mean", "std"],
                "Wasserstein_Normalized": ["mean", "std"],
            }
        )
        .round(4)
    )

    # Flatten column names
    summary.columns = ["_".join(col).strip() for col in summary.columns.values]
    summary = summary.reset_index()

    if output_path:
        # Create visualization
        fig, ax = plt.subplots(figsize=(14, len(summary) * 0.5 + 2))
        ax.axis("off")

        table = ax.table(
            cellText=summary.values,
            colLabels=summary.columns,
            cellLoc="center",
            loc="center",
            bbox=[0, 0, 1, 1],
        )

        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 2.2)

        # Style the table
        for (i, j), cell in table.get_celld().items():
            cell.set_edgecolor("black")
            cell.set_linewidth(1.5 if i == 0 else 0.8)

            if i == 0:  # Header
                cell.set_facecolor("#2C3E50")
                cell.set_text_props(weight="bold", color="white", fontsize=9)
            else:  # Data rows
                cell.set_facecolor("#ECF0F1" if i % 2 else "white")

        plt.title(
            "Distribution Distance Summary by Experiment",
            fontsize=14,
            fontweight="bold",
            pad=20,
        )
        plt.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
        plt.close()
        print(f"Summary table saved to {output_path}")

    return summary


def create_distance_heatmaps(results_df, output_dir=None):
    """
    Create heatmaps showing distribution distances across features and experiments.

    Parameters:
    -----------
    results_df : pd.DataFrame
        Results from analyze_feature_distributions
    output_dir : str or Path, optional
        Directory to save heatmap images
    """
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)

    metrics = [
        ("KL_Div_Absolute", "KL Divergence (Absolute Values)"),
        ("Wasserstein_Absolute", "Wasserstein Distance (Absolute Values)"),
        ("KL_Div_Normalized", "KL Divergence (Percentile Normalized)"),
        ("Wasserstein_Normalized", "Wasserstein Distance (Percentile Normalized)"),
    ]

    for metric, title in metrics:
        # Pivot data for heatmap
        pivot_df = results_df.pivot(
            index="Feature", columns="Experiment", values=metric
        )

        # Create heatmap
        fig, ax = plt.subplots(figsize=(12, max(8, len(pivot_df) * 0.5)))

        sns.heatmap(
            pivot_df,
            annot=True,
            fmt=".4f",
            cmap="YlOrRd",
            cbar_kws={"label": metric.replace("_", " ")},
            linewidths=0.5,
            ax=ax,
        )

        ax.set_title(title, fontsize=14, fontweight="bold", pad=15)
        ax.set_xlabel("Experiment", fontsize=11)
        ax.set_ylabel("Feature", fontsize=11)

        plt.xticks(rotation=45, ha="right")
        plt.yticks(rotation=0)
        plt.tight_layout()

        if output_dir:
            filename = f"{metric.lower()}_heatmap.png"
            filepath = output_dir / filename
            plt.savefig(filepath, dpi=300, bbox_inches="tight", facecolor="white")
            print(f"Heatmap saved to {filepath}")

        plt.show()
        plt.close()


def create_distance_comparison_plot(results_df, output_path=None):
    """
    Create a comparison plot showing absolute vs normalized distances.

    Parameters:
    -----------
    results_df : pd.DataFrame
        Results from analyze_feature_distributions
    output_path : str or Path, optional
        Path to save the plot
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # KL Divergence comparison
    ax1 = axes[0]
    x_pos = np.arange(len(results_df))
    width = 0.35

    ax1.bar(
        x_pos - width / 2,
        results_df["KL_Div_Absolute"],
        width,
        label="Absolute",
        alpha=0.8,
        color="#3498db",
    )
    ax1.bar(
        x_pos + width / 2,
        results_df["KL_Div_Normalized"],
        width,
        label="Percentile Normalized",
        alpha=0.8,
        color="#e74c3c",
    )

    ax1.set_xlabel("Feature-Experiment Combination", fontsize=11)
    ax1.set_ylabel("KL Divergence", fontsize=11)
    ax1.set_title(
        "KL Divergence: Absolute vs Normalized", fontsize=12, fontweight="bold"
    )
    ax1.legend()
    ax1.grid(axis="y", alpha=0.3)

    # Wasserstein Distance comparison
    ax2 = axes[1]
    ax2.bar(
        x_pos - width / 2,
        results_df["Wasserstein_Absolute"],
        width,
        label="Absolute",
        alpha=0.8,
        color="#3498db",
    )
    ax2.bar(
        x_pos + width / 2,
        results_df["Wasserstein_Normalized"],
        width,
        label="Percentile Normalized",
        alpha=0.8,
        color="#e74c3c",
    )

    ax2.set_xlabel("Feature-Experiment Combination", fontsize=11)
    ax2.set_ylabel("Wasserstein Distance", fontsize=11)
    ax2.set_title(
        "Wasserstein Distance: Absolute vs Normalized", fontsize=12, fontweight="bold"
    )
    ax2.legend()
    ax2.grid(axis="y", alpha=0.3)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
        print(f"Comparison plot saved to {output_path}")

    plt.show()
    plt.close()


def create_detailed_feature_table(results_df, output_path=None):
    """
    Create a detailed table showing all metrics for each feature and experiment.

    Parameters:
    -----------
    results_df : pd.DataFrame
        Results from analyze_feature_distributions
    output_path : str or Path, optional
        Path to save the table image
    """
    # Round for display
    display_df = results_df.copy()
    numeric_cols = [
        "KL_Div_Absolute",
        "Wasserstein_Absolute",
        "KL_Div_Normalized",
        "Wasserstein_Normalized",
    ]
    for col in numeric_cols:
        display_df[col] = display_df[col].round(4)

    if output_path:
        n_rows = len(display_df)
        fig, ax = plt.subplots(figsize=(16, max(8, n_rows * 0.3)))
        ax.axis("off")

        table = ax.table(
            cellText=display_df.values,
            colLabels=display_df.columns,
            cellLoc="center",
            loc="center",
            bbox=[0, 0, 1, 1],
        )

        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1, 1.8)

        # Style the table
        for (i, j), cell in table.get_celld().items():
            cell.set_edgecolor("black")
            cell.set_linewidth(1.5 if i == 0 else 0.5)

            if i == 0:  # Header
                cell.set_facecolor("#2C3E50")
                cell.set_text_props(weight="bold", color="white", fontsize=8)
            else:  # Data rows
                cell.set_facecolor("#ECF0F1" if i % 2 else "white")

        plt.title(
            "Detailed Distribution Distance Analysis",
            fontsize=14,
            fontweight="bold",
            pad=20,
        )
        plt.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
        plt.close()
        print(f"Detailed table saved to {output_path}")

    return display_df


# ============================================================================
# EXAMPLE USAGE CODE (for your notebook)
# ============================================================================


def run_complete_distribution_analysis(
    target_data,
    predicted_data,
    target_objects,
    predicted_objects,
    experiments,
    features_config,
    output_dir="distribution_analysis",
):
    """
    Run complete distribution distance analysis and create all visualizations.

    This is the main function you should call in your notebook.

    Parameters:
    -----------
    target_data : pd.DataFrame
        Target image-level data
    predicted_data : dict
        Dictionary of predicted image-level data by experiment
    target_objects : dict
        Dictionary of target object-level data by compartment
    predicted_objects : dict
        Nested dict of predicted object-level data by compartment and experiment
    experiments : dict
        Dictionary mapping folder names to experiment names
    features_config : dict
        Feature configuration dictionary
    output_dir : str
        Directory to save outputs

    Returns:
    --------
    results_df : pd.DataFrame
        Complete results dataframe
    summary_df : pd.DataFrame
        Summary statistics by experiment
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    print("=" * 80)
    print("DISTRIBUTION DISTANCE ANALYSIS")
    print("=" * 80)
    print("\nComputing KL divergence and Wasserstein distance...")
    print("- On absolute values")
    print("- On percentile-normalized values (10th=0, 90th=1)")
    print()

    # Compute all distances
    results_df = analyze_feature_distributions(
        target_data=target_data,
        predicted_data=predicted_data,
        target_objects=target_objects,
        predicted_objects=predicted_objects,
        experiments=experiments,
        features_config=features_config,
        n_bins=50,
        lower_percentile=10,
        upper_percentile=90,
    )

    print(f"\nAnalyzed {len(results_df)} feature-experiment combinations")
    print()

    # Create summary table
    print("Creating summary table...")
    summary_df = create_distance_summary_table(
        results_df, output_path=output_dir / "distance_summary.png"
    )

    # Create detailed table
    print("\nCreating detailed table...")
    create_detailed_feature_table(
        results_df, output_path=output_dir / "distance_detailed.png"
    )

    # Create heatmaps
    print("\nCreating heatmaps...")
    create_distance_heatmaps(results_df, output_dir=output_dir)

    # Create comparison plot
    print("\nCreating comparison plots...")
    create_distance_comparison_plot(
        results_df, output_path=output_dir / "distance_comparison.png"
    )

    # Save results to CSV
    csv_path = output_dir / "distribution_distances.csv"
    results_df.to_csv(csv_path, index=False)
    print(f"\nResults saved to {csv_path}")

    summary_csv_path = output_dir / "distribution_distances_summary.csv"
    summary_df.to_csv(summary_csv_path, index=False)
    print(f"Summary saved to {summary_csv_path}")

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
    print(f"\nAll outputs saved to: {output_dir}/")

    return results_df, summary_df
