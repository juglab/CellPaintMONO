"""
Comparison functions for calculating percentage error and correlations.
"""

import pandas as pd
import numpy as np
from scipy.stats import pearsonr


def calculate_percentage_error_matrix(
    target_matrix: pd.DataFrame, predicted_matrix: pd.DataFrame
) -> np.ndarray:
    """
    Calculate percentage error for each feature-sample pair.

    Formula: PE = ((Predicted - Target) / Target) * 100

    Handles:
    - NaN values (returns NaN)
    - Division by zero when target=0 (returns NaN)

    Returns: numpy array with shape (n_features, n_samples)
    """
    n_features = len(target_matrix.columns)
    n_samples = len(target_matrix.index)

    pe_matrix = np.zeros((n_features, n_samples))

    for i, sample in enumerate(target_matrix.index):
        target_vals = target_matrix.loc[sample].values
        pred_vals = predicted_matrix.loc[sample].values

        for j, (target_val, pred_val) in enumerate(zip(target_vals, pred_vals)):
            if np.isnan(target_val) or np.isnan(pred_val) or target_val == 0:
                pe_matrix[j, i] = np.nan
            else:
                pe_matrix[j, i] = ((pred_val - target_val) / target_val) * 100

    return pe_matrix


def calculate_pe_statistics_per_feature(
    pe_matrix: np.ndarray, feature_names: list
) -> pd.DataFrame:
    """
    Calculate PE statistics for each feature across all samples.

    Returns DataFrame with columns:
    - Feature
    - Mean_PE, Median_PE, Std_PE
    - Mean_Abs_PE (mean absolute PE)
    - Min_PE, Max_PE
    """
    stats = []

    for j, feature in enumerate(feature_names):
        feature_pe = pe_matrix[j, :]
        valid_pe = feature_pe[~np.isnan(feature_pe)]

        if len(valid_pe) > 0:
            stats.append(
                {
                    "Feature": feature,
                    "Mean_PE_%": np.mean(valid_pe),
                    "Median_PE_%": np.median(valid_pe),
                    "Std_PE_%": np.std(valid_pe),
                    "Mean_Abs_PE_%": np.mean(np.abs(valid_pe)),
                    "Min_PE_%": np.min(valid_pe),
                    "Max_PE_%": np.max(valid_pe),
                    "N_Valid": len(valid_pe),
                }
            )

    return pd.DataFrame(stats)


def calculate_pe_statistics_per_sample(
    pe_matrix: np.ndarray, sample_names: list
) -> pd.DataFrame:
    """
    Calculate PE statistics for each sample across all features.

    Returns DataFrame with columns:
    - Sample
    - Mean_PE, Median_PE, Std_PE
    - Mean_Abs_PE, Max_Abs_PE
    """
    stats = []

    for i, sample in enumerate(sample_names):
        sample_pe = pe_matrix[:, i]
        valid_pe = sample_pe[~np.isnan(sample_pe)]

        if len(valid_pe) > 0:
            stats.append(
                {
                    "Sample": sample,
                    "Mean_PE_%": np.mean(valid_pe),
                    "Median_PE_%": np.median(valid_pe),
                    "Std_PE_%": np.std(valid_pe),
                    "Mean_Abs_PE_%": np.mean(np.abs(valid_pe)),
                    "Max_Abs_PE_%": np.max(np.abs(valid_pe)),
                }
            )

    return pd.DataFrame(stats)


def calculate_pearson_per_sample(
    target_matrix: pd.DataFrame, predicted_matrix: pd.DataFrame
) -> pd.DataFrame:
    """
    Calculate Pearson correlation for each sample across all features.

    Returns DataFrame with columns:
    - Sample
    - Pearson_r
    - P_value
    - N_features (number of valid features used)
    """
    correlations = []

    for sample in target_matrix.index:
        target_vals = target_matrix.loc[sample].values
        pred_vals = predicted_matrix.loc[sample].values

        # Remove NaN values
        mask = ~(np.isnan(target_vals) | np.isnan(pred_vals))
        target_clean = target_vals[mask]
        pred_clean = pred_vals[mask]

        if len(target_clean) > 1:
            r, p_value = pearsonr(target_clean, pred_clean)
            correlations.append(
                {
                    "Sample": sample,
                    "Pearson_r": r,
                    "P_value": p_value,
                    "N_features": len(target_clean),
                }
            )

    return pd.DataFrame(correlations)


def rank_features_by_pe(
    pe_stats_df: pd.DataFrame, metric: str = "Mean_Abs_PE_%", ascending: bool = True
) -> pd.DataFrame:
    """
    Rank features by PE metric.

    metric: Column to sort by (e.g., 'Mean_Abs_PE_%', 'Mean_PE_%')
    ascending: True for best (lowest PE), False for worst (highest PE)
    """
    return pe_stats_df.sort_values(by=metric, ascending=ascending).reset_index(
        drop=True
    )


def get_top_n_features(
    pe_stats_df: pd.DataFrame,
    n: int = 10,
    metric: str = "Mean_Abs_PE_%",
    best: bool = True,
) -> pd.DataFrame:
    """
    Get top N best or worst features.

    best: True for lowest PE (best), False for highest PE (worst)
    """
    sorted_df = rank_features_by_pe(pe_stats_df, metric=metric, ascending=best)
    return sorted_df.head(n)


def compare_channels(
    target_matrix: pd.DataFrame,
    predicted_matrix: pd.DataFrame,
    features_by_channel: dict,
) -> pd.DataFrame:
    """
    Compare PE statistics across different channels.

    features_by_channel: dict mapping channel names to feature lists
                        e.g., {'DNA': [...], 'ER': [...], ...}

    Returns DataFrame with one row per channel containing:
    - Channel name
    - N_features
    - Mean_PE, Mean_Abs_PE, Median_PE, Std_PE, Min_PE, Max_PE
    - Pearson_r, Pearson_r_Std
    - Cells, Cytoplasm, Nuclei (feature counts by compartment)
    """
    comparison = []

    for channel, features in features_by_channel.items():
        # Get subset of matrices for this channel
        channel_target = target_matrix[features]
        channel_pred = predicted_matrix[features]

        # Calculate PE
        pe_matrix = calculate_percentage_error_matrix(channel_target, channel_pred)
        pe_valid = pe_matrix[~np.isnan(pe_matrix)]

        # Calculate correlations
        corr_df = calculate_pearson_per_sample(channel_target, channel_pred)

        # Count features by compartment (matching notebook)
        compartment_counts = {
            "Cells": len([f for f in features if "Cells" in f]),
            "Cytoplasm": len([f for f in features if "Cytoplasm" in f]),
            "Nuclei": len([f for f in features if "Nuclei" in f]),
        }

        comparison.append(
            {
                "Channel": channel,
                "N_Features": len(features),
                "N_Samples": len(target_matrix),
                "Mean_PE_%": np.mean(pe_valid),
                "Mean_Abs_PE_%": np.mean(np.abs(pe_valid)),
                "Median_PE_%": np.median(pe_valid),
                "Std_PE_%": np.std(pe_valid),
                "Min_PE_%": np.min(pe_valid),
                "Max_PE_%": np.max(pe_valid),
                "Pearson_r": corr_df["Pearson_r"].mean(),
                "Pearson_r_Std": corr_df["Pearson_r"].std(),
                "Cells": compartment_counts["Cells"],
                "Cytoplasm": compartment_counts["Cytoplasm"],
                "Nuclei": compartment_counts["Nuclei"],
            }
        )

    return pd.DataFrame(comparison)
