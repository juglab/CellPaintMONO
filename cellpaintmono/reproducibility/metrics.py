"""
Metrics calculation for multi-experiment comparisons.

This module provides functions to calculate percentage error, absolute error,
and other performance metrics for comparing target vs predicted experiments.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple


def calculate_percentage_error(
    target_vals: np.ndarray, predicted_vals: np.ndarray
) -> np.ndarray:
    """
    Calculate percentage error: PE = ((predicted - target) / target) * 100

    Args:
        target_vals: Target values array
        predicted_vals: Predicted values array

    Returns:
        Array of percentage errors

    Example:
        >>> target = np.array([100, 200, 300])
        >>> predicted = np.array([105, 190, 310])
        >>> pe = calculate_percentage_error(target, predicted)
        >>> # Returns: [5.0, -5.0, 3.33...]
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        pe = ((predicted_vals - target_vals) / target_vals) * 100
        # Handle division by zero
        pe[~np.isfinite(pe)] = np.nan
    return pe


def calculate_feature_metrics(
    target_vals: np.ndarray, predicted_vals: np.ndarray
) -> Dict[str, float]:
    """
    Calculate comprehensive metrics for a single feature.

    Args:
        target_vals: Target values array
        predicted_vals: Predicted values array

    Returns:
        Dictionary with metrics:
        - mean_target: Mean of target values
        - mean_predicted: Mean of predicted values
        - mean_absolute_error: Mean absolute error
        - mean_pe: Mean percentage error
        - abs_mean_pe: Mean absolute percentage error
        - std_pe: Standard deviation of percentage error
        - median_pe: Median percentage error

    Example:
        >>> metrics = calculate_feature_metrics(target, predicted)
        >>> print(f"MAE: {metrics['mean_absolute_error']:.2f}")
    """
    # Basic statistics
    mean_target = np.mean(target_vals)
    mean_predicted = np.mean(predicted_vals)

    # Errors
    absolute_errors = np.abs(predicted_vals - target_vals)
    mean_absolute_error = np.mean(absolute_errors)

    # Percentage errors
    pe = calculate_percentage_error(target_vals, predicted_vals)
    valid_pe = pe[~np.isnan(pe)]

    return {
        "mean_target": mean_target,
        "mean_predicted": mean_predicted,
        "mean_absolute_error": mean_absolute_error,
        "mean_pe": np.mean(valid_pe) if len(valid_pe) > 0 else np.nan,
        "abs_mean_pe": np.mean(np.abs(valid_pe)) if len(valid_pe) > 0 else np.nan,
        "std_pe": np.std(valid_pe) if len(valid_pe) > 0 else np.nan,
        "median_pe": np.median(valid_pe) if len(valid_pe) > 0 else np.nan,
    }


def calculate_per_feature_mae(
    target_df: pd.DataFrame, predicted_df: pd.DataFrame, features: List[str]
) -> Tuple[Dict[str, float], int]:
    """
    Calculate mean absolute percentage error for multiple features.

    Args:
        target_df: Target dataframe (indexed by sample)
        predicted_df: Predicted dataframe (indexed by sample)
        features: List of feature column names

    Returns:
        Tuple of (feature_maes, n_samples) where:
        - feature_maes: Dict mapping feature name to mean absolute PE
        - n_samples: Number of samples used in calculation

    Example:
        >>> features = ['AreaShape_Area', 'AreaShape_FormFactor']
        >>> maes, n = calculate_per_feature_mae(target_df, pred_df, features)
        >>> print(f"Nuclei Area MAE: {maes['AreaShape_Area']:.2f}%")
    """
    # Find common samples
    common_samples = sorted(set(target_df.index) & set(predicted_df.index))

    # Align dataframes
    target_aligned = target_df.loc[common_samples]
    predicted_aligned = predicted_df.loc[common_samples]

    feature_maes = {}
    for feature in features:
        target_vals = target_aligned[feature].values
        pred_vals = predicted_aligned[feature].values

        pe = calculate_percentage_error(target_vals, pred_vals)
        feature_maes[feature] = np.mean(np.abs(pe[~np.isnan(pe)]))

    return feature_maes, len(common_samples)


def calculate_count_feature_metrics(
    target_data: pd.DataFrame, predicted_data: pd.DataFrame, count_features: List[str]
) -> Dict[str, Dict[str, float]]:
    """
    Calculate metrics for count features (Count_Cells, Count_Cytoplasm, etc.).

    Args:
        target_data: Target Image.csv dataframe
        predicted_data: Predicted Image.csv dataframe
        count_features: List of count feature names

    Returns:
        Dictionary mapping feature name to metrics dict

    Example:
        >>> features = ['Count_Cells', 'Count_Cytoplasm', 'Count_Nuclei']
        >>> metrics = calculate_count_feature_metrics(target, pred, features)
        >>> print(f"Cell count PE: {metrics['Count_Cells']['mean_pe']:.2f}%")
    """
    results = {}
    for feature in count_features:
        target_vals = target_data[feature].values
        pred_vals = predicted_data[feature].values

        pe = calculate_percentage_error(target_vals, pred_vals)

        results[feature] = {
            "mean_pe": np.mean(pe),
            "abs_mean_pe": np.mean(np.abs(pe)),
            "std_pe": np.std(pe),
        }

    return results


def calculate_multi_experiment_metrics(
    target_data: pd.DataFrame,
    predicted_data: Dict[str, pd.DataFrame],
    features: List[str],
    feature_type: str = "count",
) -> pd.DataFrame:
    """
    Calculate metrics across multiple experiments for comparison.

    Args:
        target_data: Target dataframe
        predicted_data: Dict mapping experiment name to predicted dataframe
        features: List of features to analyze
        feature_type: 'count' for image-level counts, 'object' for object features

    Returns:
        DataFrame with one row per experiment and metrics columns

    Example:
        >>> results = calculate_multi_experiment_metrics(
        ...     target,
        ...     {'bio-crispr-1': pred1, 'bio-crispr-2': pred2},
        ...     ['Count_Cells'],
        ...     feature_type='count'
        ... )
    """
    results = []

    for exp_name, pred_df in predicted_data.items():
        for feature in features:
            if feature_type == "count":
                metrics = calculate_feature_metrics(
                    target_data[feature].values, pred_df[feature].values
                )
            else:
                # For object-level features, dataframes are already aligned by sample
                metrics = calculate_feature_metrics(
                    target_data[feature].values, pred_df[feature].values
                )

            results.append({"Experiment": exp_name, "Feature": feature, **metrics})

    return pd.DataFrame(results)


def create_consolidated_metrics_table(
    target_data: pd.DataFrame,
    predicted_data: Dict[str, pd.DataFrame],
    target_objects: Dict[str, pd.DataFrame],
    predicted_objects: Dict[str, Dict[str, pd.DataFrame]],
    count_features: List[str],
    object_features: Dict[str, List[str]],
) -> pd.DataFrame:
    """
    Create consolidated metrics table combining image and object features.

    Args:
        target_data: Target Image.csv dataframe
        predicted_data: Dict of predicted Image.csv dataframes
        target_objects: Dict mapping compartment to target object dataframe
        predicted_objects: Dict mapping compartment to dict of predicted dataframes
        count_features: List of count feature names from Image.csv
        object_features: Dict mapping compartment to list of object features

    Returns:
        DataFrame with comprehensive metrics for all experiments and features

    Example:
        >>> object_features = {
        ...     'Nuclei': ['AreaShape_Area', 'AreaShape_FormFactor'],
        ...     'Cells': ['AreaShape_Area']
        ... }
        >>> df = create_consolidated_metrics_table(
        ...     target, predicted, target_objs, pred_objs,
        ...     ['Count_Cells'], object_features
        ... )
    """
    all_results = []

    for exp_name in predicted_data.keys():
        # Count features from Image.csv
        for feature in count_features:
            metrics = calculate_feature_metrics(
                target_data[feature].values, predicted_data[exp_name][feature].values
            )

            all_results.append(
                {
                    "Experiment": exp_name,
                    "Feature": feature,
                    "Compartment": "Image",
                    **metrics,
                }
            )

        # Object-level features
        for compartment, features in object_features.items():
            for feature in features:
                metrics = calculate_feature_metrics(
                    target_objects[compartment][feature].values,
                    predicted_objects[compartment][exp_name][feature].values,
                )

                all_results.append(
                    {
                        "Experiment": exp_name,
                        "Feature": feature,
                        "Compartment": compartment,
                        **metrics,
                    }
                )

    return pd.DataFrame(all_results)


def summarize_by_experiment(
    metrics_df: pd.DataFrame, group_by: str = "Experiment"
) -> pd.DataFrame:
    """
    Summarize metrics by experiment (average across all features).

    Args:
        metrics_df: DataFrame from create_consolidated_metrics_table
        group_by: Column to group by ('Experiment', 'Feature', etc.)

    Returns:
        Summary DataFrame with averaged metrics

    Example:
        >>> summary = summarize_by_experiment(metrics_df)
        >>> print(summary[['Experiment', 'abs_mean_pe']].to_string())
    """
    numeric_cols = [
        "mean_target",
        "mean_predicted",
        "mean_absolute_error",
        "mean_pe",
        "abs_mean_pe",
        "std_pe",
        "median_pe",
    ]

    summary = metrics_df.groupby(group_by)[numeric_cols].mean().reset_index()
    return summary.round(2)
