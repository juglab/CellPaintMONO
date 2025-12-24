"""
Profile-level comparison between original and predicted data.
Metrics for evaluating MicroSplit prediction quality at the profile level.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, Optional
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics.pairwise import cosine_similarity


def match_profiles(
    target_profiles: pd.DataFrame,
    predicted_profiles: pd.DataFrame,
    match_cols: list = ["Metadata_Plate", "Metadata_Well", "Metadata_Site"],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Match target and predicted profiles by metadata.

    Parameters
    ----------
    target_profiles : pd.DataFrame
        Original (target) profiles
    predicted_profiles : pd.DataFrame
        MicroSplit predicted profiles
    match_cols : list
        Columns to use for matching

    Returns
    -------
    tuple
        (matched_target, matched_predicted) with aligned samples
    """
    # Create match key
    target_profiles = target_profiles.copy()
    predicted_profiles = predicted_profiles.copy()

    target_profiles["_match_key"] = (
        target_profiles[match_cols].astype(str).agg("_".join, axis=1)
    )
    predicted_profiles["_match_key"] = (
        predicted_profiles[match_cols].astype(str).agg("_".join, axis=1)
    )

    # Find common samples
    common_keys = set(target_profiles["_match_key"]) & set(
        predicted_profiles["_match_key"]
    )
    print(f"Found {len(common_keys)} matching profiles")

    # Filter and sort
    matched_target = target_profiles[
        target_profiles["_match_key"].isin(common_keys)
    ].sort_values("_match_key")
    matched_pred = predicted_profiles[
        predicted_profiles["_match_key"].isin(common_keys)
    ].sort_values("_match_key")

    # Remove match key
    matched_target = matched_target.drop(columns=["_match_key"])
    matched_pred = matched_pred.drop(columns=["_match_key"])

    return matched_target, matched_pred


def calculate_profile_correlations(
    target_profiles: pd.DataFrame,
    predicted_profiles: pd.DataFrame,
    method: str = "pearson",
) -> pd.DataFrame:
    """
    Calculate correlation between matched target and predicted profiles.

    Parameters
    ----------
    target_profiles : pd.DataFrame
        Matched target profiles
    predicted_profiles : pd.DataFrame
        Matched predicted profiles
    method : str
        'pearson' or 'spearman'

    Returns
    -------
    pd.DataFrame
        Per-sample correlations
    """
    # Get feature columns
    feature_cols = [
        col for col in target_profiles.columns if not col.startswith("Metadata_")
    ]

    # Calculate per-sample correlation
    correlations = []
    metadata_cols = [
        col for col in target_profiles.columns if col.startswith("Metadata_")
    ]

    for idx in range(len(target_profiles)):
        target_vals = target_profiles[feature_cols].iloc[idx].values
        pred_vals = predicted_profiles[feature_cols].iloc[idx].values

        # Remove NaN values
        mask = ~(np.isnan(target_vals) | np.isnan(pred_vals))
        target_clean = target_vals[mask]
        pred_clean = pred_vals[mask]

        if len(target_clean) > 0:
            if method == "pearson":
                corr, pval = pearsonr(target_clean, pred_clean)
            elif method == "spearman":
                corr, pval = spearmanr(target_clean, pred_clean)
            else:
                raise ValueError(f"Unknown method: {method}")
        else:
            corr, pval = np.nan, np.nan

        result = {
            **target_profiles[metadata_cols].iloc[idx].to_dict(),
            f"{method}_correlation": corr,
            f"{method}_pvalue": pval,
        }
        correlations.append(result)

    return pd.DataFrame(correlations)


def calculate_cosine_similarities(
    target_profiles: pd.DataFrame, predicted_profiles: pd.DataFrame
) -> pd.DataFrame:
    """
    Calculate cosine similarity between matched profiles.

    Parameters
    ----------
    target_profiles : pd.DataFrame
        Matched target profiles
    predicted_profiles : pd.DataFrame
        Matched predicted profiles

    Returns
    -------
    pd.DataFrame
        Per-sample cosine similarities
    """
    feature_cols = [
        col for col in target_profiles.columns if not col.startswith("Metadata_")
    ]
    metadata_cols = [
        col for col in target_profiles.columns if col.startswith("Metadata_")
    ]

    target_features = target_profiles[feature_cols].values
    pred_features = predicted_profiles[feature_cols].values

    # Calculate pairwise cosine similarity
    similarities = []
    for idx in range(len(target_features)):
        sim = cosine_similarity(
            target_features[idx : idx + 1], pred_features[idx : idx + 1]
        )[0, 0]

        result = {
            **target_profiles[metadata_cols].iloc[idx].to_dict(),
            "cosine_similarity": sim,
        }
        similarities.append(result)

    return pd.DataFrame(similarities)


def evaluate_profile_quality(
    target_profiles: pd.DataFrame,
    predicted_profiles: pd.DataFrame,
    match_cols: list = ["Metadata_Plate", "Metadata_Well", "Metadata_Site"],
    output_dir: Optional[Path] = None,
) -> Dict[str, pd.DataFrame]:
    """
    Comprehensive profile quality evaluation.

    Parameters
    ----------
    target_profiles : pd.DataFrame
        Original profiles
    predicted_profiles : pd.DataFrame
        MicroSplit predicted profiles
    match_cols : list
        Columns for matching samples
    output_dir : Path, optional
        Directory to save results

    Returns
    -------
    dict
        Dictionary with evaluation results:
        - 'matched_target': Matched target profiles
        - 'matched_predicted': Matched predicted profiles
        - 'pearson': Pearson correlations
        - 'spearman': Spearman correlations
        - 'cosine': Cosine similarities
        - 'summary': Summary statistics
    """
    print("Matching profiles...")
    matched_target, matched_pred = match_profiles(
        target_profiles, predicted_profiles, match_cols
    )

    print("Calculating Pearson correlations...")
    pearson = calculate_profile_correlations(
        matched_target, matched_pred, method="pearson"
    )

    print("Calculating Spearman correlations...")
    spearman = calculate_profile_correlations(
        matched_target, matched_pred, method="spearman"
    )

    print("Calculating cosine similarities...")
    cosine = calculate_cosine_similarities(matched_target, matched_pred)

    # Summary statistics
    summary = pd.DataFrame(
        {
            "metric": [
                "pearson_mean",
                "pearson_median",
                "pearson_std",
                "spearman_mean",
                "spearman_median",
                "spearman_std",
                "cosine_mean",
                "cosine_median",
                "cosine_std",
            ],
            "value": [
                pearson["pearson_correlation"].mean(),
                pearson["pearson_correlation"].median(),
                pearson["pearson_correlation"].std(),
                spearman["spearman_correlation"].mean(),
                spearman["spearman_correlation"].median(),
                spearman["spearman_correlation"].std(),
                cosine["cosine_similarity"].mean(),
                cosine["cosine_similarity"].median(),
                cosine["cosine_similarity"].std(),
            ],
        }
    )
    print(summary.to_string(index=False))

    results = {
        "matched_target": matched_target,
        "matched_predicted": matched_pred,
        "pearson": pearson,
        "spearman": spearman,
        "cosine": cosine,
        "summary": summary,
    }

    # Save results
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        for name, df in results.items():
            df.to_csv(output_dir / f"{name}.csv", index=False)

    return results
