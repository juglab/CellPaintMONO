"""
Profile aggregation using pycytominer.
Transforms single-cell CellProfiler data into well-level profiles.
"""

import pandas as pd
from pathlib import Path
from typing import List, Optional, Literal
import pycytominer


def aggregate_profiles(
    df_cells: pd.DataFrame,
    df_image: Optional[pd.DataFrame] = None,
    strata: List[str] = ["Metadata_Plate", "Metadata_Well"],
    features: Optional[List[str]] = None,
    operation: Literal["median", "mean"] = "median",
    output_file: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Aggregate single-cell data to well-level profiles.

    Parameters
    ----------
    df_cells : pd.DataFrame
        Single-cell data from CellProfiler
    df_image : pd.DataFrame, optional
        Image-level metadata to merge
    strata : list
        Columns to group by for aggregation
    features : list, optional
        Specific features to aggregate. If None, uses all numeric
    operation : str
        Aggregation operation ('median' or 'mean')
    output_file : Path, optional
        Where to save aggregated profiles

    Returns
    -------
    pd.DataFrame
        Aggregated profiles
    """
    # Aggregate
    profiles = pycytominer.aggregate(
        population_df=df_cells, strata=strata, features=features, operation=operation
    )

    # Merge with image metadata
    if df_image is not None:
        metadata_cols = [col for col in df_image.columns if col.startswith("Metadata_")]
        merge_cols = [col for col in strata if col in metadata_cols]

        if merge_cols:
            profiles = profiles.merge(
                df_image[metadata_cols].drop_duplicates(subset=merge_cols),
                on=merge_cols,
                how="left",
            )
    # save outputs
    if output_file is not None:
        output_file.parent.mkdir(parents=True, exist_ok=True)
        profiles.to_csv(output_file, index=False)

    return profiles


def normalize_profiles(
    profiles: pd.DataFrame,
    method: Literal["standardize", "robustize", "mad_robustize"] = "mad_robustize",
    samples: str = "Metadata_ImageType == 'target'",
    output_file: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Normalize profiles using pycytominer.

    Parameters
    ----------
    profiles : pd.DataFrame
        Aggregated profiles
    method : str
        Normalization method
    samples : str
        Query string to select control samples for normalization
    output_file : Path, optional
        Where to save normalized profiles

    Returns
    -------
    pd.DataFrame
        Normalized profiles
    """
    normalized = pycytominer.normalize(
        profiles=profiles, method=method, samples=samples
    )

    if output_file is not None:
        output_file.parent.mkdir(parents=True, exist_ok=True)
        normalized.to_csv(output_file, index=False)

    return normalized


def feature_select(
    profiles: pd.DataFrame,
    operation: List[str] = ["variance_threshold", "correlation_threshold"],
    output_file: Optional[Path] = None,
    **kwargs,
) -> pd.DataFrame:
    """
    Select features using pycytominer.

    Parameters
    ----------
    profiles : pd.DataFrame
        Normalized profiles
    operation : list
        Feature selection operations to apply
    output_file : Path, optional
        Where to save feature-selected profiles
    **kwargs
        Additional arguments for pycytominer.feature_select

    Returns
    -------
    pd.DataFrame
        Feature-selected profiles
    """
    selected = pycytominer.feature_select(
        profiles=profiles, operation=operation, **kwargs
    )

    if output_file is not None:
        output_file.parent.mkdir(parents=True, exist_ok=True)
        selected.to_csv(output_file, index=False)

    return selected


def create_profiles_pipeline(
    df_cells: pd.DataFrame,
    df_image: Optional[pd.DataFrame] = None,
    output_dir: Optional[Path] = None,
    strata: List[str] = ["Metadata_Plate", "Metadata_Well"],
    aggregation_op: Literal["median", "mean"] = "median",
    normalization_method: str = "mad_robustize",
    normalization_samples: str = "Metadata_ImageType == 'target'",
) -> pd.DataFrame:
    """
    Complete profile generation pipeline.

    Executes: Aggregate → Normalize → Feature Select

    Parameters
    ----------
    df_cells : pd.DataFrame
        Single-cell data
    df_image : pd.DataFrame, optional
        Image metadata
    output_dir : Path, optional
        Directory to save intermediate and final outputs
    strata : list
        Aggregation grouping columns
    aggregation_op : str
        Aggregation operation
    normalization_method : str
        Normalization method
    normalization_samples : str
        Query for control samples

    Returns
    -------
    pd.DataFrame
        Final processed profiles
    """
    # Step 1: Aggregate
    profiles = aggregate_profiles(
        df_cells=df_cells,
        df_image=df_image,
        strata=strata,
        operation=aggregation_op,
        output_file=output_dir / "01_aggregated.csv" if output_dir else None,
    )
    # Step 2: Normalize
    normalized = normalize_profiles(
        profiles=profiles,
        method=normalization_method,
        samples=normalization_samples,
        output_file=output_dir / "02_normalized.csv" if output_dir else None,
    )

    # Step 3: Feature selection
    selected = feature_select(
        profiles=normalized,
        operation=["variance_threshold", "correlation_threshold"],
        output_file=output_dir / "03_feature_selected.csv" if output_dir else None,
    )
    print(
        f"  Final profiles: {len(selected)} samples × {len(selected.columns)} features"
    )
    return selected
