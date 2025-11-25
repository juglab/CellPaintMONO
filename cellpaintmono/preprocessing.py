"""
Preprocessing functions for feature filtering and selection.
"""

import pandas as pd
import numpy as np
from typing import List, Optional


def identify_metadata_columns(df: pd.DataFrame) -> List[str]:
    """Return list of metadata column names (columns starting with 'Metadata_')."""
    return [col for col in df.columns if col.startswith("Metadata_")]


def identify_numeric_features(
    df: pd.DataFrame, exclude_metadata: bool = True
) -> List[str]:
    """
    Return list of numeric feature column names.
    For object-level files, also excludes tracking columns (ImageNumber, ObjectNumber).
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    if exclude_metadata:
        # Exclude Metadata_ columns
        numeric_cols = [col for col in numeric_cols if not col.startswith("Metadata_")]
        # Exclude object tracking columns (for object-level files)
        numeric_cols = [
            col
            for col in numeric_cols
            if col not in ["ImageNumber", "ObjectNumber", "Number_Object_Number"]
        ]

    return numeric_cols


def filter_object_metadata_columns(feature_list: List[str]) -> List[str]:
    """
    Remove object-level metadata columns that aren't measurements.
    These appear in Cells/Nuclei/Cytoplasm.csv but not in Image.csv.
    """
    # Columns to exclude
    exclude_patterns = [
        "Parent_",
        "Children_",
        "FileName_",
        "PathName_",
        "ImageNumber",
        "ObjectNumber",
        "Number_Object_Number",
    ]

    return [
        f for f in feature_list if not any(pattern in f for pattern in exclude_patterns)
    ]


def filter_features(
    feature_list: List[str], exclude_keywords: Optional[List[str]] = None
) -> List[str]:
    """
    Remove features containing any of the exclusion keywords.
    Default excludes: ExecutionTime, ModuleError, Height, Width, Scaling, Group, ZBF, Brightfield
    """
    if exclude_keywords is None:
        exclude_keywords = [
            "ExecutionTime",
            "ModuleError",
            "Height",
            "Width",
            "Scaling",
            "Group",
            "ZBF",
            "Brightfield",
        ]

    return [
        f
        for f in feature_list
        if not any(keyword.lower() in f.lower() for keyword in exclude_keywords)
    ]


def remove_z_location_features(feature_list: List[str]) -> List[str]:
    """Remove Z-location features (for 2D images)."""
    return [
        f
        for f in feature_list
        if not ("_Z_" in f or "_Center_Z" in f or f.split("_")[-1] == "Z")
    ]


def remove_children_count_features(feature_list: List[str]) -> List[str]:
    """Remove Children count features (typically constant)."""
    return [f for f in feature_list if not ("Children" in f and "Count" in f)]


def select_by_aggregation(feature_list: List[str], aggregation: str) -> List[str]:
    """
    Select features by aggregation type.
    aggregation: 'Mean', 'Median', or 'StDev'
    """
    return [f for f in feature_list if f.startswith(f"{aggregation}_")]


def select_by_compartment(feature_list: List[str], compartment: str) -> List[str]:
    """
    Select features by compartment.
    compartment: 'Cells', 'Cytoplasm', or 'Nuclei'
    """
    return [f for f in feature_list if f"_{compartment}_" in f]


def select_by_channel(feature_list: List[str], channel: str) -> List[str]:
    """
    Select features by channel.
    channel: 'DNA', 'ER', 'RNA', 'AGP', or 'Mito'
    """
    return [f for f in feature_list if channel in f]


def select_by_measurement(feature_list: List[str], measurement: str) -> List[str]:
    """
    Select features by measurement type.
    measurement: 'Granularity', 'Texture', 'AreaShape', 'Intensity', 'Correlation', etc.
    """
    return [f for f in feature_list if measurement in f]


def select_features(
    feature_list: List[str],
    aggregation: Optional[str] = None,
    compartment: Optional[str] = None,
    channel: Optional[str] = None,
    measurement: Optional[str] = None,
) -> List[str]:
    """
    Select features matching all specified criteria.
    Returns features that satisfy ALL provided criteria (AND logic).
    """
    result = feature_list

    if aggregation is not None:
        result = select_by_aggregation(result, aggregation)

    if compartment is not None:
        result = select_by_compartment(result, compartment)

    if channel is not None:
        result = select_by_channel(result, channel)

    if measurement is not None:
        result = select_by_measurement(result, measurement)

    return result


def apply_standard_filtering(
    feature_list: List[str], for_object_level: bool = False
) -> List[str]:
    """
    Apply the complete standard filtering pipeline from featuremaps_analysis notebook.

    Pipeline:
    1. Remove technical features (ExecutionTime, ModuleError, Height, Width, Scaling, Group)
    2. Remove unwanted channels (ZBF, Brightfield)
    3. Remove Z-location features (for 2D images)
    4. Remove Children count features
    5. If object-level: Remove object metadata columns (Parent_, FileName_, PathName_)

    This matches the exact filtering from the notebooks.
    """
    # Step 1 & 2: Technical features and unwanted channels
    filtered = filter_features(feature_list)

    # Step 3: Z-location
    filtered = remove_z_location_features(filtered)

    # Step 4: Children counts
    filtered = remove_children_count_features(filtered)

    # Step 5: Object-level specific filtering
    if for_object_level:
        filtered = filter_object_metadata_columns(filtered)

    return filtered


def separate_target_predicted(
    df: pd.DataFrame, exclude_plates: Optional[List[str]] = None
) -> tuple:
    """
    Separate target and predicted data.
    Returns: (target_df, predicted_df)
    """
    df = df.copy()

    if exclude_plates is not None:
        df = df[~df["Metadata_Plate"].isin(exclude_plates)]

    # Remove duplicate columns
    df = df.loc[:, ~df.columns.duplicated()]

    target_df = df[df["Metadata_ImageType"] == "target"].copy()
    pred_df = df[df["Metadata_ImageType"] == "test_pred"].copy()

    return target_df, pred_df


def pivot_to_feature_matrix(
    df: pd.DataFrame, sample_id_col: str, feature_cols: List[str]
) -> pd.DataFrame:
    """
    Convert to feature matrix with samples as rows, features as columns.
    Uses sample_id_col as index.
    """
    return df.set_index(sample_id_col)[feature_cols]


def align_samples(target_matrix: pd.DataFrame, predicted_matrix: pd.DataFrame) -> tuple:
    """
    Align target and predicted matrices to have same samples (rows).
    Returns: (aligned_target, aligned_predicted) sorted by sample ID
    """
    common_samples = sorted(target_matrix.index.intersection(predicted_matrix.index))

    target_aligned = target_matrix.loc[common_samples]
    pred_aligned = predicted_matrix.loc[common_samples]

    return target_aligned, pred_aligned


def remove_zero_variance_features(
    target_matrix: pd.DataFrame, predicted_matrix: pd.DataFrame
) -> tuple:
    """
    Remove features with zero variance in target data.
    Returns: (target_filtered, predicted_filtered) with same features
    """
    feature_stds = target_matrix.std(axis=0)
    valid_features = feature_stds[feature_stds > 0].index.tolist()

    target_filtered = target_matrix[valid_features]
    pred_filtered = predicted_matrix[valid_features]

    return target_filtered, pred_filtered
