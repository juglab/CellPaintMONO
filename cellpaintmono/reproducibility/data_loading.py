"""
Data loading and aggregation for CellProfiler object-level analysis.

This module provides functions to load Image.csv and Object.csv files,
create sample IDs, and aggregate object-level measurements to well-level.
Works with CRISPR, ORF, and other perturbation screening datasets.
"""

import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def create_sample_id_from_image_csv(image_df: pd.DataFrame) -> Dict[int, str]:
    """
    Create Sample_ID mapping from Image.csv metadata.

    Args:
        image_df: DataFrame with columns [Metadata_Plate, Metadata_Well, Metadata_Site]

    Returns:
        Dictionary mapping ImageNumber to Sample_ID (format: "Plate_Well_SN")

    Example:
        >>> image_df = pd.read_csv("Image.csv")
        >>> sample_map = create_sample_id_from_image_csv(image_df)
        >>> # sample_map[1] might return "BR00118045_A01_S1"
    """
    sample_map = {}

    # Create Sample_ID column
    image_df["Sample_ID"] = (
        image_df["Metadata_Plate"].astype(str)
        + "_"
        + image_df["Metadata_Well"].astype(str)
        + "_S"
        + image_df["Metadata_Site"].astype(str)
    )

    # Map ImageNumber to Sample_ID
    for idx, row in image_df.iterrows():
        sample_map[idx + 1] = row["Sample_ID"]

    return sample_map


def load_and_aggregate_objects(
    object_csv_path: Path,
    image_csv_path: Path,
    features: List[str],
    aggregation_method: str = "mean",
) -> pd.DataFrame:
    """
    Load object-level data and aggregate to well-level (one value per sample).

    Args:
        object_csv_path: Path to object CSV (e.g., Nuclei.csv, Cells.csv)
        image_csv_path: Path to Image.csv with metadata
        features: List of feature column names to aggregate
        aggregation_method: How to aggregate ('mean', 'median', 'std')

    Returns:
        DataFrame indexed by Sample_ID with aggregated features

    Example:
        >>> nuclei_features = ['AreaShape_Area', 'AreaShape_FormFactor']
        >>> nuclei_df = load_and_aggregate_objects(
        ...     Path("Nuclei.csv"),
        ...     Path("Image.csv"),
        ...     nuclei_features
        ... )
        >>> # nuclei_df has one row per Sample_ID with mean values
    """
    # Load image metadata to get Sample_ID mapping
    image_df = pd.read_csv(image_csv_path)
    sample_map = create_sample_id_from_image_csv(image_df)

    # Load object-level data
    obj_df = pd.read_csv(object_csv_path)

    # Map ImageNumber to Sample_ID
    obj_df["Sample_ID"] = obj_df["ImageNumber"].map(sample_map)

    # Aggregate by Sample_ID
    if aggregation_method == "mean":
        aggregated = obj_df.groupby("Sample_ID")[features].mean()
    elif aggregation_method == "median":
        aggregated = obj_df.groupby("Sample_ID")[features].median()
    elif aggregation_method == "std":
        aggregated = obj_df.groupby("Sample_ID")[features].std()
    else:
        raise ValueError(f"Unknown aggregation method: {aggregation_method}")

    return aggregated


def load_multiple_experiments(
    base_dir: Path,
    target_experiment: str,
    predicted_experiments: Dict[str, str],
    compartments: List[str],
    compartment_features: Dict[str, List[str]],
) -> Tuple[Dict[str, pd.DataFrame], Dict[str, Dict[str, pd.DataFrame]]]:
    """
    Load object-level data for target and multiple predicted experiments.

    Args:
        base_dir: Base directory containing experiment folders
        target_experiment: Name of target experiment folder
        predicted_experiments: Dict mapping folder names to display names
        compartments: List of compartment names (e.g., ['Nuclei', 'Cells', 'Cytoplasm'])
        compartment_features: Dict mapping compartment to list of features

    Returns:
        Tuple of (target_objects, predicted_objects) where:
        - target_objects: Dict[compartment] -> DataFrame
        - predicted_objects: Dict[compartment] -> Dict[exp_name] -> DataFrame

    Example:
        >>> compartments = ['Nuclei', 'Cells', 'Cytoplasm']
        >>> compartment_features = {
        ...     'Nuclei': ['AreaShape_Area', 'AreaShape_FormFactor'],
        ...     'Cells': ['AreaShape_Area', 'AreaShape_Eccentricity'],
        ...     'Cytoplasm': ['AreaShape_Area', 'AreaShape_Perimeter']
        ... }
        >>> target_objs, pred_objs = load_multiple_experiments(
        ...     Path("/data"),
        ...     "rand_test_data",
        ...     {"biorand_101225": "bio-crispr-1"},
        ...     compartments,
        ...     compartment_features
        ... )
    """
    target_dir = base_dir / target_experiment
    target_image_csv = target_dir / "Image.csv"

    # Load target objects
    target_objects = {}
    for compartment in compartments:
        compartment_csv = target_dir / f"{compartment}.csv"
        if compartment_csv.exists():
            target_objects[compartment] = load_and_aggregate_objects(
                compartment_csv, target_image_csv, compartment_features[compartment]
            )

    # Load predicted objects for all experiments
    predicted_objects = {compartment: {} for compartment in compartments}

    for folder_name, exp_name in predicted_experiments.items():
        pred_dir = base_dir / folder_name
        pred_image_csv = pred_dir / "Image.csv"

        for compartment in compartments:
            compartment_csv = pred_dir / f"{compartment}.csv"
            if compartment_csv.exists():
                predicted_objects[compartment][exp_name] = load_and_aggregate_objects(
                    compartment_csv, pred_image_csv, compartment_features[compartment]
                )

    return target_objects, predicted_objects


def load_image_level_data(
    base_dir: Path,
    target_experiment: str,
    predicted_experiments: Dict[str, str],
    features: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
    """
    Load Image.csv data for target and multiple predicted experiments.

    Args:
        base_dir: Base directory containing experiment folders
        target_experiment: Name of target experiment folder
        predicted_experiments: Dict mapping folder names to display names
        features: List of features to load (None = all numeric features)

    Returns:
        Tuple of (target_data, predicted_data) where:
        - target_data: DataFrame with target Image.csv data
        - predicted_data: Dict[exp_name] -> DataFrame

    Example:
        >>> target, predicted = load_image_level_data(
        ...     Path("/data"),
        ...     "rand_test_data",
        ...     {"biorand_101225": "bio-crispr-1"},
        ...     features=['Count_Cells', 'Count_Cytoplasm', 'Count_Nuclei']
        ... )
    """
    # Load target
    target_path = base_dir / target_experiment / "Image.csv"
    target_data = pd.read_csv(target_path)

    # Load predicted experiments
    predicted_data = {}
    for folder_name, exp_name in predicted_experiments.items():
        pred_path = base_dir / folder_name / "Image.csv"
        predicted_data[exp_name] = pd.read_csv(pred_path)

    # Filter to specific features if provided
    if features is not None:
        # Keep metadata columns + requested features
        metadata_cols = [
            col for col in target_data.columns if col.startswith("Metadata_")
        ]
        keep_cols = metadata_cols + features

        target_data = target_data[keep_cols]
        predicted_data = {exp: df[keep_cols] for exp, df in predicted_data.items()}

    return target_data, predicted_data


def align_samples(
    target_df: pd.DataFrame,
    predicted_df: pd.DataFrame,
    sample_id_col: str = "Sample_ID",
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    """
    Align target and predicted dataframes by sample ID.

    Args:
        target_df: Target dataframe with sample IDs as index or column
        predicted_df: Predicted dataframe with sample IDs as index or column
        sample_id_col: Name of sample ID column (if not index)

    Returns:
        Tuple of (target_aligned, predicted_aligned, common_samples)

    Example:
        >>> target_aligned, pred_aligned, samples = align_samples(
        ...     target_df, predicted_df
        ... )
        >>> # Both dataframes now have the same samples in the same order
    """
    # Ensure index is Sample_ID
    if sample_id_col in target_df.columns:
        target_df = target_df.set_index(sample_id_col)
    if sample_id_col in predicted_df.columns:
        predicted_df = predicted_df.set_index(sample_id_col)

    # Find common samples
    common_samples = sorted(set(target_df.index) & set(predicted_df.index))

    # Align
    target_aligned = target_df.loc[common_samples]
    predicted_aligned = predicted_df.loc[common_samples]

    return target_aligned, predicted_aligned, common_samples
