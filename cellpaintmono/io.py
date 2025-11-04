"""
Data loading and saving utilities for CellPaintMONO.

This module provides functions for:
- Loading CellProfiler output CSV files
- Identifying metadata and feature columns
- Saving analysis results
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional


def load_cellprofiler_data(
    base_dir: Path,
    file_name: str = "Image.csv",
    plate_prefix: str = "BR"
) -> pd.DataFrame:
    """
    Load and combine CellProfiler output files from multiple plates.
    
    Parameters
    ----------
    base_dir : Path
        Base directory containing plate subdirectories
    file_name : str, optional
        Name of the CSV file to load (default: "Image.csv")
    plate_prefix : str, optional
        Prefix to identify plate directories (default: "BR")
    
    Returns
    -------
    pd.DataFrame
        Combined dataframe with all plate data
        
    Examples
    --------
    >>> from pathlib import Path
    >>> base_dir = Path("/path/to/cellprofiler/outputs")
    >>> df = load_cellprofiler_data(base_dir)
    >>> print(f"Loaded {len(df)} samples")
    """
    base_dir = Path(base_dir)
    
    # Find all plate directories
    plates = sorted([
        d.name for d in base_dir.iterdir() 
        if d.is_dir() and d.name.startswith(plate_prefix)
    ])
    
    if not plates:
        raise ValueError(f"No plate directories found with prefix '{plate_prefix}' in {base_dir}")
    
    print(f"Found {len(plates)} plates: {', '.join(plates)}\n")
    
    # Load data from each plate
    all_data = []
    for plate in plates:
        file_path = base_dir / plate / file_name
        
        if not file_path.exists():
            print(f"Warning: {file_name} not found in {plate}, skipping...")
            continue
            
        df = pd.read_csv(file_path)
        all_data.append(df)
        
        # Print summary for this plate
        n_wells = df['Metadata_Well'].nunique() if 'Metadata_Well' in df.columns else 0
        n_target = len(df[df['Metadata_ImageType'] == 'target']) if 'Metadata_ImageType' in df.columns else 0
        n_pred = len(df[df['Metadata_ImageType'] == 'test_pred']) if 'Metadata_ImageType' in df.columns else 0
        
        print(f"{plate}: {n_wells} wells | {n_target} target | {n_pred} predicted")
    
    if not all_data:
        raise ValueError(f"No valid data files found in any plate directory")
    
    # Combine all data
    df_combined = pd.concat(all_data, ignore_index=True)
    
    # Print overall summary
    n_plates = df_combined['Metadata_Plate'].nunique() if 'Metadata_Plate' in df_combined.columns else 0
    n_wells = df_combined['Metadata_Well'].nunique() if 'Metadata_Well' in df_combined.columns else 0
    
    print(f"\nTotal: {len(df_combined)} image sites, {n_plates} plates, {n_wells} wells\n")
    
    return df_combined


def identify_feature_columns(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """
    Separate metadata columns from numeric feature columns.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe with mixed metadata and feature columns
    
    Returns
    -------
    Tuple[List[str], List[str]]
        Tuple of (metadata_columns, numeric_feature_columns)
        
    Examples
    --------
    >>> metadata_cols, feature_cols = identify_feature_columns(df)
    >>> print(f"Found {len(metadata_cols)} metadata columns")
    >>> print(f"Found {len(feature_cols)} feature columns")
    """
    # Identify metadata columns (those starting with 'Metadata_')
    metadata_cols = [col for col in df.columns if col.startswith('Metadata_')]
    
    # Identify numeric feature columns
    numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()
    
    print(f"Metadata columns: {len(metadata_cols)}")
    print(f"Numeric features: {len(numeric_features)}\n")
    
    return metadata_cols, numeric_features


def create_sample_id(
    df: pd.DataFrame,
    plate_col: str = 'Metadata_Plate',
    well_col: str = 'Metadata_Well',
    site_col: str = 'Metadata_Site'
) -> pd.Series:
    """
    Create unique sample IDs from plate, well, and site information.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe with metadata columns
    plate_col : str, optional
        Name of plate column (default: 'Metadata_Plate')
    well_col : str, optional
        Name of well column (default: 'Metadata_Well')
    site_col : str, optional
        Name of site column (default: 'Metadata_Site')
    
    Returns
    -------
    pd.Series
        Series of sample IDs in format: 'PlateID_Well_SiteNumber'
        
    Examples
    --------
    >>> df['Sample_ID'] = create_sample_id(df)
    >>> print(df['Sample_ID'].head())
    """
    # Extract last 2 characters of plate name and combine with well and site
    sample_ids = (
        df[plate_col].str[-2:] + '_' +
        df[well_col] + '_S' +
        df[site_col].astype(str)
    )
    
    return sample_ids


def separate_target_predicted(
    df: pd.DataFrame,
    image_type_col: str = 'Metadata_ImageType',
    target_label: str = 'target',
    predicted_label: str = 'test_pred',
    features: Optional[List[str]] = None,
    sample_id_col: str = 'Sample_ID'
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Separate dataframe into target and predicted samples.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe with both target and predicted samples
    image_type_col : str, optional
        Column name indicating image type (default: 'Metadata_ImageType')
    target_label : str, optional
        Label for target images (default: 'target')
    predicted_label : str, optional
        Label for predicted images (default: 'test_pred')
    features : List[str], optional
        List of feature columns to keep. If None, keeps all numeric columns
    sample_id_col : str, optional
        Column name for sample IDs (default: 'Sample_ID')
    
    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        Tuple of (target_df, predicted_df) with aligned samples
        
    Examples
    --------
    >>> target_df, pred_df = separate_target_predicted(df, features=filtered_features)
    >>> print(f"Target samples: {len(target_df)}")
    >>> print(f"Predicted samples: {len(pred_df)}")
    """
    if features is None:
        features = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Separate target and predicted
    target_df = df[df[image_type_col] == target_label][[sample_id_col] + features].copy()
    pred_df = df[df[image_type_col] == predicted_label][[sample_id_col] + features].copy()
    
    print(f"Target samples: {len(target_df)}")
    print(f"Predicted samples: {len(pred_df)}")
    
    # Set index and align samples
    target_df = target_df.set_index(sample_id_col)
    pred_df = pred_df.set_index(sample_id_col)
    
    # Get common samples
    common_samples = sorted(target_df.index.intersection(pred_df.index))
    target_df = target_df.loc[common_samples]
    pred_df = pred_df.loc[common_samples]
    
    print(f"Aligned samples: {len(common_samples)}\n")
    
    return target_df, pred_df


def save_results(
    data: pd.DataFrame,
    output_dir: Path,
    filename: str,
    create_dir: bool = True
) -> Path:
    """
    Save analysis results to CSV file.
    
    Parameters
    ----------
    data : pd.DataFrame
        Data to save
    output_dir : Path
        Directory to save the file
    filename : str
        Name of the output file
    create_dir : bool, optional
        Create output directory if it doesn't exist (default: True)
    
    Returns
    -------
    Path
        Path to the saved file
        
    Examples
    --------
    >>> output_path = save_results(stats_df, output_dir, "statistics_summary.csv")
    >>> print(f"Saved to: {output_path}")
    """
    output_dir = Path(output_dir)
    
    if create_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = output_dir / filename
    data.to_csv(output_path, index=False)
    
    print(f"Saved: {output_path}")
    
    return output_path


def save_feature_list(
    features: List[str],
    output_dir: Path,
    filename: str = "features.txt",
    description: str = ""
) -> Path:
    """
    Save a list of features to a text file.
    
    Parameters
    ----------
    features : List[str]
        List of feature names
    output_dir : Path
        Directory to save the file
    filename : str, optional
        Name of the output file (default: "features.txt")
    description : str, optional
        Description to add at the top of the file
    
    Returns
    -------
    Path
        Path to the saved file
        
    Examples
    --------
    >>> save_feature_list(
    ...     mean_features,
    ...     output_dir,
    ...     "mean_features.txt",
    ...     "Mean-based feature set"
    ... )
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = output_dir / filename
    
    with open(output_path, 'w') as f:
        if description:
            f.write(f"{description}\n")
            f.write("=" * 80 + "\n\n")
        
        f.write(f"Total features: {len(features)}\n\n")
        
        for feat in sorted(features):
            f.write(f"{feat}\n")
    
    print(f"Saved feature list: {output_path}")
    
    return output_path