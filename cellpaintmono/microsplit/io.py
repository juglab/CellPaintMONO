"""
I/O utilities for MicroSplit outputs.
Load predicted channels and metadata without requiring MicroSplit dependency.
"""

import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional
import tifffile


def load_combined_image(
    sample_id: str, combined_dir: Path
) -> Optional[tifffile.TiffFile]:
    """
    Load combined channel image.

    Parameters
    ----------
    sample_id : str
        Sample identifier
    combined_dir : Path
        Directory containing combined images

    Returns
    -------
    np.ndarray or None
        Combined image array, or None if not found
    """
    img_path = combined_dir / f"{sample_id}.tif"
    if not img_path.exists():
        return None
    return tifffile.imread(img_path)


def load_predicted_channels(
    sample_id: str,
    predicted_dir: Path,
    channels: List[str] = ["DNA", "RNA", "ER", "AGP", "Mito"],
) -> Dict[str, Optional[tifffile.TiffFile]]:
    """
    Load all predicted channels for a sample.

    Parameters
    ----------
    sample_id : str
        Sample identifier
    predicted_dir : Path
        Directory containing predicted channel subdirectories
    channels : list
        Channel names to load

    Returns
    -------
    dict
        Dictionary mapping channel names to image arrays
    """
    channel_images = {}

    for channel in channels:
        img_path = predicted_dir / channel / f"{sample_id}.tif"
        if img_path.exists():
            channel_images[channel] = tifffile.imread(img_path)
        else:
            channel_images[channel] = None

    return channel_images


def load_combine_metadata(sample_id: str, metadata_dir: Path) -> Optional[pd.DataFrame]:
    """
    Load metadata from channel combination step.

    Parameters
    ----------
    sample_id : str
        Sample identifier
    metadata_dir : Path
        Directory containing metadata CSV files

    Returns
    -------
    pd.DataFrame or None
        Metadata for the sample
    """
    metadata_path = metadata_dir / f"{sample_id}_combine.csv"
    if not metadata_path.exists():
        return None
    return pd.read_csv(metadata_path)


def verify_outputs(
    sample_ids: List[str],
    output_dir: Path,
    channels: List[str] = ["DNA", "RNA", "ER", "AGP", "Mito"],
    check_predictions: bool = True,
) -> pd.DataFrame:
    """
    Verify that expected outputs exist for all samples.

    Parameters
    ----------
    sample_ids : list
        List of sample IDs to check
    output_dir : Path
        Base output directory
    channels : list
        Expected channel names
    check_predictions : bool
        Whether to check for predicted channels

    Returns
    -------
    pd.DataFrame
        Status report for each sample
    """
    results = []

    for sample_id in sample_ids:
        # Check combined image
        combined_path = output_dir / "combined" / f"{sample_id}.tif"
        has_combined = combined_path.exists()

        # Check metadata
        metadata_path = output_dir / "metadata" / f"{sample_id}_combine.csv"
        has_metadata = metadata_path.exists()

        # Check predictions
        predicted_channels = {}
        if check_predictions:
            for channel in channels:
                pred_path = output_dir / "predicted" / channel / f"{sample_id}.tif"
                predicted_channels[channel] = pred_path.exists()

        results.append(
            {
                "sample_id": sample_id,
                "has_combined": has_combined,
                "has_metadata": has_metadata,
                **{
                    f"has_{ch}_pred": status
                    for ch, status in predicted_channels.items()
                },
            }
        )

    return pd.DataFrame(results)
