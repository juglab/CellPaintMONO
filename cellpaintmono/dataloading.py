"""
Data loading functions for CellProfiler outputs.
"""

import pandas as pd
from pathlib import Path
from typing import Optional, List, Union


def load_image_level_data(
    base_dir: Union[str, Path], plates: Optional[List[str]] = None
) -> pd.DataFrame:
    """Load Image.csv files from plate directories and concatenate."""
    base_path = Path(base_dir)

    if plates is None:
        plates = sorted(
            [
                d.name
                for d in base_path.iterdir()
                if d.is_dir() and d.name.startswith("BR")
            ]
        )

    all_data = []
    for plate in plates:
        image_file = base_path / plate / "Image.csv"
        if image_file.exists():
            df = pd.read_csv(image_file)
            all_data.append(df)

    if not all_data:
        raise FileNotFoundError(f"No Image.csv files found for plates: {plates}")

    return pd.concat(all_data, ignore_index=True)


def load_object_level_data(
    base_dir: Union[str, Path], object_type: str, plates: Optional[List[str]] = None
) -> pd.DataFrame:
    """Load object-level CSV files (Cells.csv, Nuclei.csv, or Cytoplasm.csv)."""
    base_path = Path(base_dir)

    valid_types = ["Cells", "Nuclei", "Cytoplasm"]
    if object_type not in valid_types:
        raise ValueError(f"object_type must be one of {valid_types}")

    if plates is None:
        plates = sorted(
            [
                d.name
                for d in base_path.iterdir()
                if d.is_dir() and d.name.startswith("BR")
            ]
        )

    filename = f"{object_type}.csv"
    all_data = []

    for plate in plates:
        object_file = base_path / plate / filename
        if object_file.exists():
            df = pd.read_csv(object_file)
            all_data.append(df)

    if not all_data:
        raise FileNotFoundError(f"No {filename} files found for plates: {plates}")

    return pd.concat(all_data, ignore_index=True)


def create_sample_ids(df: pd.DataFrame, include_plate: bool = True) -> pd.DataFrame:
    """Create Sample_ID column from Metadata_Plate, Metadata_Well, and Metadata_Site."""
    df = df.copy()

    if include_plate:
        df["Sample_ID"] = (
            df["Metadata_Plate"].str[-2:]
            + "_"
            + df["Metadata_Well"]
            + "_S"
            + df["Metadata_Site"].astype(str)
        )
    else:
        df["Sample_ID"] = df["Metadata_Well"] + "_S" + df["Metadata_Site"].astype(str)

    return df
