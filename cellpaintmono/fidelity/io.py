"""Loading and aligning CellProfiler CSV outputs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

IMAGE_ID = "image_id"
WELL_ID = "well_id"


@dataclass
class CellProfilerRun:
    """One CellProfiler run: the image table and one table per compartment.

    Attributes
    ----------
    image : pandas.DataFrame
        `Image.csv` with added `image_id` and `well_id` columns.
    objects : dict of str to pandas.DataFrame
        Per-object tables with added `image_id` and `well_id` columns.
    """

    image: pd.DataFrame
    objects: dict[str, pd.DataFrame]


def make_id(df: pd.DataFrame, keys: list[str]) -> pd.Series:
    """Join key columns into one string id.

    Parameters
    ----------
    df : pandas.DataFrame
        Table holding the key columns.
    keys : list of str
        Columns to join, in order.

    Returns
    -------
    pandas.Series
        Ids of the form `value1_value2_...`.
    """
    return df[keys].astype(str).agg("_".join, axis=1)


def load_run(
    folder: str | Path,
    compartments: list[str],
    image_keys: list[str],
    well_keys: list[str],
) -> CellProfilerRun:
    """Load a CellProfiler run exported with ExportToSpreadsheet.

    Parameters
    ----------
    folder : str or pathlib.Path
        Folder with `Image.csv` and `<compartment>.csv` files.
    compartments : list of str
        Compartments to load.
    image_keys : list of str
        `Image.csv` columns identifying a field of view.
    well_keys : list of str
        `Image.csv` columns identifying a well.

    Returns
    -------
    CellProfilerRun
        The loaded run.

    Raises
    ------
    ValueError
        If image ids are not unique within the run.
    """
    folder = Path(folder)
    image = pd.read_csv(folder / "Image.csv")
    image[IMAGE_ID] = make_id(image, image_keys)
    image[WELL_ID] = make_id(image, well_keys)
    if image[IMAGE_ID].duplicated().any():
        raise ValueError(f"Image keys {image_keys} are not unique in {folder}")
    image_map = image.set_index("ImageNumber")[IMAGE_ID]
    well_map = image.set_index("ImageNumber")[WELL_ID]
    objects = {}
    for comp in compartments:
        df = pd.read_csv(folder / f"{comp}.csv")
        df[IMAGE_ID] = df["ImageNumber"].map(image_map)
        df[WELL_ID] = df["ImageNumber"].map(well_map)
        objects[comp] = df
    return CellProfilerRun(image=image, objects=objects)


def common_images(reference: CellProfilerRun, test: CellProfilerRun) -> list[str]:
    """Image ids present in both runs, in reference order.

    Parameters
    ----------
    reference : CellProfilerRun
        Reference run.
    test : CellProfilerRun
        Test run.

    Returns
    -------
    list of str
        Shared image ids.

    Raises
    ------
    ValueError
        If the runs share no image.
    """
    shared = set(test.image[IMAGE_ID])
    ids = [i for i in reference.image[IMAGE_ID] if i in shared]
    if not ids:
        raise ValueError("Reference and test runs share no image id")
    return ids


def restrict_to_images(run: CellProfilerRun, image_ids: list[str]) -> CellProfilerRun:
    """Keep only the given images in a run.

    Parameters
    ----------
    run : CellProfilerRun
        Run to filter.
    image_ids : list of str
        Image ids to keep.

    Returns
    -------
    CellProfilerRun
        Filtered copy; the image table follows the order of `image_ids`.
    """
    image = run.image.set_index(IMAGE_ID).loc[image_ids].reset_index()
    objects = {
        comp: df[df[IMAGE_ID].isin(image_ids)].reset_index(drop=True)
        for comp, df in run.objects.items()
    }
    return CellProfilerRun(image=image, objects=objects)


def well_labels(image: pd.DataFrame, well_column: str = "Metadata_Well") -> dict:
    """Short display labels for wells.

    Parameters
    ----------
    image : pandas.DataFrame
        Image table with a `well_id` column.
    well_column : str
        Column with the plate position of the well.

    Returns
    -------
    dict
        Well id to label. The plate position alone when it is unique across
        plates, the full well id otherwise.
    """
    wells = image.drop_duplicates(WELL_ID)
    if well_column in wells and wells[well_column].is_unique:
        return dict(zip(wells[WELL_ID], wells[well_column].astype(str), strict=True))
    return dict(zip(wells[WELL_ID], wells[WELL_ID], strict=True))
