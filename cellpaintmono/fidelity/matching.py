"""Object matching between runs and segmentation summaries."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

from .io import IMAGE_ID, CellProfilerRun

LOCATION = ["Location_Center_X", "Location_Center_Y"]
PAIR_COLUMNS = [IMAGE_ID, "ref_object", "test_object", "distance"]


def match_objects(
    reference: pd.DataFrame,
    test: pd.DataFrame,
    max_distance: float,
    image_ids: list[str],
) -> pd.DataFrame:
    """One-to-one centroid matching per image.

    The assignment minimises the summed distance over all objects of an image,
    and pairs further apart than `max_distance` are dropped afterwards.

    Parameters
    ----------
    reference : pandas.DataFrame
        Reference object table with `image_id`, `ObjectNumber` and centroids.
    test : pandas.DataFrame
        Test object table, same columns.
    max_distance : float
        Largest centroid distance of a kept pair, in pixels.
    image_ids : list of str
        Images to match.

    Returns
    -------
    pandas.DataFrame
        Columns `image_id`, `ref_object`, `test_object`, `distance`.
    """
    ref_groups = dict(tuple(reference.groupby(IMAGE_ID)))
    test_groups = dict(tuple(test.groupby(IMAGE_ID)))
    parts = []
    for image_id in image_ids:
        a, b = ref_groups.get(image_id), test_groups.get(image_id)
        if a is None or b is None:
            continue
        dist = cdist(a[LOCATION].to_numpy(), b[LOCATION].to_numpy())
        rows, cols = linear_sum_assignment(dist)
        keep = dist[rows, cols] <= max_distance
        parts.append(
            pd.DataFrame(
                {
                    IMAGE_ID: image_id,
                    "ref_object": a["ObjectNumber"].to_numpy()[rows[keep]],
                    "test_object": b["ObjectNumber"].to_numpy()[cols[keep]],
                    "distance": dist[rows, cols][keep],
                }
            )
        )
    if not parts:
        return pd.DataFrame(columns=PAIR_COLUMNS)
    return pd.concat(parts, ignore_index=True)


def pair_through_parent(
    primary_pairs: pd.DataFrame,
    reference: pd.DataFrame,
    test: pd.DataFrame,
    parent_column: str,
) -> pd.DataFrame:
    """Pair objects of another compartment through matched parent objects.

    Parameters
    ----------
    primary_pairs : pandas.DataFrame
        Output of `match_objects` for the primary compartment.
    reference : pandas.DataFrame
        Reference table of the dependent compartment.
    test : pandas.DataFrame
        Test table of the dependent compartment.
    parent_column : str
        Column holding the parent object number, e.g. `Parent_Nuclei`.

    Returns
    -------
    pandas.DataFrame
        Columns `image_id`, `ref_object`, `test_object`, `distance` (the
        distance of the parent pair). Pairs whose parent has no child in either
        run are dropped.

    Raises
    ------
    KeyError
        If `parent_column` is missing.
    """
    for table in (reference, test):
        if parent_column not in table:
            raise KeyError(f"Column {parent_column} not found")
    ref_child = reference.drop_duplicates([IMAGE_ID, parent_column]).set_index(
        [IMAGE_ID, parent_column]
    )["ObjectNumber"]
    test_child = test.drop_duplicates([IMAGE_ID, parent_column]).set_index(
        [IMAGE_ID, parent_column]
    )["ObjectNumber"]
    ref_key = pd.MultiIndex.from_arrays(
        [primary_pairs[IMAGE_ID], primary_pairs["ref_object"]]
    )
    test_key = pd.MultiIndex.from_arrays(
        [primary_pairs[IMAGE_ID], primary_pairs["test_object"]]
    )
    out = primary_pairs[[IMAGE_ID, "distance"]].copy()
    out["ref_object"] = ref_child.reindex(ref_key).to_numpy()
    out["test_object"] = test_child.reindex(test_key).to_numpy()
    out = out.dropna(subset=["ref_object", "test_object"])
    out[["ref_object", "test_object"]] = out[["ref_object", "test_object"]].astype(int)
    return out[PAIR_COLUMNS].reset_index(drop=True)


def paired_values(
    reference: pd.DataFrame,
    test: pd.DataFrame,
    pairs: pd.DataFrame,
    features: list[str],
) -> tuple[npt.NDArray, npt.NDArray]:
    """Feature values of paired objects, row-aligned.

    Parameters
    ----------
    reference : pandas.DataFrame
        Reference object table.
    test : pandas.DataFrame
        Test object table.
    pairs : pandas.DataFrame
        Object pairs of this compartment.
    features : list of str
        Features to extract.

    Returns
    -------
    tuple of numpy.ndarray
        Reference and test values, each shape (pairs, features).
    """
    ref = reference.set_index([IMAGE_ID, "ObjectNumber"])
    tst = test.set_index([IMAGE_ID, "ObjectNumber"])
    ref_key = pd.MultiIndex.from_arrays([pairs[IMAGE_ID], pairs["ref_object"]])
    test_key = pd.MultiIndex.from_arrays([pairs[IMAGE_ID], pairs["test_object"]])
    return (
        ref.loc[ref_key, features].to_numpy(dtype=float),
        tst.loc[test_key, features].to_numpy(dtype=float),
    )


def matched_mask(
    table: pd.DataFrame, pairs: pd.DataFrame, side: str
) -> npt.NDArray[np.bool_]:
    """Which objects of a table appear in a set of pairs.

    Parameters
    ----------
    table : pandas.DataFrame
        Object table.
    pairs : pandas.DataFrame
        Object pairs.
    side : {"ref", "test"}
        Which side of the pairs the table belongs to.

    Returns
    -------
    numpy.ndarray
        Boolean mask over the rows of `table`.
    """
    key = pd.MultiIndex.from_arrays([pairs[IMAGE_ID], pairs[f"{side}_object"]])
    own = pd.MultiIndex.from_arrays([table[IMAGE_ID], table["ObjectNumber"]])
    return own.isin(key)


def segmentation_summary(
    reference: CellProfilerRun,
    test: CellProfilerRun,
    primary_pairs: pd.DataFrame,
    primary: str,
) -> pd.DataFrame:
    """Object counts, matches and thresholds per image.

    Parameters
    ----------
    reference : CellProfilerRun
        Reference run.
    test : CellProfilerRun
        Test run.
    primary_pairs : pandas.DataFrame
        Matched objects of the primary compartment.
    primary : str
        Primary compartment name.

    Returns
    -------
    pandas.DataFrame
        One row per image: reference and test counts, matched count, median
        centroid shift and every `Threshold_FinalThreshold_*` column of both runs.
    """
    ids = reference.image[IMAGE_ID]
    out = pd.DataFrame({IMAGE_ID: ids})
    out["n_ref"] = ids.map(reference.objects[primary].groupby(IMAGE_ID).size())
    out["n_test"] = ids.map(test.objects[primary].groupby(IMAGE_ID).size())
    grouped = primary_pairs.groupby(IMAGE_ID)
    out["n_matched"] = ids.map(grouped.size()).fillna(0).astype(int)
    out["median_shift_px"] = ids.map(grouped["distance"].median())
    out["count_change"] = out["n_test"] / out["n_ref"] - 1
    thresholds = [
        c for c in reference.image if c.startswith("Threshold_FinalThreshold")
    ]
    ref_t = reference.image.set_index(IMAGE_ID)
    test_t = test.image.set_index(IMAGE_ID)
    for col in thresholds:
        name = col.removeprefix("Threshold_FinalThreshold_")
        out[f"threshold_{name}_ref"] = ids.map(ref_t[col])
        if col in test_t:
            out[f"threshold_{name}_test"] = ids.map(test_t[col])
    return out
