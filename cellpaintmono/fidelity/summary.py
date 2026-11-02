"""Summary of agreement and distortion per feature group."""

from __future__ import annotations

from typing import Any

import pandas as pd

DEFAULT_GROUPS: list[dict[str, Any]] = [
    {"name": "Shape, size, neighbours", "families": ["Shape", "Neighbours"]},
    {
        "name": "DNA, ER, RNA intensity and radial distribution",
        "families": ["Intensity", "Radial distrib."],
        "channels": ["DNA", "ER", "RNA"],
    },
    {
        "name": "AGP and Mito intensity",
        "families": ["Intensity"],
        "channels": ["AGP", "Mito"],
    },
    {"name": "Granularity beyond scale 1", "families": ["Granularity"], "min_scale": 2},
    {"name": "Colocalisation", "families": ["Colocalisation"]},
]


def group_mask(table: pd.DataFrame, group: dict[str, Any]) -> pd.Series:
    """Rows of a per-feature table that belong to a feature group.

    Parameters
    ----------
    table : pandas.DataFrame
        Per-feature table with `family`, `channel` and `scale` columns.
    group : dict
        `families` (required), optional `channels` and `min_scale`.

    Returns
    -------
    pandas.Series
        Boolean mask.
    """
    mask = table["family"].isin(group["families"])
    if "channels" in group:
        mask &= table["channel"].isin(group["channels"])
    if "min_scale" in group:
        mask &= table["scale"] >= group["min_scale"]
    return mask


def group_summary(
    agreement: pd.DataFrame,
    distortion: pd.DataFrame,
    groups: list[dict[str, Any]],
) -> pd.DataFrame:
    """Median agreement and distortion per feature group.

    For each group the median over its features is taken per compartment;
    the table reports the range of those medians over compartments, and the
    median over all features of the group.

    Parameters
    ----------
    agreement : pandas.DataFrame
        `agreement_per_feature` table.
    distortion : pandas.DataFrame
        `distortion_per_feature` table.
    groups : list of dict
        Group definitions, see `group_mask`.

    Returns
    -------
    pandas.DataFrame
        One row per group with `rho_*`, `ratio_to_noise_*` and
        `fraction_of_well_*` columns (`min`, `max` over compartments, `all`).
    """
    rows = []
    for group in groups:
        row: dict[str, Any] = {"group": group["name"]}
        for table, column in (
            (agreement, "rho"),
            (distortion, "ratio_to_noise"),
            (distortion, "fraction_of_well"),
        ):
            sub = table[group_mask(table, group)]
            per_comp = sub.groupby("compartment")[column].median()
            row[f"{column}_min"] = per_comp.min()
            row[f"{column}_max"] = per_comp.max()
            row[f"{column}_all"] = sub[column].median()
        row["n_features"] = int(group_mask(distortion, group).sum())
        rows.append(row)
    return pd.DataFrame(rows)
