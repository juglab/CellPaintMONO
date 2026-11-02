"""Feature selection, annotation and robust scaling."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import pandas as pd

NON_FEATURE_PREFIXES = (
    "ImageNumber",
    "ObjectNumber",
    "Number_Object_Number",
    "Location",
    "Parent",
    "Children",
    "Metadata",
    "FileName",
    "PathName",
    "image_id",
    "well_id",
)
NON_FEATURE_TOKENS = ("BoundingBox", "Center", "Orientation", "ObjectNumber")

FAMILY_BY_PREFIX = {
    "AreaShape": "Shape",
    "Intensity": "Intensity",
    "Neighbors": "Neighbours",
    "RadialDistribution": "Radial distrib.",
    "Granularity": "Granularity",
    "Correlation": "Colocalisation",
    "Texture": "Texture",
}
ZERNIKE_FAMILY = "Zernike shape"
FAMILY_ORDER = [
    "Shape",
    "Intensity",
    "Neighbours",
    "Radial distrib.",
    ZERNIKE_FAMILY,
    "Granularity",
    "Texture",
    "Colocalisation",
]
CHANNEL_FAMILIES = ("Intensity", "Granularity", "Radial distrib.", "Texture")


def is_feature(name: str, exclude_images: list[str]) -> bool:
    """Whether a column is a measurement rather than bookkeeping.

    Parameters
    ----------
    name : str
        Column name.
    exclude_images : list of str
        Image names whose measurements are not features.

    Returns
    -------
    bool
        True for measurement columns.
    """
    if name.startswith(NON_FEATURE_PREFIXES):
        return False
    tokens = (*NON_FEATURE_TOKENS, *exclude_images)
    return not any(t in name for t in tokens)


def select_features(
    reference: pd.DataFrame, test: pd.DataFrame, exclude_images: list[str]
) -> list[str]:
    """Numeric measurement columns present in both tables.

    Parameters
    ----------
    reference : pandas.DataFrame
        Reference object table.
    test : pandas.DataFrame
        Test object table.
    exclude_images : list of str
        Image names whose measurements are dropped.

    Returns
    -------
    list of str
        Feature names in reference column order.
    """
    return [
        c
        for c in reference.columns
        if c in test.columns
        and is_feature(c, exclude_images)
        and reference[c].dtype.kind in "fi"
        and test[c].dtype.kind in "fi"
    ]


def robust_spread(values: npt.NDArray) -> float:
    """Scaled median absolute deviation, ignoring NaN.

    Parameters
    ----------
    values : numpy.ndarray
        One-dimensional values.

    Returns
    -------
    float
        1.4826 * MAD, comparable to a standard deviation for normal data.
    """
    med = np.nanmedian(values)
    return float(1.4826 * np.nanmedian(np.abs(values - med)))


def filter_for_distances(
    reference: pd.DataFrame, features: list[str], max_missing: float
) -> list[str]:
    """Drop features that cannot be robustly scaled.

    Parameters
    ----------
    reference : pandas.DataFrame
        Reference object table.
    features : list of str
        Candidate features.
    max_missing : float
        Largest allowed fraction of missing reference values.

    Returns
    -------
    list of str
        Features with non-zero reference MAD and few missing values.
    """
    keep = []
    for f in features:
        values = reference[f].to_numpy(dtype=float)
        if np.isnan(values).mean() > max_missing:
            continue
        if robust_spread(values) > 0:
            keep.append(f)
    return keep


def detect_channels(table: pd.DataFrame, exclude_images: list[str]) -> list[str]:
    """Channel names from `Intensity_MeanIntensity_<channel>` columns.

    Parameters
    ----------
    table : pandas.DataFrame
        Object table.
    exclude_images : list of str
        Image names that are not channels.

    Returns
    -------
    list of str
        Channel names in column order.
    """
    prefix = "Intensity_MeanIntensity_"
    return [
        c[len(prefix) :]
        for c in table.columns
        if c.startswith(prefix) and c[len(prefix) :] not in exclude_images
    ]


def feature_family(name: str) -> str:
    """Feature family of a CellProfiler column.

    Parameters
    ----------
    name : str
        Feature name, e.g. `Intensity_MeanIntensity_DNA`.

    Returns
    -------
    str
        Family label; Zernike moments are separated from other shape features.
    """
    prefix = name.split("_")[0]
    if prefix == "AreaShape" and "Zernike" in name:
        return ZERNIKE_FAMILY
    return FAMILY_BY_PREFIX.get(prefix, prefix)


def feature_channels(name: str, channels: list[str]) -> list[str]:
    """Channels named in a feature, in the order they appear.

    Parameters
    ----------
    name : str
        Feature name.
    channels : list of str
        Known channel names.

    Returns
    -------
    list of str
        Zero, one or two channel names.
    """
    padded = f"_{name}_"
    found = [(padded.find(f"_{ch}_"), ch) for ch in channels if f"_{ch}_" in padded]
    return [ch for _, ch in sorted(found)]


def granularity_scale(name: str) -> int | None:
    """Scale of a granularity feature.

    Parameters
    ----------
    name : str
        Feature name, e.g. `Granularity_3_Mito`.

    Returns
    -------
    int or None
        Scale, or None for other features.
    """
    parts = name.split("_")
    if parts[0] == "Granularity" and len(parts) > 1 and parts[1].isdigit():
        return int(parts[1])
    return None


def annotate_features(features: list[str], channels: list[str]) -> pd.DataFrame:
    """Family, channel and granularity scale of each feature.

    Parameters
    ----------
    features : list of str
        Feature names.
    channels : list of str
        Known channel names.

    Returns
    -------
    pandas.DataFrame
        Columns `feature`, `family`, `channel` (single-channel families only,
        otherwise None), `channels` (all channels joined by `-`) and `scale`.
    """
    rows = []
    for f in features:
        family = feature_family(f)
        found = feature_channels(f, channels)
        single = found[0] if family in CHANNEL_FAMILIES and len(found) == 1 else None
        rows.append(
            {
                "feature": f,
                "family": family,
                "channel": single,
                "channels": "-".join(found) or None,
                "scale": granularity_scale(f),
            }
        )
    return pd.DataFrame(rows)


@dataclass
class RobustScaler:
    """Robust z-scoring fitted on reference data.

    Attributes
    ----------
    features : list of str
        Scaled features.
    median : numpy.ndarray
        Reference median per feature.
    spread : numpy.ndarray
        Reference 1.4826 * MAD per feature.
    clip : float
        Absolute bound applied after scaling.
    """

    features: list[str]
    median: npt.NDArray
    spread: npt.NDArray
    clip: float

    @classmethod
    def fit(
        cls, reference: pd.DataFrame, features: list[str], clip: float
    ) -> RobustScaler:
        """Fit on a reference table.

        Parameters
        ----------
        reference : pandas.DataFrame
            Reference object table.
        features : list of str
            Features to scale; each must have a non-zero MAD.
        clip : float
            Absolute bound applied after scaling.

        Returns
        -------
        RobustScaler
            The fitted scaler.
        """
        values = reference[features].to_numpy(dtype=float)
        median = np.nanmedian(values, axis=0)
        spread = 1.4826 * np.nanmedian(np.abs(values - median), axis=0)
        return cls(features, median, spread, clip)

    def transform(self, table: pd.DataFrame) -> npt.NDArray:
        """Scale a table.

        Parameters
        ----------
        table : pandas.DataFrame
            Object table with the scaler's features.

        Returns
        -------
        numpy.ndarray
            Robust z-scores, shape (objects, features); NaN stays NaN.
        """
        values = table[self.features].to_numpy(dtype=float)
        return np.clip((values - self.median) / self.spread, -self.clip, self.clip)
