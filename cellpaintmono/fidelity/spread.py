"""Cell-to-cell spread and intensity shifts of matched objects."""

from __future__ import annotations

import numpy as np
import pandas as pd


def spread_ratios(
    reference: pd.DataFrame,
    test: pd.DataFrame,
    channels: list[str],
    statistics: tuple[str, ...] = ("MeanIntensity", "StdIntensity"),
) -> pd.DataFrame:
    """SD across matched objects, test divided by reference, per channel.

    Parameters
    ----------
    reference : pandas.DataFrame
        Reference values of matched objects, `Intensity_*` columns.
    test : pandas.DataFrame
        Test values of the same objects in the same row order.
    channels : list of str
        Channels to report.
    statistics : tuple of str
        Intensity statistics, e.g. `MeanIntensity` (mean intensity) and
        `StdIntensity` (within-object spread).

    Returns
    -------
    pandas.DataFrame
        Per channel: one `sd_ratio_<statistic>` column per statistic and
        `median_shift` = median test / median reference - 1 of the mean intensity.
    """
    rows = []
    for ch in channels:
        row: dict[str, float | str] = {"channel": ch}
        for stat in statistics:
            col = f"Intensity_{stat}_{ch}"
            row[f"sd_ratio_{stat}"] = float(np.nanstd(test[col], ddof=1)) / float(
                np.nanstd(reference[col], ddof=1)
            )
        mean_col = f"Intensity_MeanIntensity_{ch}"
        row["median_shift"] = (
            float(np.nanmedian(test[mean_col]))
            / float(np.nanmedian(reference[mean_col]))
            - 1
        )
        rows.append(row)
    return pd.DataFrame(rows)
