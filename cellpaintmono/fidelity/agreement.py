"""Per-object agreement between matched reference and test objects."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy.stats import spearmanr

from .features import FAMILY_ORDER


def spearman_per_feature(reference: npt.NDArray, test: npt.NDArray) -> npt.NDArray:
    """Spearman correlation of each column between two aligned arrays.

    Parameters
    ----------
    reference : numpy.ndarray
        Reference values, shape (objects, features).
    test : numpy.ndarray
        Test values of the same objects, same shape.

    Returns
    -------
    numpy.ndarray
        One correlation per feature; NaN when a column is constant.
    """
    rho = np.full(reference.shape[1], np.nan)
    for j in range(reference.shape[1]):
        a, b = reference[:, j], test[:, j]
        ok = ~(np.isnan(a) | np.isnan(b))
        if ok.sum() > 2 and np.ptp(a[ok]) > 0 and np.ptp(b[ok]) > 0:
            rho[j] = spearmanr(a[ok], b[ok])[0]
    return rho


def family_table(
    table: pd.DataFrame,
    value: str,
    compartments: list[str],
    column: str = "family",
    order: list[str] | None = None,
) -> pd.DataFrame:
    """Median of a per-feature value per group and compartment.

    Parameters
    ----------
    table : pandas.DataFrame
        Per-feature table with `compartment`, the grouping column and `value`.
    value : str
        Column to summarise.
    compartments : list of str
        Column order of the result.
    column : str
        Grouping column, e.g. `family` or `channel`.
    order : list of str or None
        Row order; defaults to the standard family order for `family`.

    Returns
    -------
    pandas.DataFrame
        Groups as rows, compartments as columns.
    """
    out = table.groupby([column, "compartment"])[value].median().unstack()
    out = out.reindex(columns=compartments)
    if order is None and column == "family":
        order = FAMILY_ORDER
    if order is not None:
        out = out.reindex([o for o in order if o in out.index])
    return out
