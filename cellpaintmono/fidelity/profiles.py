"""Well-level profiles: identity retrieval and distance geometry."""

from __future__ import annotations

from itertools import combinations

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy.stats import spearmanr


def well_profiles(
    z: npt.NDArray, wells: npt.NDArray, features: list[str], prefix: str
) -> pd.DataFrame:
    """Median of each feature over the cells of a well.

    Parameters
    ----------
    z : numpy.ndarray
        Robust z-scores, shape (cells, features).
    wells : numpy.ndarray
        Well id of each cell.
    features : list of str
        Feature names of the columns of `z`.
    prefix : str
        Prefix added to the column names (the compartment).

    Returns
    -------
    pandas.DataFrame
        Wells as rows, `<prefix>_<feature>` as columns.
    """
    df = pd.DataFrame(z, columns=[f"{prefix}_{f}" for f in features])
    df["well_id"] = wells
    return df.groupby("well_id").median()


def standardise(
    reference: pd.DataFrame, test: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Standardise both profile sets with the reference across-well mean and SD.

    Parameters
    ----------
    reference : pandas.DataFrame
        Reference profiles, wells as rows.
    test : pandas.DataFrame
        Test profiles with the same wells and columns.

    Returns
    -------
    tuple of pandas.DataFrame
        Standardised reference and test profiles, test rows in reference order.
        Features constant across reference wells are dropped.
    """
    test = test.loc[reference.index, reference.columns]
    mean, sd = reference.mean(), reference.std()
    keep = sd > 0
    return (
        ((reference - mean) / sd).loc[:, keep],
        ((test - mean) / sd).loc[:, keep],
    )


def remove_global_offset(reference: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    """Subtract the mean test minus reference difference shared by all wells.

    Parameters
    ----------
    reference : pandas.DataFrame
        Standardised reference profiles.
    test : pandas.DataFrame
        Standardised test profiles, same wells and columns.

    Returns
    -------
    pandas.DataFrame
        Test profiles without the global offset.
    """
    return test - (test - reference).mean()


def retrieval(reference: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    """Pearson correlation of every test profile with every reference profile.

    Parameters
    ----------
    reference : pandas.DataFrame
        Reference profiles, wells as rows.
    test : pandas.DataFrame
        Test profiles, same wells and columns.

    Returns
    -------
    pandas.DataFrame
        Test wells as rows, reference wells as columns.
    """
    n = len(test)
    corr = np.corrcoef(test.to_numpy(), reference.to_numpy())[:n, n:]
    return pd.DataFrame(corr, index=test.index, columns=reference.index)


def hit_rate(corr: pd.DataFrame) -> float:
    """Fraction of test wells whose best-correlated reference well is their own.

    Parameters
    ----------
    corr : pandas.DataFrame
        Output of `retrieval`.

    Returns
    -------
    float
        Hit rate between 0 and 1.
    """
    best = corr.columns[corr.to_numpy().argmax(axis=1)]
    return float(np.mean(best == corr.index))


def distance_geometry(
    reference: pd.DataFrame, test: pd.DataFrame
) -> tuple[pd.DataFrame, float]:
    """Pairwise Euclidean distances between wells in both series.

    Parameters
    ----------
    reference : pandas.DataFrame
        Standardised reference profiles.
    test : pandas.DataFrame
        Standardised test profiles, same wells and columns.

    Returns
    -------
    tuple
        Table of well pairs with `ref_distance` and `test_distance`, and the
        Spearman correlation between the two distance columns.
    """
    rows = [
        {
            "well_a": a,
            "well_b": b,
            "ref_distance": float(np.linalg.norm(reference.loc[a] - reference.loc[b])),
            "test_distance": float(np.linalg.norm(test.loc[a] - test.loc[b])),
        }
        for a, b in combinations(reference.index, 2)
    ]
    out = pd.DataFrame(rows)
    return out, float(spearmanr(out["ref_distance"], out["test_distance"])[0])
