"""Distribution distances calibrated by a noise floor and a well-to-well yardstick.

All distances are computed on robust z-scores, so a distance of 0.5 means that
cells move on average half a typical reference cell-to-cell spread.

- Noise floor: the distance after shuffling reference/test labels within an image,
  i.e. two samples of the same population at this sample size.
- Well-to-well yardstick: the distance between reference cells of two wells minus
  their own shuffle null, i.e. a typical biological (and plate) difference.
"""

from __future__ import annotations

from itertools import combinations

import numpy as np
import numpy.typing as npt
import ot
import pandas as pd
from scipy.stats import wasserstein_distance
from sklearn.decomposition import PCA


def w1(a: npt.NDArray, b: npt.NDArray) -> float:
    """One-dimensional Wasserstein-1 distance, ignoring NaN.

    Parameters
    ----------
    a : numpy.ndarray
        First sample.
    b : numpy.ndarray
        Second sample.

    Returns
    -------
    float
        Distance, or NaN if either sample is empty.
    """
    a, b = a[~np.isnan(a)], b[~np.isnan(b)]
    if len(a) == 0 or len(b) == 0:
        return np.nan
    return float(wasserstein_distance(a, b))


def w1_shuffle_null(
    a: npt.NDArray, b: npt.NDArray, rng: np.random.Generator, n_shuffles: int
) -> npt.NDArray:
    """W1 distances after shuffling the labels of two samples.

    Parameters
    ----------
    a : numpy.ndarray
        First sample.
    b : numpy.ndarray
        Second sample.
    rng : numpy.random.Generator
        Random generator.
    n_shuffles : int
        Number of shuffles.

    Returns
    -------
    numpy.ndarray
        One distance per shuffle; group sizes are kept.
    """
    a, b = a[~np.isnan(a)], b[~np.isnan(b)]
    pool = np.concatenate([a, b])
    out = np.empty(n_shuffles)
    for k in range(n_shuffles):
        perm = rng.permutation(pool)
        out[k] = w1(perm[: len(a)], perm[len(a) :])
    return out


def per_image_distances(
    z_ref: npt.NDArray,
    z_test: npt.NDArray,
    ref_images: npt.NDArray,
    test_images: npt.NDArray,
    image_ids: list[str],
    rng: np.random.Generator,
    n_shuffles: int,
) -> tuple[npt.NDArray, npt.NDArray]:
    """Reference vs test W1 and its noise floor per image and feature.

    Parameters
    ----------
    z_ref : numpy.ndarray
        Reference robust z-scores, shape (cells, features).
    z_test : numpy.ndarray
        Test robust z-scores, shape (cells, features).
    ref_images : numpy.ndarray
        Image id of each reference cell.
    test_images : numpy.ndarray
        Image id of each test cell.
    image_ids : list of str
        Images to compare.
    rng : numpy.random.Generator
        Random generator.
    n_shuffles : int
        Shuffles per image and feature for the noise floor.

    Returns
    -------
    tuple of numpy.ndarray
        Distance and mean shuffle distance, each shape (images, features).
    """
    n_features = z_ref.shape[1]
    distance = np.full((len(image_ids), n_features), np.nan)
    noise = np.full((len(image_ids), n_features), np.nan)
    for i, image_id in enumerate(image_ids):
        a, b = z_ref[ref_images == image_id], z_test[test_images == image_id]
        for j in range(n_features):
            distance[i, j] = w1(a[:, j], b[:, j])
            noise[i, j] = w1_shuffle_null(a[:, j], b[:, j], rng, n_shuffles).mean()
    return distance, noise


def well_pairs(
    wells: pd.DataFrame,
    mode: str,
    max_pairs: int | None,
    rng: np.random.Generator,
) -> list[tuple[str, str]]:
    """Pairs of different wells for the yardstick.

    Parameters
    ----------
    wells : pandas.DataFrame
        One row per well with columns `well_id` and `plate`.
    mode : {"all", "same_plate"}
        All pairs of wells, or only pairs on the same plate.
    max_pairs : int or None
        Random subset size; all pairs if None.
    rng : numpy.random.Generator
        Random generator.

    Returns
    -------
    list of tuple of str
        Well id pairs.

    Raises
    ------
    ValueError
        If no pair qualifies.
    """
    plate = dict(zip(wells["well_id"], wells["plate"], strict=True))
    pairs = list(combinations(wells["well_id"], 2))
    if mode == "same_plate":
        pairs = [p for p in pairs if plate[p[0]] == plate[p[1]]]
    if not pairs:
        raise ValueError(f"No well pairs for mode {mode!r}")
    if max_pairs is not None and len(pairs) > max_pairs:
        idx = rng.choice(len(pairs), max_pairs, replace=False)
        pairs = [pairs[i] for i in sorted(idx)]
    return pairs


def well_yardstick(
    z_ref: npt.NDArray,
    ref_wells: npt.NDArray,
    pairs: list[tuple[str, str]],
    rng: np.random.Generator,
    n_shuffles: int,
) -> tuple[npt.NDArray, npt.NDArray]:
    """W1 between reference cells of two wells and its shuffle null.

    Parameters
    ----------
    z_ref : numpy.ndarray
        Reference robust z-scores, shape (cells, features).
    ref_wells : numpy.ndarray
        Well id of each reference cell.
    pairs : list of tuple of str
        Well pairs.
    rng : numpy.random.Generator
        Random generator.
    n_shuffles : int
        Shuffles per pair and feature.

    Returns
    -------
    tuple of numpy.ndarray
        Distance and mean shuffle distance, each shape (pairs, features).
    """
    n_features = z_ref.shape[1]
    distance = np.full((len(pairs), n_features), np.nan)
    noise = np.full((len(pairs), n_features), np.nan)
    for k, (well_a, well_b) in enumerate(pairs):
        a, b = z_ref[ref_wells == well_a], z_ref[ref_wells == well_b]
        for j in range(n_features):
            distance[k, j] = w1(a[:, j], b[:, j])
            noise[k, j] = w1_shuffle_null(a[:, j], b[:, j], rng, n_shuffles).mean()
    return distance, noise


def distortion_table(
    distance: npt.NDArray,
    noise: npt.NDArray,
    well_distance: npt.NDArray,
    well_noise: npt.NDArray,
    features: list[str],
) -> pd.DataFrame:
    """Per-feature distortion relative to the noise floor and the yardstick.

    Parameters
    ----------
    distance : numpy.ndarray
        Reference vs test distance per image, shape (images, features).
    noise : numpy.ndarray
        Noise floor per image, same shape.
    well_distance : numpy.ndarray
        Reference well-pair distance, shape (pairs, features).
    well_noise : numpy.ndarray
        Shuffle null of each well pair, same shape.
    features : list of str
        Feature names.

    Returns
    -------
    pandas.DataFrame
        Per feature: mean `distance` and `noise` over images,
        `ratio_to_noise` = distance / noise, mean `well_distance` and
        `well_noise` over pairs, and `fraction_of_well` =
        (distance - noise) / (well_distance - well_noise).
    """
    out = pd.DataFrame(
        {
            "feature": features,
            "distance": np.nanmean(distance, axis=0),
            "noise": np.nanmean(noise, axis=0),
            "well_distance": np.nanmean(well_distance, axis=0),
            "well_noise": np.nanmean(well_noise, axis=0),
        }
    )
    out["ratio_to_noise"] = out["distance"] / out["noise"]
    out["fraction_of_well"] = (out["distance"] - out["noise"]) / (
        out["well_distance"] - out["well_noise"]
    )
    return out


def fit_pca(z_ref: npt.NDArray, n_components: int, seed: int) -> PCA:
    """PCA fitted on reference cells.

    Parameters
    ----------
    z_ref : numpy.ndarray
        Reference robust z-scores; NaN is treated as the median (0).
    n_components : int
        Number of components.
    seed : int
        Seed for the randomised solver, if used.

    Returns
    -------
    sklearn.decomposition.PCA
        Fitted model.
    """
    return PCA(n_components, random_state=seed).fit(np.nan_to_num(z_ref))


def ot_distance(a: npt.NDArray, b: npt.NDArray) -> float:
    """Exact Wasserstein-1 distance between two point clouds.

    Parameters
    ----------
    a : numpy.ndarray
        First cloud, shape (n, dims).
    b : numpy.ndarray
        Second cloud, shape (m, dims).

    Returns
    -------
    float
        Optimal transport cost with Euclidean ground cost and uniform weights.
    """
    cost = ot.dist(a, b, metric="euclidean")
    return float(
        ot.emd2(np.full(len(a), 1 / len(a)), np.full(len(b), 1 / len(b)), cost)
    )


def ot_shuffle_null(
    a: npt.NDArray, b: npt.NDArray, rng: np.random.Generator, n_shuffles: int
) -> npt.NDArray:
    """Optimal transport distances after shuffling the labels of two clouds.

    Parameters
    ----------
    a : numpy.ndarray
        First cloud.
    b : numpy.ndarray
        Second cloud.
    rng : numpy.random.Generator
        Random generator.
    n_shuffles : int
        Number of shuffles.

    Returns
    -------
    numpy.ndarray
        One distance per shuffle.
    """
    pool = np.vstack([a, b])
    cost = ot.dist(pool, pool, metric="euclidean")
    n = len(a)
    wa, wb = np.full(n, 1 / n), np.full(len(b), 1 / len(b))
    out = np.empty(n_shuffles)
    for k in range(n_shuffles):
        perm = rng.permutation(len(pool))
        out[k] = ot.emd2(wa, wb, cost[np.ix_(perm[:n], perm[n:])])
    return out


def _subsample(
    x: npt.NDArray, max_cells: int | None, rng: np.random.Generator
) -> npt.NDArray:
    if max_cells is None or len(x) <= max_cells:
        return x
    return x[rng.choice(len(x), max_cells, replace=False)]


def multivariate_per_image(
    e_ref: npt.NDArray,
    e_test: npt.NDArray,
    ref_images: npt.NDArray,
    test_images: npt.NDArray,
    image_ids: list[str],
    rng: np.random.Generator,
    n_shuffles: int,
    max_cells: int | None,
) -> pd.DataFrame:
    """Multivariate reference vs test distance per image with a permutation test.

    Parameters
    ----------
    e_ref : numpy.ndarray
        Reference cells in PCA space.
    e_test : numpy.ndarray
        Test cells in PCA space.
    ref_images : numpy.ndarray
        Image id of each reference cell.
    test_images : numpy.ndarray
        Image id of each test cell.
    image_ids : list of str
        Images to compare.
    rng : numpy.random.Generator
        Random generator.
    n_shuffles : int
        Label shuffles per image.
    max_cells : int or None
        Cells subsampled per image and series.

    Returns
    -------
    pandas.DataFrame
        Per image: `distance`, `noise` (mean shuffle distance), `p_value`.
    """
    rows = []
    for image_id in image_ids:
        a = _subsample(e_ref[ref_images == image_id], max_cells, rng)
        b = _subsample(e_test[test_images == image_id], max_cells, rng)
        observed = ot_distance(a, b)
        null = ot_shuffle_null(a, b, rng, n_shuffles)
        rows.append(
            {
                "image_id": image_id,
                "distance": observed,
                "noise": null.mean(),
                "p_value": (1 + (null >= observed).sum()) / (1 + n_shuffles),
            }
        )
    return pd.DataFrame(rows)


def multivariate_yardstick(
    e_ref: npt.NDArray,
    ref_wells: npt.NDArray,
    pairs: list[tuple[str, str]],
    rng: np.random.Generator,
    n_shuffles: int,
    max_cells: int | None,
) -> pd.DataFrame:
    """Multivariate distance between reference cells of two wells.

    Parameters
    ----------
    e_ref : numpy.ndarray
        Reference cells in PCA space.
    ref_wells : numpy.ndarray
        Well id of each reference cell.
    pairs : list of tuple of str
        Well pairs.
    rng : numpy.random.Generator
        Random generator.
    n_shuffles : int
        Label shuffles per pair.
    max_cells : int or None
        Cells subsampled per well.

    Returns
    -------
    pandas.DataFrame
        Per pair: `well_a`, `well_b`, `distance`, `noise`.
    """
    rows = []
    for well_a, well_b in pairs:
        a = _subsample(e_ref[ref_wells == well_a], max_cells, rng)
        b = _subsample(e_ref[ref_wells == well_b], max_cells, rng)
        rows.append(
            {
                "well_a": well_a,
                "well_b": well_b,
                "distance": ot_distance(a, b),
                "noise": ot_shuffle_null(a, b, rng, n_shuffles).mean(),
            }
        )
    return pd.DataFrame(rows)
