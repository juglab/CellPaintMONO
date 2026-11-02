"""Analysis steps of a feature fidelity comparison.

`prepare` loads both runs once; each `run_*` function computes the tables of one
analysis and returns them by name. The numbered scripts in `scripts/fidelity/`
call these functions and write the tables and figures.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pandas as pd

from . import distances as dist
from .agreement import family_table, spearman_per_feature
from .config import FidelityConfig
from .features import (
    RobustScaler,
    annotate_features,
    detect_channels,
    filter_for_distances,
    select_features,
)
from .io import (
    IMAGE_ID,
    WELL_ID,
    CellProfilerRun,
    common_images,
    load_run,
    restrict_to_images,
)
from .matching import (
    match_objects,
    matched_mask,
    pair_through_parent,
    paired_values,
    segmentation_summary,
)
from .profiles import (
    distance_geometry,
    hit_rate,
    remove_global_offset,
    retrieval,
    standardise,
    well_profiles,
)
from .spread import spread_ratios

UNIVARIATE_DISTORTION_FAMILIES = ("Intensity", "Granularity", "Radial distrib.")


@dataclass
class FidelityData:
    """Both runs, aligned and prepared for every analysis.

    Attributes
    ----------
    config : FidelityConfig
        The configuration.
    reference : CellProfilerRun
        Reference run, restricted to shared images.
    test : CellProfilerRun
        Test run, restricted to shared images.
    image_ids : list of str
        Shared images in reference order.
    channels : list of str
        Channel names.
    features : dict of str to list of str
        Per compartment: all features (per-object agreement).
    distance_features : dict of str to list of str
        Per compartment: features with non-zero reference MAD and few missing
        values (distances and profiles).
    scalers : dict of str to RobustScaler
        Per compartment: robust scaling fitted on the reference.
    object_pairs : dict of str to pandas.DataFrame
        Per compartment: pairs from independent centroid matching.
    paired_objects : dict of str to pandas.DataFrame
        Per compartment: pairs derived from the matched primary objects.
    """

    config: FidelityConfig
    reference: CellProfilerRun
    test: CellProfilerRun
    image_ids: list[str]
    channels: list[str]
    features: dict[str, list[str]]
    distance_features: dict[str, list[str]]
    scalers: dict[str, RobustScaler]
    object_pairs: dict[str, pd.DataFrame]
    paired_objects: dict[str, pd.DataFrame]

    def z(self, compartment: str, series: str) -> npt.NDArray:
        """Robust z-scores of one compartment and series.

        Parameters
        ----------
        compartment : str
            Compartment name.
        series : {"ref", "test"}
            Which run.

        Returns
        -------
        numpy.ndarray
            Shape (objects, distance features).
        """
        run = self.reference if series == "ref" else self.test
        return self.scalers[compartment].transform(run.objects[compartment])

    def annotations(self, compartment: str, distance: bool = False) -> pd.DataFrame:
        """Family and channel of each feature of a compartment.

        Parameters
        ----------
        compartment : str
            Compartment name.
        distance : bool
            Annotate the distance features instead of all features.

        Returns
        -------
        pandas.DataFrame
            Output of `annotate_features` with a `compartment` column.
        """
        feats = (self.distance_features if distance else self.features)[compartment]
        out = annotate_features(feats, self.channels)
        out.insert(0, "compartment", compartment)
        return out


def prepare(config: FidelityConfig) -> FidelityData:
    """Load both runs, select features and match objects.

    Parameters
    ----------
    config : FidelityConfig
        The configuration.

    Returns
    -------
    FidelityData
        Everything the analysis steps need.
    """
    load = {
        "compartments": config.compartments,
        "image_keys": config.image_keys,
        "well_keys": config.well_keys,
    }
    reference = load_run(config.reference_dir, **load)
    test = load_run(config.test_dir, **load)
    image_ids = common_images(reference, test)
    reference = restrict_to_images(reference, image_ids)
    test = restrict_to_images(test, image_ids)

    primary = config.primary_compartment
    channels = config.channels or detect_channels(
        reference.objects[primary], config.exclude_images
    )
    features, distance_features, scalers = {}, {}, {}
    for comp in config.compartments:
        ref, tst = reference.objects[comp], test.objects[comp]
        features[comp] = select_features(ref, tst, config.exclude_images)
        distance_features[comp] = filter_for_distances(
            ref, features[comp], config.max_missing
        )
        scalers[comp] = RobustScaler.fit(ref, distance_features[comp], config.clip)

    object_pairs = {
        comp: match_objects(
            reference.objects[comp],
            test.objects[comp],
            config.max_match_distance[comp],
            image_ids,
        )
        for comp in config.compartments
    }
    paired_objects = {}
    for comp in config.compartments:
        if comp == primary:
            paired_objects[comp] = object_pairs[primary]
        else:
            paired_objects[comp] = pair_through_parent(
                object_pairs[primary],
                reference.objects[comp],
                test.objects[comp],
                f"Parent_{primary}",
            )
    return FidelityData(
        config=config,
        reference=reference,
        test=test,
        image_ids=image_ids,
        channels=channels,
        features=features,
        distance_features=distance_features,
        scalers=scalers,
        object_pairs=object_pairs,
        paired_objects=paired_objects,
    )


def run_segmentation(data: FidelityData) -> dict[str, pd.DataFrame]:
    """Object counts, matching and sizes of matched and unmatched objects.

    Parameters
    ----------
    data : FidelityData
        Prepared data.

    Returns
    -------
    dict of str to pandas.DataFrame
        `segmentation_per_image`, `object_pairs` (independent centroid matches
        of every compartment) and `primary_object_sizes` (area and match status
        of every primary object).
    """
    primary = data.config.primary_compartment
    pairs = data.object_pairs[primary]
    per_image = segmentation_summary(data.reference, data.test, pairs, primary)
    object_pairs = pd.concat(
        [p.assign(compartment=c) for c, p in data.object_pairs.items()],
        ignore_index=True,
    )
    sizes = []
    for side, run in (("ref", data.reference), ("test", data.test)):
        table = run.objects[primary]
        matched = matched_mask(table, pairs, side)
        sizes.append(
            pd.DataFrame(
                {
                    "series": side,
                    IMAGE_ID: table[IMAGE_ID],
                    "area": table["AreaShape_Area"],
                    "matched": matched,
                }
            )
        )
    return {
        "segmentation_per_image": per_image,
        "object_pairs": object_pairs,
        "primary_object_sizes": pd.concat(sizes, ignore_index=True),
    }


def _paired(data: FidelityData, comp: str, features: list[str]):
    return paired_values(
        data.reference.objects[comp],
        data.test.objects[comp],
        data.paired_objects[comp],
        features,
    )


def _split_feature(entry: str) -> tuple[str, str]:
    comp, feature = entry.split(":", 1)
    return comp, feature


def run_agreement(data: FidelityData) -> dict[str, pd.DataFrame]:
    """Spearman correlation per feature over matched objects.

    Parameters
    ----------
    data : FidelityData
        Prepared data.

    Returns
    -------
    dict of str to pandas.DataFrame
        `agreement_per_feature`, `agreement_family` (median rho per family and
        compartment) and `agreement_examples` (paired values of the example
        features, long format).
    """
    tables = []
    for comp in data.config.compartments:
        ref, tst = _paired(data, comp, data.features[comp])
        ann = data.annotations(comp)
        ann["rho"] = spearman_per_feature(ref, tst)
        ann["n_pairs"] = len(ref)
        tables.append(ann)
    per_feature = pd.concat(tables, ignore_index=True)

    examples = []
    for entry in data.config.example_features:
        comp, feature = _split_feature(entry)
        if comp not in data.features or feature not in data.features[comp]:
            continue
        ref, tst = _paired(data, comp, [feature])
        examples.append(
            pd.DataFrame(
                {
                    "compartment": comp,
                    "feature": feature,
                    "ref": ref[:, 0],
                    "test": tst[:, 0],
                }
            )
        )
    return {
        "agreement_per_feature": per_feature,
        "agreement_family": family_table(per_feature, "rho", data.config.compartments),
        "agreement_examples": pd.concat(examples, ignore_index=True)
        if examples
        else pd.DataFrame(columns=["compartment", "feature", "ref", "test"]),
    }


def _well_table(data: FidelityData) -> pd.DataFrame:
    plate_key = data.config.well_keys[0]
    wells = data.reference.image.drop_duplicates(WELL_ID)
    return pd.DataFrame({"well_id": wells[WELL_ID], "plate": wells[plate_key]})


def run_distances(data: FidelityData) -> dict[str, pd.DataFrame]:
    """Calibrated per-feature and multivariate distances.

    Parameters
    ----------
    data : FidelityData
        Prepared data.

    Returns
    -------
    dict of str to pandas.DataFrame
        `distortion_per_feature`, `distortion_noise_family` (median distance /
        noise per family), `distortion_well_channel` (median fraction of a well
        difference per channel, intensity, granularity and radial features),
        `multivariate_per_image`, `multivariate_well_pairs`,
        `multivariate_summary`, `worked_example` and `worked_example_null`.
    """
    cfg = data.config
    rng = np.random.default_rng(cfg.seed)
    pairs = dist.well_pairs(_well_table(data), cfg.well_pairs, cfg.max_well_pairs, rng)

    per_feature, mv_images, mv_pairs, examples, example_null = [], [], [], [], []
    for comp in cfg.compartments:
        z_ref, z_test = data.z(comp, "ref"), data.z(comp, "test")
        ref_obj, test_obj = data.reference.objects[comp], data.test.objects[comp]
        ref_images = ref_obj[IMAGE_ID].to_numpy()
        test_images = test_obj[IMAGE_ID].to_numpy()
        ref_wells = ref_obj[WELL_ID].to_numpy()

        distance, noise = dist.per_image_distances(
            z_ref, z_test, ref_images, test_images, data.image_ids, rng, cfg.n_shuffles
        )
        well_distance, well_noise = dist.well_yardstick(
            z_ref, ref_wells, pairs, rng, cfg.n_shuffles
        )
        table = dist.distortion_table(
            distance, noise, well_distance, well_noise, data.distance_features[comp]
        )
        per_feature.append(data.annotations(comp, distance=True).merge(table))

        for entry in cfg.worked_example_features:
            ex_comp, feature = _split_feature(entry)
            if ex_comp != comp or feature not in data.distance_features[comp]:
                continue
            ex, null = _worked_example(
                data, comp, feature, z_ref, z_test, table, pairs, rng
            )
            examples.append(ex)
            example_null.append(null)

        pca = dist.fit_pca(z_ref, cfg.n_components, cfg.seed)
        e_ref = pca.transform(np.nan_to_num(z_ref))
        e_test = pca.transform(np.nan_to_num(z_test))
        images = dist.multivariate_per_image(
            e_ref,
            e_test,
            ref_images,
            test_images,
            data.image_ids,
            rng,
            cfg.n_shuffles_multivariate,
            cfg.max_cells_multivariate,
        )
        images.insert(0, "compartment", comp)
        images["explained_variance"] = pca.explained_variance_ratio_.sum()
        mv_images.append(images)
        yard = dist.multivariate_yardstick(
            e_ref,
            ref_wells,
            pairs,
            rng,
            cfg.n_shuffles_multivariate,
            cfg.max_cells_multivariate,
        )
        yard.insert(0, "compartment", comp)
        mv_pairs.append(yard)

    per_feature_df = pd.concat(per_feature, ignore_index=True)
    mv_images_df = pd.concat(mv_images, ignore_index=True)
    mv_pairs_df = pd.concat(mv_pairs, ignore_index=True)
    return {
        "distortion_per_feature": per_feature_df,
        "distortion_noise_family": family_table(
            per_feature_df, "ratio_to_noise", cfg.compartments
        ),
        "distortion_well_channel": distortion_by_channel(
            per_feature_df, cfg.compartments, data.channels
        ),
        "multivariate_per_image": mv_images_df,
        "multivariate_well_pairs": mv_pairs_df,
        "multivariate_summary": multivariate_summary(
            mv_images_df, mv_pairs_df, cfg.compartments
        ),
        "worked_example": pd.DataFrame(examples),
        "worked_example_null": pd.concat(example_null, ignore_index=True)
        if example_null
        else pd.DataFrame(columns=["compartment", "feature", "kind", "value"]),
    }


def _worked_example(
    data: FidelityData,
    comp: str,
    feature: str,
    z_ref: npt.NDArray,
    z_test: npt.NDArray,
    table: pd.DataFrame,
    pairs: list[tuple[str, str]],
    rng: np.random.Generator,
    n_null: int = 200,
) -> tuple[dict, pd.DataFrame]:
    cfg = data.config
    image_id = cfg.worked_example_image or data.image_ids[0]
    j = data.distance_features[comp].index(feature)
    ref_obj, test_obj = data.reference.objects[comp], data.test.objects[comp]
    a = z_ref[ref_obj[IMAGE_ID].to_numpy() == image_id, j]
    b = z_test[test_obj[IMAGE_ID].to_numpy() == image_id, j]
    observed = dist.w1(a, b)
    null = dist.w1_shuffle_null(a, b, rng, n_null)
    well = ref_obj.loc[ref_obj[IMAGE_ID] == image_id, WELL_ID].iloc[0]
    ref_wells = ref_obj[WELL_ID].to_numpy()
    others = sorted({w for p in pairs if well in p for w in p} - {well})
    other = [
        dist.w1(z_ref[ref_wells == well, j], z_ref[ref_wells == w, j]) for w in others
    ]
    row = table.set_index("feature").loc[feature]
    summary = {
        "compartment": comp,
        "feature": feature,
        IMAGE_ID: image_id,
        "distance": observed,
        "noise": null.mean(),
        "ratio_to_noise": observed / null.mean(),
        "fraction_of_well": (observed - null.mean())
        / (row["well_distance"] - row["well_noise"]),
    }
    long = pd.concat(
        [
            pd.DataFrame({"kind": "null", "value": null}),
            pd.DataFrame({"kind": "other_well", "value": other}),
        ],
        ignore_index=True,
    ).assign(compartment=comp, feature=feature)
    return summary, long


def distortion_by_channel(
    per_feature: pd.DataFrame, compartments: list[str], channels: list[str]
) -> pd.DataFrame:
    """Median fraction of a well difference per channel and compartment.

    Parameters
    ----------
    per_feature : pandas.DataFrame
        `distortion_per_feature` table.
    compartments : list of str
        Column order.
    channels : list of str
        Row order.

    Returns
    -------
    pandas.DataFrame
        Channels as rows, compartments as columns; intensity, granularity and
        radial distribution features only.
    """
    sub = per_feature[per_feature["family"].isin(UNIVARIATE_DISTORTION_FAMILIES)]
    return family_table(
        sub.dropna(subset=["channel"]),
        "fraction_of_well",
        compartments,
        column="channel",
        order=channels,
    )


def multivariate_summary(
    per_image: pd.DataFrame, well_pairs: pd.DataFrame, compartments: list[str]
) -> pd.DataFrame:
    """Multivariate distances above the noise floor per compartment.

    Parameters
    ----------
    per_image : pandas.DataFrame
        `multivariate_per_image` table.
    well_pairs : pandas.DataFrame
        `multivariate_well_pairs` table.
    compartments : list of str
        Row order.

    Returns
    -------
    pandas.DataFrame
        Per compartment: mean distance, noise and ratio, mean excess over the
        noise floor for reference vs test and for well pairs, their ratio, and
        the number of images with p < 0.05.
    """
    img = per_image.assign(excess=per_image["distance"] - per_image["noise"])
    wp = well_pairs.assign(excess=well_pairs["distance"] - well_pairs["noise"])
    out = pd.DataFrame(
        {
            "distance": img.groupby("compartment")["distance"].mean(),
            "noise": img.groupby("compartment")["noise"].mean(),
            "excess": img.groupby("compartment")["excess"].mean(),
            "well_excess": wp.groupby("compartment")["excess"].mean(),
            "n_significant": img.assign(sig=img["p_value"] < 0.05)
            .groupby("compartment")["sig"]
            .sum(),
            "n_images": img.groupby("compartment").size(),
            "explained_variance": img.groupby("compartment")[
                "explained_variance"
            ].first(),
        }
    ).reindex(compartments)
    out["ratio_to_noise"] = out["distance"] / out["noise"]
    out["fraction_of_well"] = out["excess"] / out["well_excess"]
    return out


def run_profiles(data: FidelityData) -> dict[str, pd.DataFrame]:
    """Well profiles: identity retrieval before and after offset removal.

    Profiles are the median robust z-score of every distance feature of every
    compartment over the cells of a well, standardised across wells with the
    reference mean and SD.

    Parameters
    ----------
    data : FidelityData
        Prepared data.

    Returns
    -------
    dict of str to pandas.DataFrame
        `profile_correlation`, `profile_correlation_offset_removed`,
        `profile_distances` and `profile_summary`.
    """
    ref_parts, test_parts = [], []
    for comp in data.config.compartments:
        feats = data.distance_features[comp]
        ref_wells = data.reference.objects[comp][WELL_ID].to_numpy()
        test_wells = data.test.objects[comp][WELL_ID].to_numpy()
        ref_parts.append(well_profiles(data.z(comp, "ref"), ref_wells, feats, comp))
        test_parts.append(well_profiles(data.z(comp, "test"), test_wells, feats, comp))
    order = list(dict.fromkeys(data.reference.image[WELL_ID]))
    ref = pd.concat(ref_parts, axis=1).loc[order]
    test = pd.concat(test_parts, axis=1)
    ref, test = standardise(ref, test)
    corr = retrieval(ref, test)
    corr_offset = retrieval(ref, remove_global_offset(ref, test))
    distances, rho = distance_geometry(ref, test)
    summary = pd.DataFrame(
        {
            "n_wells": [len(ref)],
            "n_features": [ref.shape[1]],
            "hit_rate": [hit_rate(corr)],
            "hit_rate_offset_removed": [hit_rate(corr_offset)],
            "distance_spearman": [rho],
        }
    )
    return {
        "profile_correlation": corr,
        "profile_correlation_offset_removed": corr_offset,
        "profile_distances": distances,
        "profile_summary": summary,
    }


def run_spread(data: FidelityData) -> dict[str, pd.DataFrame]:
    """Spread ratios and intensity shifts of matched objects.

    Parameters
    ----------
    data : FidelityData
        Prepared data.

    Returns
    -------
    dict of str to pandas.DataFrame
        `spread_ratios` (per compartment and channel) and `spread_values`
        (mean intensity of every matched object in the spread compartment,
        long format).
    """
    ratios, values = [], []
    for comp in data.config.compartments:
        feats = [
            f
            for ch in data.channels
            for f in (f"Intensity_MeanIntensity_{ch}", f"Intensity_StdIntensity_{ch}")
            if f in data.features[comp]
        ]
        ref, tst = _paired(data, comp, feats)
        ref_df, test_df = (
            pd.DataFrame(ref, columns=feats),
            pd.DataFrame(tst, columns=feats),
        )
        chans = [ch for ch in data.channels if f"Intensity_StdIntensity_{ch}" in feats]
        ratios.append(spread_ratios(ref_df, test_df, chans).assign(compartment=comp))
        if comp == data.config.spread_compartment:
            for ch in chans:
                col = f"Intensity_MeanIntensity_{ch}"
                for series, df in (("ref", ref_df), ("test", test_df)):
                    values.append(
                        pd.DataFrame(
                            {"channel": ch, "series": series, "value": df[col]}
                        )
                    )
    return {
        "spread_ratios": pd.concat(ratios, ignore_index=True),
        "spread_values": pd.concat(values, ignore_index=True),
    }


def write_tables(tables: dict[str, pd.DataFrame], folder: str | Path) -> None:
    """Write tables as CSV files named after their keys.

    Parameters
    ----------
    tables : dict of str to pandas.DataFrame
        Tables to write.
    folder : str or pathlib.Path
        Output folder, created if needed.
    """
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        keep_index = not isinstance(table.index, pd.RangeIndex)
        table.to_csv(folder / f"{name}.csv", index=keep_index)


def worked_example_values(
    data: FidelityData, worked: pd.DataFrame
) -> dict[str, tuple[npt.NDArray, npt.NDArray]]:
    """Robust z-scores of the worked example features on the example image.

    Parameters
    ----------
    data : FidelityData
        Prepared data.
    worked : pandas.DataFrame
        `worked_example` table.

    Returns
    -------
    dict
        Feature name to (reference, test) values.
    """
    out = {}
    for row in worked.itertuples():
        j = data.distance_features[row.compartment].index(row.feature)
        z = {}
        for series, run in (("ref", data.reference), ("test", data.test)):
            mask = run.objects[row.compartment][IMAGE_ID].to_numpy() == row.image_id
            z[series] = data.z(row.compartment, series)[mask, j]
        out[row.feature] = (z["ref"], z["test"])
    return out
