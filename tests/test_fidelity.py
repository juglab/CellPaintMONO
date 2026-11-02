import numpy as np
import pandas as pd
import pytest

from cellpaintmono.fidelity import distances as dist
from cellpaintmono.fidelity.features import (
    RobustScaler,
    annotate_features,
    detect_channels,
    feature_family,
    filter_for_distances,
    is_feature,
)
from cellpaintmono.fidelity.matching import match_objects, pair_through_parent
from cellpaintmono.fidelity.profiles import (
    distance_geometry,
    hit_rate,
    remove_global_offset,
    retrieval,
)

CHANNELS = ["DNA", "ER", "RNA", "AGP", "Mito"]


def objects(image_id, xy):
    xy = np.asarray(xy, dtype=float)
    return pd.DataFrame(
        {
            "image_id": image_id,
            "ObjectNumber": np.arange(1, len(xy) + 1),
            "Location_Center_X": xy[:, 0],
            "Location_Center_Y": xy[:, 1],
        }
    )


def test_is_feature_drops_bookkeeping():
    assert is_feature("Intensity_MeanIntensity_DNA", ["RescaleIntensity"])
    for name in [
        "Location_Center_X",
        "AreaShape_BoundingBoxArea",
        "AreaShape_Center_X",
        "AreaShape_Orientation",
        "Neighbors_FirstClosestObjectNumber_Adjacent",
        "Parent_Nuclei",
        "Intensity_MeanIntensity_RescaleIntensity",
    ]:
        assert not is_feature(name, ["RescaleIntensity"])


def test_family_and_channels():
    ann = annotate_features(
        [
            "AreaShape_Zernike_2_0",
            "AreaShape_Area",
            "Granularity_3_Mito",
            "Correlation_K_AGP_RNA",
            "RadialDistribution_FracAtD_ER_2of4",
        ],
        CHANNELS,
    ).set_index("feature")
    assert ann.loc["AreaShape_Zernike_2_0", "family"] == "Zernike shape"
    assert ann.loc["AreaShape_Area", "channel"] is None
    assert ann.loc["Granularity_3_Mito", "channel"] == "Mito"
    assert ann.loc["Granularity_3_Mito", "scale"] == 3
    assert ann.loc["Correlation_K_AGP_RNA", "channels"] == "AGP-RNA"
    assert ann.loc["Correlation_K_AGP_RNA", "channel"] is None
    assert ann.loc["RadialDistribution_FracAtD_ER_2of4", "channel"] == "ER"
    assert feature_family("Neighbors_NumberOfNeighbors_5") == "Neighbours"


def test_detect_channels_excludes_helpers():
    df = pd.DataFrame(
        columns=[
            "Intensity_MeanIntensity_DNA",
            "Intensity_MeanIntensity_RescaleIntensity",
            "Intensity_MeanIntensity_AO_green",
        ]
    )
    assert detect_channels(df, ["RescaleIntensity"]) == ["DNA", "AO_green"]


def test_filter_and_robust_scaler():
    rng = np.random.default_rng(0)
    ref = pd.DataFrame(
        {
            "a": rng.normal(10, 2, 500),
            "constant": np.r_[np.ones(499), 5.0],
            "missing": np.r_[np.full(100, np.nan), rng.normal(size=400)],
        }
    )
    keep = filter_for_distances(ref, list(ref), max_missing=0.05)
    assert keep == ["a"]
    scaler = RobustScaler.fit(ref, keep, clip=3)
    z = scaler.transform(ref)
    assert abs(np.median(z)) < 1e-9
    assert np.median(np.abs(z)) * 1.4826 == pytest.approx(1, rel=1e-6)
    assert np.abs(z).max() <= 3


def test_match_objects_drops_far_pairs():
    ref = objects("img", [[0, 0], [100, 100], [200, 200]])
    test = objects("img", [[1, 0], [100, 103], [260, 200], [400, 400]])
    pairs = match_objects(ref, test, max_distance=8, image_ids=["img"])
    assert list(zip(pairs["ref_object"], pairs["test_object"], strict=True)) == [
        (1, 1),
        (2, 2),
    ]
    assert pairs["distance"].tolist() == pytest.approx([1, 3])


def test_pair_through_parent():
    primary = pd.DataFrame(
        {
            "image_id": "img",
            "ref_object": [1, 2],
            "test_object": [2, 3],
            "distance": 0.5,
        }
    )
    ref = pd.DataFrame(
        {"image_id": "img", "ObjectNumber": [10, 11], "Parent_Nuclei": [1, 2]}
    )
    test = pd.DataFrame(
        {"image_id": "img", "ObjectNumber": [20, 21], "Parent_Nuclei": [2, 9]}
    )
    pairs = pair_through_parent(primary, ref, test, "Parent_Nuclei")
    assert pairs[["ref_object", "test_object"]].values.tolist() == [[10, 20]]


def test_noise_floor_matches_same_population():
    rng = np.random.default_rng(1)
    a, b = rng.normal(size=300), rng.normal(size=300)
    noise = dist.w1_shuffle_null(a, b, rng, 50).mean()
    assert dist.w1(a, b) / noise < 2
    assert dist.w1(a, b + 1) == pytest.approx(1, abs=0.15)


def test_distortion_table_formula():
    out = dist.distortion_table(
        distance=np.array([[0.5], [0.7]]),
        noise=np.array([[0.1], [0.1]]),
        well_distance=np.array([[1.2], [0.8]]),
        well_noise=np.array([[0.2], [0.2]]),
        features=["f"],
    ).iloc[0]
    assert out["ratio_to_noise"] == pytest.approx(6)
    assert out["fraction_of_well"] == pytest.approx(0.5 / 0.8)


def test_well_pairs_same_plate():
    wells = pd.DataFrame({"well_id": ["a", "b", "c"], "plate": ["p1", "p1", "p2"]})
    rng = np.random.default_rng(0)
    assert dist.well_pairs(wells, "all", None, rng) == [
        ("a", "b"),
        ("a", "c"),
        ("b", "c"),
    ]
    assert dist.well_pairs(wells, "same_plate", None, rng) == [("a", "b")]
    assert len(dist.well_pairs(wells, "all", 2, rng)) == 2


def test_ot_distance_shift():
    rng = np.random.default_rng(2)
    a = rng.normal(size=(100, 3))
    assert dist.ot_distance(a, a) == pytest.approx(0, abs=1e-9)
    assert dist.ot_distance(a, a + np.array([3, 0, 0])) == pytest.approx(3)


def test_profiles_offset_and_geometry():
    rng = np.random.default_rng(3)
    ref = pd.DataFrame(rng.normal(size=(6, 50)), index=list("abcdef"))
    test = ref + rng.normal(0, 0.1, ref.shape) + rng.normal(0, 3, 50)
    assert hit_rate(retrieval(ref, remove_global_offset(ref, test))) == 1
    _, rho = distance_geometry(ref, test)
    assert rho > 0.8
