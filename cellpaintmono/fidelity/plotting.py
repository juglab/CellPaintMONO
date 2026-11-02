"""Figures of a feature fidelity comparison.

Each `plot_*` function takes the tables returned by the matching `run_*` step in
`pipeline` and returns a matplotlib figure. Reference = grey, test = blue.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import Rectangle

from .features import ZERNIKE_FAMILY

REF_COLOR = "#9E9E9E"
TEST_COLOR = "#2F6FDB"
LIGHT_TEST_COLOR = "#9DBDF2"
DARK_REF_COLOR = "#7A7A7A"
LIGHT_REF_COLOR = "#CFCFCF"
STRIP_FAMILIES = [
    "Shape",
    "Neighbours",
    "Radial distrib.",
    "Granularity",
    "Texture",
    "Intensity",
    "Colocalisation",
]
STYLE = {
    "font.size": 8,
    "axes.titlesize": 8,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "legend.frameon": False,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.titlelocation": "left",
    "figure.dpi": 110,
    "savefig.dpi": 300,
    "pdf.fonttype": 42,
}


def use_style() -> None:
    """Apply the shared matplotlib style."""
    mpl.rcParams.update(STYLE)


def save_figure(fig: plt.Figure, folder: str | Path, name: str) -> None:
    """Save a figure as PDF and PNG.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figure to save.
    folder : str or pathlib.Path
        Output folder, created if needed.
    name : str
        File name without extension.
    """
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(folder / f"{name}.{ext}", bbox_inches="tight")


def _letter(ax: plt.Axes, letter: str) -> None:
    ax.text(-0.12, 1.12, letter, transform=ax.transAxes, fontweight="bold", va="top")


def _heatmap(
    ax: plt.Axes,
    table: pd.DataFrame,
    cmap: str,
    norm: mpl.colors.Normalize,
    fmt: str = "{:.2f}",
) -> mpl.image.AxesImage:
    values = table.to_numpy(dtype=float)
    image = ax.imshow(values, cmap=cmap, norm=norm, aspect="auto")
    ax.set_xticks(range(table.shape[1]), table.columns)
    ax.set_yticks(range(table.shape[0]), table.index)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0)
    cm = plt.get_cmap(cmap)
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            v = values[i, j]
            if np.isnan(v):
                ax.text(j, i, "-", ha="center", va="center")
                continue
            r, g, b, _ = cm(norm(v))
            dark = 0.299 * r + 0.587 * g + 0.114 * b < 0.55
            ax.text(
                j,
                i,
                fmt.format(v),
                ha="center",
                va="center",
                color="white" if dark else "black",
            )
    return image


def plot_worked_example(
    worked: pd.DataFrame,
    null: pd.DataFrame,
    values: dict[str, tuple[np.ndarray, np.ndarray]],
    labels: tuple[str, str] = ("original", "MicroSplit"),
) -> plt.Figure:
    """Distributions and calibrated distances of example features on one image.

    Parameters
    ----------
    worked : pandas.DataFrame
        `worked_example` table.
    null : pandas.DataFrame
        `worked_example_null` table.
    values : dict
        Feature name to (reference, test) robust z-scores of the example image.
    labels : tuple of str
        Reference and test names.

    Returns
    -------
    matplotlib.figure.Figure
        One row per feature: histograms, then null distribution, observed
        distance and distances to other wells.
    """
    use_style()
    n = len(worked)
    fig, axes = plt.subplots(n, 2, figsize=(9, 3 * n), squeeze=False)
    for k, row in enumerate(worked.itertuples()):
        ref, test = values[row.feature]
        ax = axes[k, 0]
        _letter(ax, "abcdefgh"[2 * k])
        bins = np.histogram_bin_edges(np.concatenate([ref, test]), 30)
        ax.hist(ref, bins, color=REF_COLOR, label=labels[0])
        ax.hist(test, bins, histtype="step", color=TEST_COLOR, lw=1.5, label=labels[1])
        ax.set_title(f"{row.compartment}: {row.feature}")
        ax.set_xlabel("robust z-score (units of the reference spread)")
        ax.set_ylabel("cells")
        if k == 0:
            ax.legend()

        ax = axes[k, 1]
        _letter(ax, "abcdefgh"[2 * k + 1])
        sub = null[null["feature"] == row.feature]
        ax.hist(sub.loc[sub["kind"] == "null", "value"], 20, color=LIGHT_REF_COLOR)
        top = ax.get_ylim()[1]
        for v in sub.loc[sub["kind"] == "other_well", "value"]:
            ax.plot([v, v], [0, 0.6 * top], color="black", lw=1)
        ax.plot([row.distance] * 2, [0, 0.9 * top], color=TEST_COLOR, lw=2)
        ax.text(
            row.distance,
            0.9 * top,
            f" observed\n W = {row.distance:.2f}",
            color=TEST_COLOR,
            va="top",
        )
        ax.set_yticks([])
        ax.spines["left"].set_visible(False)
        ax.set_xlabel("Wasserstein-1 distance (robust z units)")
        ax.set_title(
            f"{row.ratio_to_noise:.1f}x the noise floor; "
            f"{100 * row.fraction_of_well:.0f}% of a typical well-to-well difference"
        )
    fig.tight_layout()
    return fig


def plot_segmentation(
    per_image: pd.DataFrame,
    object_pairs: pd.DataFrame,
    sizes: pd.DataFrame,
    primary: str,
    labels: tuple[str, str] = ("original", "MicroSplit"),
    max_shift: float = 15,
) -> plt.Figure:
    """Object counts, centroid shifts and sizes of unmatched objects.

    Parameters
    ----------
    per_image : pandas.DataFrame
        `segmentation_per_image` table.
    object_pairs : pandas.DataFrame
        `object_pairs` table.
    sizes : pandas.DataFrame
        `primary_object_sizes` table.
    primary : str
        Primary compartment name.
    labels : tuple of str
        Reference and test names.
    max_shift : float
        Upper limit of the shift histogram, in pixels.

    Returns
    -------
    matplotlib.figure.Figure
        Panels a (counts), b (shifts), c (area ECDF).
    """
    use_style()
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.3))
    ax = axes[0]
    _letter(ax, "a")
    change = per_image["count_change"]
    total = per_image["n_test"].sum() / per_image["n_ref"].sum() - 1
    lo, hi = (
        per_image[["n_ref", "n_test"]].min().min(),
        per_image[["n_ref", "n_test"]].max().max(),
    )
    ax.plot([lo, hi], [lo, hi], color=LIGHT_REF_COLOR, lw=1)
    ax.scatter(per_image["n_ref"], per_image["n_test"], color=TEST_COLOR, s=18)
    ax.set_xlabel(f"{primary.lower()} per image, {labels[0]}")
    ax.set_ylabel(f"{primary.lower()} per image, {labels[1]}")
    ax.set_title(
        f"{100 * total:+.0f}% objects (range {100 * change.min():+.0f} "
        f"to {100 * change.max():+.0f}%)"
    )

    ax = axes[1]
    _letter(ax, "b")
    bins = np.linspace(0, max_shift, 31)
    n_ref = sizes[sizes["series"] == "ref"].shape[0]
    titles = []
    for comp, color in zip(
        object_pairs["compartment"].unique()[:2],
        (TEST_COLOR, LIGHT_REF_COLOR),
        strict=False,
    ):
        d = object_pairs.loc[object_pairs["compartment"] == comp, "distance"]
        ax.hist(
            d,
            bins,
            color=color,
            alpha=0.9,
            label=f"{comp} (median {d.median():.1f} px)",
        )
        titles.append(f"{100 * len(d) / n_ref:.0f}% of {comp.lower()}")
    ax.set_xlabel("centroid shift of matched object (px)")
    ax.set_ylabel("objects")
    ax.set_title(", ".join(titles) + " matched")
    ax.legend()

    ax = axes[2]
    _letter(ax, "c")
    groups = [
        (
            "matched",
            sizes[(sizes["series"] == "ref") & sizes["matched"]],
            REF_COLOR,
            "-",
        ),
        (
            f"{labels[1]} only",
            sizes[(sizes["series"] == "test") & ~sizes["matched"]],
            TEST_COLOR,
            "-",
        ),
        (
            f"{labels[0]} only",
            sizes[(sizes["series"] == "ref") & ~sizes["matched"]],
            "black",
            "--",
        ),
    ]
    for name, g, color, ls in groups:
        x = np.sort(g["area"].to_numpy())
        ax.step(
            x,
            np.arange(1, len(x) + 1) / len(x),
            color=color,
            ls=ls,
            label=f"{name}, n={len(x)}",
        )
    ax.set_xscale("log")
    ax.set_xlabel(f"{primary.lower()} area (px, log)")
    ax.set_ylabel("cumulative fraction")
    ax.set_title("Sizes of matched and unmatched objects")
    ax.legend(loc="upper left")
    fig.tight_layout()
    return fig


def plot_agreement(
    per_feature: pd.DataFrame,
    family: pd.DataFrame,
    examples: pd.DataFrame,
    granularity_compartment: str = "Cells",
    highlight_channels: tuple[str, ...] = ("AGP", "Mito"),
    labels: tuple[str, str] = ("original", "MicroSplit"),
) -> plt.Figure:
    """Per-cell agreement: example scatters, family heatmap, granularity by scale.

    Parameters
    ----------
    per_feature : pandas.DataFrame
        `agreement_per_feature` table.
    family : pandas.DataFrame
        `agreement_family` table.
    examples : pandas.DataFrame
        `agreement_examples` table.
    granularity_compartment : str
        Compartment of panel c.
    highlight_channels : tuple of str
        Channels drawn in blue in panel c.
    labels : tuple of str
        Reference and test names.

    Returns
    -------
    matplotlib.figure.Figure
        Panels a (scatters), b (heatmap), c (granularity).
    """
    use_style()
    feats = list(
        dict.fromkeys(zip(examples["compartment"], examples["feature"], strict=True))
    )
    n_rows = int(np.ceil(len(feats) / 3))
    fig = plt.figure(figsize=(11, 3 * n_rows + 3.2))
    grid = fig.add_gridspec(n_rows + 1, 6, height_ratios=[1] * n_rows + [1.05])
    rho = per_feature.set_index(["compartment", "feature"])["rho"]
    for k, (comp, feat) in enumerate(feats):
        ax = fig.add_subplot(grid[k // 3, 2 * (k % 3) : 2 * (k % 3) + 2])
        if k == 0:
            _letter(ax, "a")
        d = examples[(examples["compartment"] == comp) & (examples["feature"] == feat)]
        lo = np.nanpercentile(np.r_[d["ref"], d["test"]], 0.5)
        hi = np.nanpercentile(np.r_[d["ref"], d["test"]], 99.5)
        ax.hexbin(
            d["ref"],
            d["test"],
            gridsize=35,
            cmap="Greys",
            mincnt=1,
            bins="log",
            extent=(lo, hi, lo, hi),
        )
        ax.plot([lo, hi], [lo, hi], color=TEST_COLOR, ls="--", lw=1)
        ax.set_title(f"{comp}: {feat}\nSpearman ρ = {rho[(comp, feat)]:.2f}")  # noqa: RUF001
        ax.set_xlabel(labels[0])
        ax.set_ylabel(labels[1])

    ax = fig.add_subplot(grid[n_rows, :4])
    _letter(ax, "b")
    image = _heatmap(ax, family.T, "Blues", mpl.colors.Normalize(0, 1))
    ax.set_xticks(range(family.shape[0]), family.index, rotation=20, ha="right")
    ax.set_title("Median per-cell agreement (Spearman ρ) by feature family")  # noqa: RUF001
    fig.colorbar(image, ax=ax, fraction=0.03)

    ax = fig.add_subplot(grid[n_rows, 4:])
    _letter(ax, "c")
    gran = per_feature[
        (per_feature["family"] == "Granularity")
        & (per_feature["compartment"] == granularity_compartment)
    ]
    for ch, g in gran.groupby("channel"):
        g = g.sort_values("scale")
        color = TEST_COLOR if ch in highlight_channels else REF_COLOR
        ax.plot(g["scale"], g["rho"], color=color, lw=1.5 if color == TEST_COLOR else 1)
        ax.text(
            g["scale"].iloc[-1] + 0.3, g["rho"].iloc[-1], ch, color=color, va="center"
        )
    ax.set_ylim(0, 1)
    ax.set_xlabel("granularity scale (1 = finest)")
    ax.set_ylabel("Spearman ρ")  # noqa: RUF001
    ax.set_title(f"Granularity agreement by scale ({granularity_compartment})")
    fig.tight_layout()
    return fig


def plot_distortion(
    per_feature: pd.DataFrame,
    well_channel: pd.DataFrame,
    multivariate: pd.DataFrame,
    labels: tuple[str, str] = ("original", "MicroSplit"),
    seed: int = 0,
) -> plt.Figure:
    """Distances against the noise floor and the well-to-well yardstick.

    Parameters
    ----------
    per_feature : pandas.DataFrame
        `distortion_per_feature` table.
    well_channel : pandas.DataFrame
        `distortion_well_channel` table.
    multivariate : pandas.DataFrame
        `multivariate_summary` table.
    labels : tuple of str
        Reference and test names.
    seed : int
        Seed of the horizontal jitter.

    Returns
    -------
    matplotlib.figure.Figure
        Panels a (per feature / noise), b (fraction of a well difference per
        channel), c (multivariate excess over noise).
    """
    use_style()
    rng = np.random.default_rng(seed)
    fig = plt.figure(figsize=(11, 7))
    grid = fig.add_gridspec(2, 2, width_ratios=[1.5, 1])
    ax = fig.add_subplot(grid[0, :])
    _letter(ax, "a")
    data = per_feature.assign(
        family=per_feature["family"].replace({ZERNIKE_FAMILY: "Shape"})
    )
    comps = list(dict.fromkeys(data["compartment"]))
    families = [f for f in STRIP_FAMILIES if f in set(data["family"])]
    width = 0.8 / len(comps)
    for i, fam in enumerate(families):
        for j, comp in enumerate(comps):
            v = data.loc[
                (data["family"] == fam) & (data["compartment"] == comp),
                "ratio_to_noise",
            ].dropna()
            if v.empty:
                continue
            x = i + (j - (len(comps) - 1) / 2) * width
            ax.scatter(
                x + rng.uniform(-width / 4, width / 4, len(v)),
                v,
                s=3,
                color=LIGHT_TEST_COLOR,
                alpha=0.6,
                lw=0,
            )
            ax.plot(
                [x - width / 3, x + width / 3], [v.median()] * 2, color="#333333", lw=2
            )
            if i == 0:
                ax.text(x, 0.4, comp[:4], ha="center", fontsize=6)
    ax.axhline(1, color="0.5", ls="--", lw=0.8)
    ax.set_yscale("log", base=2)
    ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda v, _: f"{v:g}×"))  # noqa: RUF001
    ax.set_ylim(0.35, None)
    ax.set_xticks(range(len(families)), families)
    ax.set_ylabel("distance ÷ noise floor (log)")
    ax.set_title(
        "Per feature distance relative to the noise floor "
        "(dots = features, bars = medians)"
    )

    ax = fig.add_subplot(grid[1, 0])
    _letter(ax, "b")
    table = well_channel.T
    vmax = max(2.0, float(np.nanmax(table.to_numpy())))
    image = _heatmap(ax, table, "RdBu_r", TwoSlopeNorm(1.0, 0.0, vmax))
    fig.colorbar(image, ax=ax, fraction=0.04)
    ax.set_title(
        "Distortion as a fraction of a well-to-well difference\n"
        "(intensity, granularity, radial features; 1 = switching wells)"
    )

    ax = fig.add_subplot(grid[1, 1])
    _letter(ax, "c")
    x = np.arange(len(multivariate))
    ax.bar(
        x - 0.2,
        multivariate["excess"],
        0.4,
        color=TEST_COLOR,
        label=f"{labels[0]} vs {labels[1]}, same image",
    )
    ax.bar(
        x + 0.2,
        multivariate["well_excess"],
        0.4,
        color=LIGHT_REF_COLOR,
        label=f"{labels[0]} vs {labels[0]}, different wells",
    )
    for xi, (e, f) in enumerate(
        zip(multivariate["excess"], multivariate["fraction_of_well"], strict=True)
    ):
        ax.text(
            xi - 0.2, e, f"{100 * f:.0f}%", ha="center", va="bottom", color=TEST_COLOR
        )
    ax.set_xticks(x, multivariate.index)
    ax.set_ylabel("W above noise floor (PCA space)")
    ax.set_title("All features at once")
    ax.legend(loc="upper left")
    fig.tight_layout()
    return fig


def _box_best(ax: plt.Axes, corr: pd.DataFrame) -> None:
    best = corr.to_numpy().argmax(axis=1)
    for i, j in enumerate(best):
        color = "black" if corr.columns[j] == corr.index[i] else "#D62728"
        ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, ec=color, lw=1.5))


def plot_profiles(
    corr: pd.DataFrame,
    corr_offset: pd.DataFrame,
    distances: pd.DataFrame,
    summary: pd.DataFrame,
    well_labels: dict | None = None,
    labels: tuple[str, str] = ("original", "MicroSplit"),
) -> plt.Figure:
    """Well profile retrieval and distance geometry.

    Parameters
    ----------
    corr : pandas.DataFrame
        `profile_correlation` table.
    corr_offset : pandas.DataFrame
        `profile_correlation_offset_removed` table.
    distances : pandas.DataFrame
        `profile_distances` table.
    summary : pandas.DataFrame
        `profile_summary` table.
    well_labels : dict or None
        Well id to display label.
    labels : tuple of str
        Reference and test names.

    Returns
    -------
    matplotlib.figure.Figure
        Panels a (as measured), b (offset removed), c (distance geometry).
    """
    use_style()
    well_labels = well_labels or {}
    s = summary.iloc[0]
    n = int(s["n_wells"])
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    norm = mpl.colors.Normalize(-0.8, 0.8)
    titles = [
        f"As measured: right well found {round(s['hit_rate'] * n)}/{n}",
        f"Global offset removed: right well found "
        f"{round(s['hit_rate_offset_removed'] * n)}/{n}",
    ]
    for ax, table, letter, title in zip(
        axes[:2], (corr, corr_offset), "ab", titles, strict=True
    ):
        _letter(ax, letter)
        image = ax.imshow(table.to_numpy(), cmap="RdBu_r", norm=norm)
        ticks = [well_labels.get(w, w) for w in table.columns]
        ax.set_xticks(range(n), ticks, rotation=90)
        ax.set_yticks(range(n), [well_labels.get(w, w) for w in table.index])
        _box_best(ax, table)
        ax.set_xlabel(f"{labels[0]} well profile")
        ax.set_ylabel(f"{labels[1]} well profile")
        ax.set_title(title + " (box = best match)")
    fig.colorbar(image, ax=axes[1], fraction=0.046, label="profile correlation")
    ax = axes[2]
    _letter(ax, "c")
    lo = distances[["ref_distance", "test_distance"]].min().min() * 0.9
    hi = distances[["ref_distance", "test_distance"]].max().max() * 1.05
    ax.plot([lo, hi], [lo, hi], color=LIGHT_REF_COLOR, lw=1)
    ax.scatter(
        distances["ref_distance"], distances["test_distance"], s=12, color=TEST_COLOR
    )
    ax.set_xlabel(f"distance between two wells, {labels[0]}")
    ax.set_ylabel(f"same pair, {labels[1]}")
    ax.set_title(
        f"Well-to-well geometry ({len(distances)} pairs, "
        f"Spearman ρ = {s['distance_spearman']:.2f})"  # noqa: RUF001
    )
    fig.tight_layout()
    return fig


def plot_spread(
    ratios: pd.DataFrame,
    values: pd.DataFrame,
    compartment: str = "Cells",
    n_hist: int = 2,
    labels: tuple[str, str] = ("original", "MicroSplit"),
) -> plt.Figure:
    """Loss of cell-to-cell spread and intensity shifts of matched cells.

    Parameters
    ----------
    ratios : pandas.DataFrame
        `spread_ratios` table.
    values : pandas.DataFrame
        `spread_values` table.
    compartment : str
        Compartment shown in panel a; must match `values`.
    n_hist : int
        Histograms for the channels with the lowest mean-intensity spread ratio.
    labels : tuple of str
        Reference and test names.

    Returns
    -------
    matplotlib.figure.Figure
        Panel a (SD ratios) and one histogram panel per selected channel.
    """
    use_style()
    r = ratios[ratios["compartment"] == compartment].reset_index(drop=True)
    worst = list(r.nsmallest(n_hist, "sd_ratio_MeanIntensity")["channel"])
    fig, axes = plt.subplots(
        1,
        1 + n_hist,
        figsize=(4 + 3.3 * n_hist, 3.3),
        gridspec_kw={"width_ratios": [1.3] + [1] * n_hist},
    )
    ax = axes[0]
    _letter(ax, "a")
    x = np.arange(len(r))
    dark = [TEST_COLOR if c in worst else DARK_REF_COLOR for c in r["channel"]]
    light = [LIGHT_TEST_COLOR if c in worst else LIGHT_REF_COLOR for c in r["channel"]]
    ax.bar(
        x - 0.18,
        r["sd_ratio_MeanIntensity"],
        0.36,
        color=dark,
        label="cell-to-cell spread of mean intensity",
    )
    ax.bar(
        x + 0.18,
        r["sd_ratio_StdIntensity"],
        0.36,
        color=light,
        label="cell-to-cell spread of within-cell Std",
    )
    ax.axhline(1, color="0.4", ls="--", lw=0.8)
    ax.set_xticks(x, r["channel"])
    ax.set_ylabel(f"SD {labels[1]} ÷ SD {labels[0]}")
    ax.set_title(f"Cell-to-cell variation kept ({compartment}, matched)")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.42))
    shift = r.set_index("channel")["median_shift"]
    for k, (ax, ch) in enumerate(zip(axes[1:], worst, strict=True)):
        _letter(ax, "bcdefg"[k])
        v = values[values["channel"] == ch]
        ref, test = (
            v.loc[v["series"] == "ref", "value"],
            v.loc[v["series"] == "test", "value"],
        )
        bins = np.linspace(*np.nanpercentile(np.r_[ref, test], [0.5, 99.5]), 50)
        ax.hist(ref, bins, color=REF_COLOR, label=labels[0])
        ax.hist(test, bins, histtype="step", color=TEST_COLOR, lw=1.5, label=labels[1])
        ax.set_xlabel(f"mean {ch} intensity per cell")
        ax.set_yticks([])
        ax.spines["left"].set_visible(False)
        ax.set_title(f"{ch}: {100 * shift[ch]:+.0f}% median")
        if k == 0:
            ax.legend()
    fig.tight_layout()
    return fig
