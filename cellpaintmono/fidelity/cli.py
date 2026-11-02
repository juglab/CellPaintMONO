"""Command-line entry for the numbered scripts in `scripts/fidelity/`."""

from __future__ import annotations

import argparse
import logging
from collections.abc import Callable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

from . import pipeline, plotting
from .config import FidelityConfig
from .io import well_labels
from .summary import group_summary

logger = logging.getLogger(__name__)
Tables = dict[str, pd.DataFrame]


def _labels(data: pipeline.FidelityData) -> tuple[str, str]:
    return data.config.reference_label, data.config.test_label


def _figure_segmentation(data: pipeline.FidelityData, t: Tables) -> dict:
    fig = plotting.plot_segmentation(
        t["segmentation_per_image"],
        t["object_pairs"],
        t["primary_object_sizes"],
        data.config.primary_compartment,
        _labels(data),
    )
    return {"segmentation": fig}


def _figure_agreement(data: pipeline.FidelityData, t: Tables) -> dict:
    fig = plotting.plot_agreement(
        t["agreement_per_feature"],
        t["agreement_family"],
        t["agreement_examples"],
        data.config.spread_compartment,
        labels=_labels(data),
    )
    return {"agreement": fig}


def _figure_distances(data: pipeline.FidelityData, t: Tables) -> dict:
    figs = {
        "distortion": plotting.plot_distortion(
            t["distortion_per_feature"],
            t["distortion_well_channel"],
            t["multivariate_summary"],
            _labels(data),
            data.config.seed,
        )
    }
    if not t["worked_example"].empty:
        figs["worked_example"] = plotting.plot_worked_example(
            t["worked_example"],
            t["worked_example_null"],
            pipeline.worked_example_values(data, t["worked_example"]),
            _labels(data),
        )
    return figs


def _figure_profiles(data: pipeline.FidelityData, t: Tables) -> dict:
    fig = plotting.plot_profiles(
        t["profile_correlation"],
        t["profile_correlation_offset_removed"],
        t["profile_distances"],
        t["profile_summary"],
        well_labels(data.reference.image),
        _labels(data),
    )
    return {"profiles": fig}


def _figure_spread(data: pipeline.FidelityData, t: Tables) -> dict:
    fig = plotting.plot_spread(
        t["spread_ratios"],
        t["spread_values"],
        data.config.spread_compartment,
        labels=_labels(data),
    )
    return {"spread": fig}


STEPS: dict[str, tuple[Callable, Callable]] = {
    "segmentation": (pipeline.run_segmentation, _figure_segmentation),
    "agreement": (pipeline.run_agreement, _figure_agreement),
    "distances": (pipeline.run_distances, _figure_distances),
    "profiles": (pipeline.run_profiles, _figure_profiles),
    "spread": (pipeline.run_spread, _figure_spread),
}


def run_summary(config: FidelityConfig) -> None:
    """Summarise agreement and distortion per feature group.

    Reads the tables written by the agreement and distances steps.

    Parameters
    ----------
    config : FidelityConfig
        The configuration.
    """
    agreement = pd.read_csv(config.table_dir / "agreement_per_feature.csv")
    distortion = pd.read_csv(config.table_dir / "distortion_per_feature.csv")
    summary = group_summary(agreement, distortion, config.feature_groups)
    pipeline.write_tables({"feature_group_summary": summary}, config.table_dir)


def run_steps(config: FidelityConfig, steps: list[str]) -> None:
    """Run analysis steps and write their tables and figures.

    Parameters
    ----------
    config : FidelityConfig
        The configuration.
    steps : list of str
        Step names: keys of `STEPS`, and `summary`, which needs the tables of
        `agreement` and `distances`.
    """
    analysis = [s for s in steps if s != "summary"]
    if analysis:
        logger.info("Loading %s and %s", config.reference_dir, config.test_dir)
        data = pipeline.prepare(config)
    for step in analysis:
        logger.info("Running %s", step)
        run, figures = STEPS[step]
        tables = run(data)
        pipeline.write_tables(tables, config.table_dir)
        for name, fig in figures(data, tables).items():
            plotting.save_figure(fig, config.figure_dir, name)
            plt.close(fig)
    if "summary" in steps:
        logger.info("Running summary")
        run_summary(config)
    logger.info("Outputs in %s", config.output_dir)


def main(steps: list[str] | None = None) -> None:
    """Parse arguments and run one step, or all steps.

    Parameters
    ----------
    steps : list of str or None
        Steps run by the calling script; all steps if None.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="YAML configuration file")
    parser.add_argument("--reference_dir", help="Override the reference run folder")
    parser.add_argument("--test_dir", help="Override the test run folder")
    parser.add_argument("--output_dir", help="Override the output folder")
    parser.add_argument("--test_label", help="Override the test series name")
    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(message)s")
    logger.setLevel(logging.INFO)
    config = FidelityConfig.from_yaml(
        args.config,
        reference_dir=args.reference_dir,
        test_dir=args.test_dir,
        output_dir=args.output_dir,
        test_label=args.test_label,
    )
    run_steps(config, steps or [*STEPS, "summary"])
