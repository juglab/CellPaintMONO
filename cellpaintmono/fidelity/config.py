"""Configuration of a feature fidelity comparison."""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Literal

import yaml

from .summary import DEFAULT_GROUPS


@dataclass
class FidelityConfig:
    """Settings for comparing a reference and a test CellProfiler run.

    Parameters
    ----------
    reference_dir : pathlib.Path
        Folder with the reference run (`Image.csv` and one CSV per compartment).
    test_dir : pathlib.Path
        Folder with the test run, same layout as the reference.
    output_dir : pathlib.Path
        Folder for the `tables/` and `figures/` outputs.
    reference_label : str
        Name of the reference series in tables and figures.
    test_label : str
        Name of the test series in tables and figures.
    compartments : list of str
        Object tables to compare; each is read from `<name>.csv`.
    primary_compartment : str
        Compartment matched by centroid; the others are paired through
        `Parent_<primary_compartment>`.
    image_keys : list of str
        `Image.csv` columns that identify one field of view in both runs.
    well_keys : list of str
        `Image.csv` columns that identify one well.
    channels : list of str or None
        Channel names as they appear in feature names. Detected from
        `Intensity_MeanIntensity_*` columns when None.
    exclude_images : list of str
        Image names whose features are dropped (pipeline helper images).
    max_match_distance : dict of str to float
        Largest centroid distance, in pixels, for a matched object pair.
    clip : float
        Robust z-scores are clipped to [-clip, clip].
    max_missing : float
        Features with a larger fraction of missing reference values are dropped
        from the distance analyses.
    n_shuffles : int
        Label shuffles per feature for the one-dimensional noise floors.
    n_shuffles_multivariate : int
        Label shuffles for the multivariate noise floor and p-value.
    n_components : int
        Principal components used for the multivariate distance.
    max_cells_multivariate : int or None
        Cells per image and series subsampled for exact optimal transport.
    well_pairs : {"all", "same_plate"}
        Which pairs of wells make up the well-to-well yardstick.
    max_well_pairs : int or None
        Random subset of well pairs used for the yardstick; all pairs if None.
    seed : int
        Seed of every random step.
    example_features : list of str
        `Compartment:Feature` entries shown as per-cell scatter plots.
    worked_example_image : str or None
        Image id used for the worked example; the first image if None.
    worked_example_features : list of str
        `Compartment:Feature` entries used in the worked example.
    spread_compartment : str
        Compartment used for the spread and intensity shift analysis.
    feature_groups : list of dict
        Feature groups of the summary table: `name`, `families`, and optionally
        `channels` and `min_scale` (see `summary.group_mask`).
    """

    reference_dir: Path
    test_dir: Path
    output_dir: Path
    reference_label: str = "original"
    test_label: str = "MicroSplit"
    compartments: list[str] = field(
        default_factory=lambda: ["Nuclei", "Cells", "Cytoplasm"]
    )
    primary_compartment: str = "Nuclei"
    image_keys: list[str] = field(
        default_factory=lambda: ["Metadata_Plate", "Metadata_Well", "Metadata_Site"]
    )
    well_keys: list[str] = field(
        default_factory=lambda: ["Metadata_Plate", "Metadata_Well"]
    )
    channels: list[str] | None = None
    exclude_images: list[str] = field(default_factory=lambda: ["RescaleIntensity"])
    max_match_distance: dict[str, float] = field(
        default_factory=lambda: {"Nuclei": 8.0, "Cells": 15.0, "Cytoplasm": 15.0}
    )
    clip: float = 10.0
    max_missing: float = 0.05
    n_shuffles: int = 10
    n_shuffles_multivariate: int = 100
    n_components: int = 30
    max_cells_multivariate: int | None = 2000
    well_pairs: Literal["all", "same_plate"] = "all"
    max_well_pairs: int | None = None
    seed: int = 0
    example_features: list[str] = field(
        default_factory=lambda: [
            "Nuclei:AreaShape_Area",
            "Cells:Intensity_MeanIntensity_DNA",
            "Cells:Intensity_MeanIntensity_RNA",
            "Cells:Intensity_MeanIntensity_Mito",
            "Cells:Granularity_3_Mito",
            "Cells:Correlation_K_AGP_RNA",
        ]
    )
    worked_example_image: str | None = None
    worked_example_features: list[str] = field(
        default_factory=lambda: [
            "Cells:Intensity_MeanIntensity_DNA",
            "Cells:Intensity_StdIntensity_Mito",
        ]
    )
    spread_compartment: str = "Cells"
    feature_groups: list[dict[str, Any]] = field(
        default_factory=lambda: [dict(g) for g in DEFAULT_GROUPS]
    )

    def __post_init__(self) -> None:
        """Convert path fields to pathlib.Path."""
        self.reference_dir = Path(self.reference_dir)
        self.test_dir = Path(self.test_dir)
        self.output_dir = Path(self.output_dir)

    @property
    def table_dir(self) -> Path:
        """Folder for CSV tables."""
        return self.output_dir / "tables"

    @property
    def figure_dir(self) -> Path:
        """Folder for figures."""
        return self.output_dir / "figures"

    @classmethod
    def from_yaml(cls, path: str | Path, **overrides: Any) -> FidelityConfig:
        """Read a configuration from YAML.

        Parameters
        ----------
        path : str or pathlib.Path
            YAML file whose keys are the field names of this class.
        **overrides : Any
            Values that replace those in the file (e.g. from the command line).

        Returns
        -------
        FidelityConfig
            The configuration.

        Raises
        ------
        ValueError
            If the file contains unknown keys.
        """
        with open(path) as f:
            values = yaml.safe_load(f) or {}
        values.update({k: v for k, v in overrides.items() if v is not None})
        unknown = set(values) - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError(f"Unknown config keys: {sorted(unknown)}")
        return cls(**values)
