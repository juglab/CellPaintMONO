# Feature fidelity: test vs reference CellProfiler features

Compares two CellProfiler runs of the same pipeline on the same fields of view, for
example original images (reference) and MicroSplit-predicted images (test). Every
distance is read against two yardsticks measured on the same data: a shuffle noise
floor and the difference between two wells.

Implementation: `cellpaintmono/fidelity/`.

## Input

Two folders exported with ExportToSpreadsheet, each with `Image.csv` and one CSV per
compartment (default `Nuclei`, `Cells`, `Cytoplasm`). Images are paired on
`Metadata_Plate`, `Metadata_Well`, `Metadata_Site`; only images present in both runs
are compared.

## Run

Write a config (copy `configs/fidelity/example.yaml`; all options are
documented in `cellpaintmono/fidelity/config.py`), then:

```
python scripts/fidelity/run_all.py --config configs/fidelity/<exp>.yaml
sbatch scripts/fidelity/run_fidelity.sh configs/fidelity/<exp>.yaml
```

Steps can also run one at a time; `06_summary.py` needs the tables of steps 2 and 3.
`--reference_dir`, `--test_dir`, `--output_dir` and `--test_label` override the
config, e.g. to compare several test series against one reference.

| Step | Question | Tables | Figure |
|---|---|---|---|
| `01_segmentation.py` | Are the same objects found? | `segmentation_per_image`, `object_pairs`, `primary_object_sizes` | `segmentation` |
| `02_agreement.py` | Does the same cell get the same values? | `agreement_per_feature`, `agreement_family`, `agreement_examples` | `agreement` |
| `03_distances.py` | Per image, how far do distributions move? | `distortion_*`, `multivariate_*`, `worked_example*` | `distortion`, `worked_example` |
| `04_profiles.py` | Do well profiles keep their identity? | `profile_*` | `profiles` |
| `05_spread.py` | Is cell-to-cell variation kept per channel? | `spread_ratios`, `spread_values` | `spread` |
| `06_summary.py` | Summary per feature group | `feature_group_summary` | |

Outputs go to `<output_dir>/tables/*.csv` and `<output_dir>/figures/*.{pdf,png}`.

## Methods

- **Features**: all numeric columns except bookkeeping (`Location`, `BoundingBox`,
  `Center`, `Orientation`, object numbers, parents, children) and helper images
  (`exclude_images`). Family from the column prefix; Zernike moments are a separate
  family. Channels from `Intensity_MeanIntensity_*` unless given.
- **Robust z-scores** (distances, profiles): reference median and 1.4826 MAD,
  clipped at ±10; features with zero reference MAD or > 5% missing values dropped.
- **Matching**: per image, Hungarian assignment of centroids on the full distance
  matrix, pairs beyond 8 px (nuclei) or 15 px (cells, cytoplasm) dropped. Per-cell
  agreement uses matched nuclei and the objects linked to them by `Parent_Nuclei`.
- **Agreement**: Spearman ρ per feature over matched objects (all features).
- **Distances**: per image and feature, Wasserstein-1 between reference and test
  cells (`d`) and its noise floor (mean of 10 label shuffles). Well yardstick: W1
  between reference cells of two wells minus its shuffle null, over all well pairs
  (or same-plate pairs, or a random subset). Per feature, `d` and noise are
  averaged over images; reported as `d / noise` and
  `(d - noise) / (well - well noise)`.
- **Multivariate**: PCA (30 components) fitted on reference cells; exact optimal
  transport (POT, Euclidean cost) per image with 100 shuffles for the noise floor and
  p-value; the same distance between wells as yardstick.
- **Profiles**: median robust z-score per feature and well over all compartments,
  standardised with the reference across-well mean and SD. Pearson correlation for
  identity retrieval, before and after removing the mean test minus reference
  offset; Spearman ρ of pairwise Euclidean well distances.
- **Spread**: SD over matched cells, test / reference, of the mean intensity and the
  within-cell Std per channel; median shift of the mean intensity.

## Scaling to large experiments

Per-image steps scale linearly with the number of images. The yardstick grows with
the number of well pairs: set `max_well_pairs` and consider `well_pairs: same_plate`.
`max_cells_multivariate` caps the cells per image for exact optimal transport.
