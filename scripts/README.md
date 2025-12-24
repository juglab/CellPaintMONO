# MicroSplit Scripts

Standalone scripts executed in the `microsplit` conda environment due to numpy version incompatibility with the main CellPaintMono environment.

## Environment Setup

```bash
# Create microsplit environment
conda env create -f microsplit_env.yml

# Activate when running scripts
conda activate microsplit
```

## Scripts

### `run_microsplit_combine.py`
Combines multiple Cell Painting channels into a single image for MicroSplit input.

**Input**: Individual channel TIFF files
**Output**: Combined TIFF file + metadata CSV

### `run_microsplit_predict.py`
Runs MicroSplit model inference to predict individual channels from combined image.

**Input**: Combined TIFF file + trained model checkpoint
**Output**: Predicted channel TIFF files + metadata CSV

## Usage

Scripts are typically called via `cellpaintmono.microsplit.workflow` functions, which handle config generation and subprocess execution.

Manual execution:
```bash
conda activate microsplit
python scripts/run_microsplit_combine.py --config configs/sample_combine.yaml
python scripts/run_microsplit_predict.py --config configs/sample_predict.yaml
```

## Dependencies

- numpy<2.0.0 (required by careamics)
- microsplit-reproducibility (from git)
- torch, torchvision
- tifffile, pandas, pyyaml
