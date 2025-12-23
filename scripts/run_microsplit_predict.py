#!/usr/bin/env python
"""
Run MicroSplit predictions on combined images.
Must be run in microsplit conda environment.
"""

import argparse
import yaml
from pathlib import Path
import tifffile
import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    sample_id = config["sample_id"]
    input_dir = Path(config["input_dir"])
    output_dir = Path(config["output_dir"])
    model_checkpoint = Path(config["model_checkpoint"])
    target_channels = config["target_channels"]
    num_samples = config.get("num_samples", 10)

    # Import MicroSplit modules (only available in microsplit env)
    try:
        from microsplit_reproducibility.notebook_utils.JUMP import (
            load_pretrained_model,
            full_frame_evaluation,
        )
    except ImportError as e:
        print("ERROR: MicroSplit modules not available. Run in microsplit environment.")
        print(f"Details: {e}")
        return 1

    # Load model
    print(f"Loading model from {model_checkpoint}")
    try:
        model = load_pretrained_model(
            checkpoint_path=str(model_checkpoint), channel_list=target_channels
        )
    except Exception as e:
        print(f"ERROR: Failed to load model: {e}")
        return 1

    # Load combined image
    combined_path = input_dir / f"{sample_id}.tif"
    if not combined_path.exists():
        print(f"ERROR: Combined image not found: {combined_path}")
        return 1

    combined_img = tifffile.imread(combined_path)
    print(f"Loaded combined image: {combined_img.shape}, dtype={combined_img.dtype}")

    # Predict
    print(f"Running prediction with {num_samples} MMSE samples")
    try:
        predictions = full_frame_evaluation(
            model=model, input_image=combined_img, num_samples=num_samples
        )
    except Exception as e:
        print(f"ERROR: Prediction failed: {e}")
        return 1

    print(f"Predictions shape: {predictions.shape}")

    # Save each predicted channel
    output_dir.mkdir(parents=True, exist_ok=True)

    for i, channel_name in enumerate(target_channels):
        channel_dir = output_dir / channel_name
        channel_dir.mkdir(parents=True, exist_ok=True)

        # Extract channel prediction
        channel_pred = predictions[..., i]

        # Convert to uint16
        if channel_pred.dtype != np.uint16:
            pred_min = channel_pred.min()
            pred_max = channel_pred.max()
            if pred_max > pred_min:
                scaled = (channel_pred - pred_min) / (pred_max - pred_min)
                channel_pred = (scaled * 65535).astype(np.uint16)
            else:
                channel_pred = channel_pred.astype(np.uint16)

        output_path = channel_dir / f"{sample_id}.tif"
        tifffile.imwrite(output_path, channel_pred)
        print(f"Saved {channel_name} to {output_path}")

    # Save prediction metadata
    metadata_dir = output_dir.parent / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)

    metadata = {
        "sample_id": sample_id,
        "model_checkpoint": str(model_checkpoint),
        "num_samples": num_samples,
        "channels": ",".join(target_channels),
        "prediction_shape": str(predictions.shape),
    }

    metadata_df = pd.DataFrame([metadata])
    metadata_path = metadata_dir / f"{sample_id}_predict.csv"
    metadata_df.to_csv(metadata_path, index=False)

    print(f"SUCCESS: Predictions saved to {output_dir}")
    return 0


if __name__ == "__main__":
    exit(main())
