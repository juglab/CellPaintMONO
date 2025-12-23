#!/usr/bin/env python
"""
Combine channels for MicroSplit input.
MUST be run in microsplit conda environment.
"""

import argparse
import yaml
from pathlib import Path
import tifffile
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple


def combine_channels(
    channel_images: Dict[str, np.ndarray],
    channels_to_combine: List[str],
    normalize: bool = True,
) -> Tuple[np.ndarray, Dict]:
    """Combine multiple channels into single image."""
    processed = {}

    for channel in channels_to_combine:
        img = channel_images[channel].astype(np.float32)

        if normalize:
            img_min, img_max = img.min(), img.max()
            if img_max > img_min:
                img = (img - img_min) / (img_max - img_min)

        processed[channel] = img

    # Equal weighting
    weight = 1.0 / len(channels_to_combine)
    combined = sum(processed[ch] * weight for ch in channels_to_combine)

    stats = {
        "min": float(combined.min()),
        "max": float(combined.max()),
        "mean": float(combined.mean()),
        "std": float(combined.std()),
    }

    return combined, stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    sample_id = config["sample_id"]
    input_dir = Path(config["input_dir"])
    output_dir = Path(config["output_dir"])
    channels = config["channels"]

    # Load channel images
    channel_images = {}
    for channel in channels:
        img_path = input_dir / channel / f"{sample_id}.tif"
        if not img_path.exists():
            print(f"ERROR: Image not found: {img_path}")
            return 1
        channel_images[channel] = tifffile.imread(img_path)

    # Combine
    combined, stats = combine_channels(
        channel_images, channels, normalize=config.get("normalize", True)
    )

    # Save combined image
    output_path = output_dir / f"{sample_id}.tif"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert to uint16
    if combined.dtype != np.uint16:
        combined_scaled = (combined * 65535).astype(np.uint16)
    else:
        combined_scaled = combined

    tifffile.imwrite(output_path, combined_scaled)

    # Save metadata
    metadata_dir = output_dir.parent / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)

    metadata = {
        "sample_id": sample_id,
        "channels": ",".join(channels),
        **{f"combined_{k}": v for k, v in stats.items()},
    }

    metadata_df = pd.DataFrame([metadata])
    metadata_path = metadata_dir / f"{sample_id}_combine.csv"
    metadata_df.to_csv(metadata_path, index=False)
    return 0


if __name__ == "__main__":
    exit(main())
