"""
Interface for running MicroSplit operations without direct dependency.
Generates configs and calls standalone scripts in separate environment.
"""

import pandas as pd
from pathlib import Path
from typing import List, Optional
import subprocess
import yaml


def create_combine_config(
    sample_id: str,
    input_dir: Path,
    output_dir: Path,
    channels: List[str] = ["DNA", "RNA", "ER", "AGP", "Mito"],
    normalize: bool = True,
) -> Path:
    """
    Generate configuration for channel combination.

    Parameters
    ----------
    sample_id : str
        Unique identifier for the sample
    input_dir : Path
        Directory containing channel subdirectories with TIFF files
    output_dir : Path
        Directory for combined images and metadata
    channels : list
        Channel names to combine
    normalize : bool
        Whether to normalize channels before combining

    Returns
    -------
    Path
        Path to generated config file
    """
    config = {
        "sample_id": sample_id,
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "channels": channels,
        "normalize": normalize,
    }

    config_dir = Path("configs")
    config_dir.mkdir(exist_ok=True)
    config_path = config_dir / f"{sample_id}_combine.yaml"

    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)

    return config_path


def create_predict_config(
    sample_id: str,
    input_dir: Path,
    output_dir: Path,
    model_checkpoint: Path,
    target_channels: List[str] = ["DNA", "RNA", "ER", "AGP", "Mito"],
    num_samples: int = 10,
) -> Path:
    """
    Generate configuration for MicroSplit prediction.

    Parameters
    ----------
    sample_id : str
        Unique identifier for the sample
    input_dir : Path
        Directory containing combined images
    output_dir : Path
        Directory for predicted channels
    model_checkpoint : Path
        Path to trained MicroSplit model
    target_channels : list
        Names of channels to predict
    num_samples : int
        Number of MMSE samples for prediction

    Returns
    -------
    Path
        Path to generated config file
    """
    config = {
        "sample_id": sample_id,
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "model_checkpoint": str(model_checkpoint),
        "target_channels": target_channels,
        "num_samples": num_samples,
    }

    config_dir = Path("configs")
    config_dir.mkdir(exist_ok=True)
    config_path = config_dir / f"{sample_id}_predict.yaml"

    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)

    return config_path


def run_combine(
    config_path: Path, env_name: str = "microsplit", verbose: bool = True
) -> bool:
    """
    Execute channel combination in MicroSplit environment.

    Parameters
    ----------
    config_path : Path
        Path to combination config file
    env_name : str
        Name of conda environment with MicroSplit installed
    verbose : bool
        Print stdout/stderr

    Returns
    -------
    bool
        True if successful, False otherwise
    """
    cmd = [
        "conda",
        "run",
        "-n",
        env_name,
        "--no-capture-output",
        "python",
        "scripts/run_microsplit_combine.py",
        "--config",
        str(config_path),
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)

    if verbose:
        if result.stdout:
            print(result.stdout)
        if result.stderr:
            print(f"STDERR: {result.stderr}")

    return result.returncode == 0


def run_predict(
    config_path: Path, env_name: str = "microsplit", verbose: bool = True
) -> bool:
    """
    Execute MicroSplit prediction in separate environment.

    Parameters
    ----------
    config_path : Path
        Path to prediction config file
    env_name : str
        Name of conda environment with MicroSplit installed
    verbose : bool
        Print stdout/stderr

    Returns
    -------
    bool
        True if successful, False otherwise
    """
    cmd = [
        "conda",
        "run",
        "-n",
        env_name,
        "--no-capture-output",
        "python",
        "scripts/run_microsplit_predict.py",
        "--config",
        str(config_path),
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)

    if verbose:
        if result.stdout:
            print(result.stdout)
        if result.stderr:
            print(f"STDERR: {result.stderr}")

    return result.returncode == 0


def batch_process_samples(
    samples_df: pd.DataFrame,
    input_dir: Path,
    output_dir: Path,
    model_checkpoint: Optional[Path] = None,
    channels: List[str] = ["DNA", "RNA", "ER", "AGP", "Mito"],
    combine_only: bool = False,
    env_name: str = "microsplit",
) -> pd.DataFrame:
    """
    Process multiple samples through MicroSplit pipeline.

    Parameters
    ----------
    samples_df : pd.DataFrame
        DataFrame with 'Sample_ID' column
    input_dir : Path
        Directory with original channel images
    output_dir : Path
        Directory for outputs
    model_checkpoint : Path, optional
        Path to trained model (required if combine_only=False)
    channels : list
        Channel names
    combine_only : bool
        If True, only combine channels without prediction
    env_name : str
        Name of conda environment

    Returns
    -------
    pd.DataFrame
        Results with status for each sample
    """
    results = []

    for _, row in samples_df.iterrows():
        sample_id = row["Sample_ID"]

        # Combine channels
        combine_config = create_combine_config(
            sample_id=sample_id,
            input_dir=input_dir,
            output_dir=output_dir / "combined",
            channels=channels,
        )

        combine_success = run_combine(combine_config, env_name=env_name)

        predict_success = None
        if not combine_only and model_checkpoint is not None:
            # Predict channels
            predict_config = create_predict_config(
                sample_id=sample_id,
                input_dir=output_dir / "combined",
                output_dir=output_dir / "predicted",
                model_checkpoint=model_checkpoint,
                target_channels=channels,
            )

            predict_success = run_predict(predict_config, env_name=env_name)

        results.append(
            {
                "Sample_ID": sample_id,
                "combine_success": combine_success,
                "predict_success": predict_success,
            }
        )

    return pd.DataFrame(results)
