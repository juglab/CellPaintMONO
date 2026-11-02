#!/bin/bash
#SBATCH --job-name=feature_fidelity
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=%x_%j.out
# Usage: sbatch scripts/fidelity/run_fidelity.sh configs/fidelity/<experiment>.yaml
set -euo pipefail
CONFIG=${1:?config file required}
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
uv run python scripts/fidelity/run_all.py --config "$CONFIG"
