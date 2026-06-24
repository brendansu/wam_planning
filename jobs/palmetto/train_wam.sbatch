#!/bin/bash
#SBATCH --job-name=wam-smoke
#SBATCH --output=runs/slurm-%j.out
#SBATCH --error=runs/slurm-%j.err
#SBATCH --time=00:20:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=2

set -euo pipefail

module purge
module load python

python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"

python scripts/train.py --config configs/local_smoke.yaml
python scripts/visualize_rollout.py --config configs/local_smoke.yaml
