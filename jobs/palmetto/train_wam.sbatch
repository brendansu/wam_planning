#!/bin/bash
#SBATCH --job-name=wam-train
#SBATCH --output=runs/slurm-%j.out
#SBATCH --error=runs/slurm-%j.err
#SBATCH --time=04:00:00
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4

set -euo pipefail

module purge
module load python

python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"

python scripts/train.py --config configs/train_wam.yaml
