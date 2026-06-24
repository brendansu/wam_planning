#!/bin/bash
#SBATCH --job-name=wam-eval
#SBATCH --output=runs/slurm-%j.out
#SBATCH --error=runs/slurm-%j.err
#SBATCH --time=01:00:00
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2

set -euo pipefail

module purge
module load python

source .venv/bin/activate
python scripts/evaluate.py --config configs/eval_planner.yaml
