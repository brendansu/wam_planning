#!/bin/bash
#SBATCH --job-name=wam-eval
#SBATCH --output=runs/slurm-%j.out
#SBATCH --error=runs/slurm-%j.err
#SBATCH --time=00:05:00
#SBATCH --mem=1G
#SBATCH --cpus-per-task=1

set -euo pipefail

echo "Planner evaluation is reserved for Milestone 2+."
