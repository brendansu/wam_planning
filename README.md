# World-Action Model Planning

Research scaffold for action-conditioned autonomous-driving planning. The core loop is:

```text
scene history + candidate ego trajectory -> predicted risk/cost -> planner choice
```

The current checked-in scope is Milestone 0 and Milestone 1 only. It uses synthetic scenes so local smoke tests do not depend on Waymo, nuPlan, CARLA, or cluster data.

## Local Setup

Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

Run the Milestone 0 smoke check:

```powershell
python scripts/train.py --config configs/local_smoke.yaml
```

This loads synthetic data, samples candidate ego trajectories, runs one placeholder model forward pass, and writes `runs/local_smoke/smoke_summary.json`. It does not train a planner yet.

Generate a synthetic scene plot:

```powershell
python scripts/visualize_rollout.py --config configs/local_smoke.yaml
```

Run tests:

```powershell
python -m pytest
```

## Structure

```text
configs/              YAML experiment configs
data/sample/          tiny checked-in samples only
jobs/palmetto/        sbatch templates
scripts/              train/evaluate/visualize entry points
src/wam_planning/     installable Python package
runs/                 generated run artifacts, gitignored
```

## Current Components

- `SyntheticDrivingDataset`: deterministic ego, agent, and lane tensors.
- `CandidateTrajectorySampler`: simple speed/lane/acceleration primitives.
- `WorldActionModel`: placeholder MLP used only for a forward-pass smoke check.
- `train.py`: Milestone 0 smoke check, not a full training loop.
- `visualize_rollout.py`: Milestone 1 visualization for one synthetic scene and sampled candidates.

## Milestone Boundary

Implemented now:

- Milestone 0: repo skeleton, installable package, config loading, local smoke check, Palmetto smoke template.
- Milestone 1: synthetic dataset loader, lane/agent/ego scene tensors, candidate trajectory sampling, scene visualization.

Deferred:

- Milestone 2: rule-based planner and constant-velocity baseline.
- Milestone 3: supervised world-action model training.
- Milestone 4+: learned planning evaluation, HPC sweeps, VLM conditioning.

## Palmetto Workflow

Sync the repo to Palmetto, then submit:

```bash
sbatch jobs/palmetto/train_wam.sbatch
```

Expected artifacts:

```text
runs/
  experiment_name/
    smoke_summary.json
    plots/
```

Only sync selected metrics, plots, videos, and checkpoints back locally.
