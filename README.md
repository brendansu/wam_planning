# World-Action Model Planning

Research scaffold for action-conditioned autonomous-driving planning. The core loop is:

```text
scene history + candidate ego trajectory -> predicted risk/cost -> planner choice
```

The first milestone uses synthetic scenes so local smoke tests do not depend on Waymo, nuPlan, CARLA, or cluster data.

## Local Setup

Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

Run the smoke training job:

```powershell
python scripts/train.py --config configs/local_smoke.yaml
```

Generate a synthetic scene plot:

```powershell
python scripts/visualize_rollout.py --config configs/local_smoke.yaml
```

Evaluate the rule-based planner:

```powershell
python scripts/evaluate.py --config configs/local_smoke.yaml
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
- `RuleBasedPlanner`: collision, offroad, comfort, and progress scoring.
- `WorldActionModel`: compact MLP over scene history and candidate action.
- `train.py`: supervised smoke training against rule-based synthetic labels.

## Palmetto Workflow

Sync the repo to Palmetto, then submit:

```bash
sbatch jobs/palmetto/train_wam.sbatch
sbatch jobs/palmetto/eval_planner.sbatch
```

Expected artifacts:

```text
runs/
  experiment_name/
    checkpoints/
    metrics.jsonl
    eval_summary.json
    plots/
```

Only sync selected metrics, plots, videos, and checkpoints back locally.
