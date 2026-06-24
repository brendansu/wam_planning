from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from wam_planning.datasets.synthetic import SyntheticDrivingDataset, SyntheticSceneConfig
from wam_planning.models import WorldActionModel
from wam_planning.planners import CandidateTrajectorySampler
from wam_planning.utils.config import ensure_dir, load_config
from wam_planning.utils.device import resolve_device
from wam_planning.utils.seed import set_seed


def make_dataset(cfg: dict) -> SyntheticDrivingDataset:
    data_cfg = cfg["data"]
    return SyntheticDrivingDataset(
        SyntheticSceneConfig(
            num_scenes=data_cfg["num_scenes"],
            history_steps=data_cfg["history_steps"],
            future_steps=data_cfg["future_steps"],
            num_agents=data_cfg["num_agents"],
            dt=data_cfg["dt"],
            seed=cfg["seed"],
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Milestone 0 smoke check: load synthetic data, sample candidates, run one model forward pass."
    )
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["seed"])
    device = resolve_device(cfg["smoke"]["device"])
    run_dir = ensure_dir(cfg["run_dir"])

    loader = DataLoader(make_dataset(cfg), batch_size=cfg["smoke"]["batch_size"], shuffle=False)
    batch = next(iter(loader))

    sampler = CandidateTrajectorySampler(
        cfg["data"]["future_steps"],
        cfg["data"]["num_candidates"],
        cfg["data"]["dt"],
    )
    model = WorldActionModel(
        cfg["data"]["history_steps"],
        cfg["data"]["future_steps"],
        cfg["data"]["num_agents"],
        cfg["model"]["hidden_dim"],
    ).to(device)

    ego = batch["ego_history"].to(device)
    agents = batch["agent_history"].to(device)
    candidates = sampler.sample(ego)
    with torch.no_grad():
        outputs = model(ego, agents, candidates)

    summary = {
        "status": "ok",
        "milestone": "0/1 smoke",
        "device": str(device),
        "batch_size": ego.shape[0],
        "ego_history_shape": list(ego.shape),
        "agent_history_shape": list(agents.shape),
        "lane_centerlines_shape": list(batch["lane_centerlines"].shape),
        "candidate_shape": list(candidates.shape),
        "model_outputs": {name: list(value.shape) for name, value in outputs.items()},
    }
    out_path = Path(run_dir) / "smoke_summary.json"
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
