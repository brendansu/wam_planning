from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from wam_planning.datasets.synthetic import SyntheticDrivingDataset, SyntheticSceneConfig
from wam_planning.evaluators.metrics import summarize_planner_scores
from wam_planning.models import WorldActionModel
from wam_planning.planners import CandidateTrajectorySampler, RuleBasedPlanner
from wam_planning.utils.config import ensure_dir, load_config
from wam_planning.utils.device import resolve_device
from wam_planning.utils.seed import set_seed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["seed"])
    device = resolve_device(cfg["train"]["device"])
    run_dir = ensure_dir(cfg["run_dir"])
    data_cfg = cfg["data"]
    dataset = SyntheticDrivingDataset(
        SyntheticSceneConfig(
            num_scenes=data_cfg["num_val_scenes"],
            history_steps=data_cfg["history_steps"],
            future_steps=data_cfg["future_steps"],
            num_agents=data_cfg["num_agents"],
            dt=data_cfg["dt"],
            seed=cfg["seed"] + 20_000,
        )
    )
    loader = DataLoader(dataset, batch_size=cfg["train"]["batch_size"])
    sampler = CandidateTrajectorySampler(data_cfg["future_steps"], data_cfg["num_candidates"], data_cfg["dt"])
    rule_planner = RuleBasedPlanner(**cfg["planner"])
    model = WorldActionModel(
        data_cfg["history_steps"], data_cfg["future_steps"], data_cfg["num_agents"], cfg["model"]["hidden_dim"]
    ).to(device)

    ckpt_path = cfg["model"].get("checkpoint")
    if ckpt_path and Path(ckpt_path).exists():
        checkpoint = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(checkpoint["model"])

    totals = []
    with torch.no_grad():
        for batch in loader:
            ego = batch["ego_history"].to(device)
            agents = batch["agent_history"].to(device)
            candidates = sampler.sample(ego)
            rule_choice = rule_planner.choose(candidates, agents)
            totals.append(summarize_planner_scores(rule_choice["scores"], rule_choice["index"]))

    summary = {
        key: sum(row[key] for row in totals) / len(totals)
        for key in totals[0]
    }
    out_path = Path(run_dir) / "eval_summary.json"
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
