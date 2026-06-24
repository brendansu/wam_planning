from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from wam_planning.datasets.synthetic import SyntheticDrivingDataset, SyntheticSceneConfig
from wam_planning.models import WorldActionModel
from wam_planning.planners import CandidateTrajectorySampler, RuleBasedPlanner
from wam_planning.utils.config import ensure_dir, load_config
from wam_planning.utils.device import resolve_device
from wam_planning.utils.seed import set_seed


def make_dataset(cfg: dict, split: str) -> SyntheticDrivingDataset:
    data_cfg = cfg["data"]
    return SyntheticDrivingDataset(
        SyntheticSceneConfig(
            num_scenes=data_cfg[f"num_{split}_scenes"],
            history_steps=data_cfg["history_steps"],
            future_steps=data_cfg["future_steps"],
            num_agents=data_cfg["num_agents"],
            dt=data_cfg["dt"],
            seed=cfg["seed"] + (0 if split == "train" else 10_000),
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["seed"])
    device = resolve_device(cfg["train"]["device"])
    run_dir = ensure_dir(cfg["run_dir"])
    ckpt_dir = ensure_dir(run_dir / "checkpoints")

    train_loader = DataLoader(
        make_dataset(cfg, "train"),
        batch_size=cfg["train"]["batch_size"],
        shuffle=True,
    )
    val_loader = DataLoader(make_dataset(cfg, "val"), batch_size=cfg["train"]["batch_size"])

    sampler = CandidateTrajectorySampler(
        cfg["data"]["future_steps"], cfg["data"]["num_candidates"], cfg["data"]["dt"]
    )
    teacher = RuleBasedPlanner(**cfg["planner"])
    model = WorldActionModel(
        cfg["data"]["history_steps"],
        cfg["data"]["future_steps"],
        cfg["data"]["num_agents"],
        cfg["model"]["hidden_dim"],
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["train"]["learning_rate"])

    metrics_path = Path(run_dir) / "metrics.jsonl"
    best_val = float("inf")
    with metrics_path.open("w", encoding="utf-8") as metrics_file:
        for epoch in range(cfg["train"]["epochs"]):
            model.train()
            train_loss = 0.0
            for batch in tqdm(train_loader, desc=f"epoch {epoch + 1} train"):
                ego = batch["ego_history"].to(device)
                agents = batch["agent_history"].to(device)
                candidates = sampler.sample(ego)
                with torch.no_grad():
                    targets = teacher.score(candidates, agents)
                    target_risk = targets["collision_risk"]
                    target_cost = targets["total_cost"]
                pred = model(ego, agents, candidates)
                loss = F.mse_loss(pred["collision_risk"], target_risk) + F.mse_loss(
                    pred["cost"], target_cost
                )
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            val_loss = evaluate_loss(model, val_loader, sampler, teacher, device)
            row = {
                "epoch": epoch + 1,
                "train_loss": train_loss / len(train_loader),
                "val_loss": val_loss,
            }
            metrics_file.write(json.dumps(row) + "\n")
            metrics_file.flush()

            if val_loss < best_val:
                best_val = val_loss
                torch.save(
                    {"model": model.state_dict(), "config": cfg, "val_loss": val_loss},
                    ckpt_dir / "best.pt",
                )

    print(f"finished training on {device}; best_val_loss={best_val:.4f}; run_dir={run_dir}")


@torch.no_grad()
def evaluate_loss(model, loader, sampler, teacher, device: torch.device) -> float:
    model.eval()
    total = 0.0
    for batch in loader:
        ego = batch["ego_history"].to(device)
        agents = batch["agent_history"].to(device)
        candidates = sampler.sample(ego)
        targets = teacher.score(candidates, agents)
        pred = model(ego, agents, candidates)
        loss = F.mse_loss(pred["collision_risk"], targets["collision_risk"]) + F.mse_loss(
            pred["cost"], targets["total_cost"]
        )
        total += loss.item()
    return total / len(loader)


if __name__ == "__main__":
    main()
