from __future__ import annotations

import torch


def summarize_planner_scores(scores: dict[str, torch.Tensor], chosen_idx: torch.Tensor) -> dict[str, float]:
    batch_idx = torch.arange(chosen_idx.shape[0], device=chosen_idx.device)
    return {
        name: values[batch_idx, chosen_idx].float().mean().item()
        for name, values in scores.items()
        if name != "total_cost"
    }
