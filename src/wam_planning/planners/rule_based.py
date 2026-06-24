from __future__ import annotations

import torch


class RuleBasedPlanner:
    def __init__(
        self,
        collision_weight: float,
        offroad_weight: float,
        comfort_weight: float,
        progress_weight: float,
    ):
        self.collision_weight = collision_weight
        self.offroad_weight = offroad_weight
        self.comfort_weight = comfort_weight
        self.progress_weight = progress_weight

    def score(self, candidates: torch.Tensor, agent_history: torch.Tensor) -> dict[str, torch.Tensor]:
        final_agents = agent_history[:, :, -1, :2]
        cand_xy = candidates[..., :2]
        dists = torch.cdist(cand_xy.reshape(candidates.shape[0], -1, 2), final_agents)
        dists = dists.reshape(candidates.shape[0], candidates.shape[1], candidates.shape[2], -1)

        min_dist = dists.amin(dim=(2, 3))
        collision_risk = torch.sigmoid(3.0 - min_dist)
        offroad_risk = (candidates[..., 1].abs().amax(dim=-1) > 5.25).float()
        jerk_cost = candidates[..., 3].diff(dim=-1).abs().mean(dim=-1)
        progress = candidates[..., 0].amax(dim=-1) - candidates[:, :, 0, 0]

        total_cost = (
            self.collision_weight * collision_risk
            + self.offroad_weight * offroad_risk
            + self.comfort_weight * jerk_cost
            - self.progress_weight * progress
        )
        return {
            "total_cost": total_cost,
            "collision_risk": collision_risk,
            "offroad_risk": offroad_risk,
            "jerk_cost": jerk_cost,
            "progress": progress,
        }

    def choose(self, candidates: torch.Tensor, agent_history: torch.Tensor) -> dict[str, torch.Tensor]:
        scores = self.score(candidates, agent_history)
        best_idx = scores["total_cost"].argmin(dim=1)
        batch_idx = torch.arange(candidates.shape[0], device=candidates.device)
        return {
            "trajectory": candidates[batch_idx, best_idx],
            "index": best_idx,
            "scores": scores,
        }
