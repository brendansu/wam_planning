from __future__ import annotations

import torch
from torch import nn


class WorldActionModel(nn.Module):
    def __init__(self, history_steps: int, future_steps: int, num_agents: int, hidden_dim: int):
        super().__init__()
        scene_dim = history_steps * 4 + num_agents * history_steps * 4
        action_dim = future_steps * 4
        self.net = nn.Sequential(
            nn.Linear(scene_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2),
        )

    def forward(
        self,
        ego_history: torch.Tensor,
        agent_history: torch.Tensor,
        candidates: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        batch, num_candidates = candidates.shape[:2]
        scene = torch.cat(
            [
                ego_history.flatten(start_dim=1),
                agent_history.flatten(start_dim=1),
            ],
            dim=-1,
        )
        scene = scene[:, None, :].expand(-1, num_candidates, -1)
        features = torch.cat([scene, candidates.flatten(start_dim=2)], dim=-1)
        out = self.net(features)
        return {
            "collision_risk": torch.sigmoid(out[..., 0]),
            "cost": out[..., 1],
        }
