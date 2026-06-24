from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class SyntheticSceneConfig:
    num_scenes: int
    history_steps: int
    future_steps: int
    num_agents: int
    dt: float
    seed: int


class SyntheticDrivingDataset(Dataset):
    """Small deterministic synthetic scenes for local smoke tests."""

    def __init__(self, cfg: SyntheticSceneConfig):
        self.cfg = cfg
        generator = torch.Generator().manual_seed(cfg.seed)
        self._scenes = [self._make_scene(generator) for _ in range(cfg.num_scenes)]

    def __len__(self) -> int:
        return len(self._scenes)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return self._scenes[idx]

    def _make_scene(self, generator: torch.Generator) -> dict[str, torch.Tensor]:
        h = self.cfg.history_steps
        a = self.cfg.num_agents
        dt = self.cfg.dt
        times = torch.arange(-(h - 1), 1, dtype=torch.float32) * dt

        ego_speed = torch.empty(1).uniform_(6.0, 12.0, generator=generator).item()
        ego_y = torch.empty(1).uniform_(-0.2, 0.2, generator=generator).item()
        ego_history = torch.stack(
            [
                ego_speed * times,
                torch.full_like(times, ego_y),
                torch.zeros_like(times),
                torch.full_like(times, ego_speed),
            ],
            dim=-1,
        )

        agent_pos = torch.empty(a, 2).uniform_(-20.0, 35.0, generator=generator)
        agent_pos[:, 1] = torch.empty(a).uniform_(-6.0, 6.0, generator=generator)
        agent_vel = torch.empty(a, 2).uniform_(-1.0, 1.0, generator=generator)
        agent_vel[:, 0] += torch.empty(a).uniform_(3.0, 11.0, generator=generator)
        agent_yaw = torch.atan2(agent_vel[:, 1], agent_vel[:, 0])
        agent_speed = torch.linalg.norm(agent_vel, dim=-1)

        agent_history = []
        for t in times:
            xy = agent_pos + agent_vel * t
            agent_history.append(
                torch.cat(
                    [xy, agent_yaw[:, None], agent_speed[:, None]],
                    dim=-1,
                )
            )
        agent_history = torch.stack(agent_history, dim=1)

        lane_centerlines = torch.tensor(
            [
                [[-30.0, -3.5], [0.0, -3.5], [40.0, -3.5]],
                [[-30.0, 0.0], [0.0, 0.0], [40.0, 0.0]],
                [[-30.0, 3.5], [0.0, 3.5], [40.0, 3.5]],
            ],
            dtype=torch.float32,
        )

        return {
            "ego_history": ego_history,
            "agent_history": agent_history,
            "lane_centerlines": lane_centerlines,
        }
