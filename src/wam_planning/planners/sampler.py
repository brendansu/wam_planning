from __future__ import annotations

import torch


class CandidateTrajectorySampler:
    def __init__(self, future_steps: int, num_candidates: int, dt: float):
        self.future_steps = future_steps
        self.num_candidates = num_candidates
        self.dt = dt

    def sample(self, ego_history: torch.Tensor) -> torch.Tensor:
        device = ego_history.device
        batch = ego_history.shape[0]
        last = ego_history[:, -1]
        speeds = torch.linspace(4.0, 14.0, steps=8, device=device)
        lateral_targets = torch.tensor([-3.5, 0.0, 3.5], device=device)
        accelerations = torch.tensor([-1.0, 0.0, 1.0], device=device)

        primitives = torch.cartesian_prod(speeds, lateral_targets, accelerations)
        if primitives.shape[0] < self.num_candidates:
            repeats = (self.num_candidates + primitives.shape[0] - 1) // primitives.shape[0]
            primitives = primitives.repeat(repeats, 1)
        primitives = primitives[: self.num_candidates]

        t = torch.arange(1, self.future_steps + 1, device=device, dtype=torch.float32) * self.dt
        candidates = []
        for speed, y_target, accel in primitives:
            x = last[:, 0:1] + speed * t + 0.5 * accel * t.square()
            blend = torch.linspace(0.15, 1.0, self.future_steps, device=device)
            y = last[:, 1:2] + (y_target - last[:, 1:2]) * blend
            yaw = torch.atan2(torch.gradient(y, spacing=(t,), dim=1)[0], torch.gradient(x, spacing=(t,), dim=1)[0])
            v = torch.clamp(speed + accel * t, min=0.0).expand(batch, -1)
            candidates.append(torch.stack([x, y, yaw, v], dim=-1))
        return torch.stack(candidates, dim=1)
