from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import torch


def plot_scene(
    ego_history: torch.Tensor,
    agent_history: torch.Tensor,
    lane_centerlines: torch.Tensor,
    candidates: torch.Tensor,
    output_path: str | Path,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    for lane in lane_centerlines:
        ax.plot(lane[:, 0], lane[:, 1], color="0.7", linewidth=1.5)
    ax.plot(ego_history[:, 0], ego_history[:, 1], color="tab:blue", linewidth=2, label="ego history")
    for agent in agent_history:
        ax.plot(agent[:, 0], agent[:, 1], color="tab:gray", alpha=0.8)
        ax.scatter(agent[-1, 0], agent[-1, 1], color="tab:gray", s=12)
    for candidate in candidates:
        ax.plot(candidate[:, 0], candidate[:, 1], color="tab:orange", alpha=0.25)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.2)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
