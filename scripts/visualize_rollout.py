from __future__ import annotations

import argparse
from pathlib import Path

from wam_planning.datasets.synthetic import SyntheticDrivingDataset, SyntheticSceneConfig
from wam_planning.planners import CandidateTrajectorySampler
from wam_planning.utils.config import load_config
from wam_planning.utils.seed import set_seed
from wam_planning.viz.scene import plot_scene


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["seed"])
    data_cfg = cfg["data"]
    dataset = SyntheticDrivingDataset(
        SyntheticSceneConfig(
            num_scenes=1,
            history_steps=data_cfg["history_steps"],
            future_steps=data_cfg["future_steps"],
            num_agents=data_cfg["num_agents"],
            dt=data_cfg["dt"],
            seed=cfg["seed"],
        )
    )
    scene = dataset[0]
    sampler = CandidateTrajectorySampler(data_cfg["future_steps"], data_cfg["num_candidates"], data_cfg["dt"])
    candidates = sampler.sample(scene["ego_history"].unsqueeze(0))[0]
    output = args.output or str(Path(cfg["run_dir"]) / "plots" / "synthetic_scene.png")
    plot_scene(
        scene["ego_history"],
        scene["agent_history"],
        scene["lane_centerlines"],
        candidates,
        output,
    )
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
