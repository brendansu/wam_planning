from wam_planning.datasets.synthetic import SyntheticDrivingDataset, SyntheticSceneConfig


def test_synthetic_scene_shapes() -> None:
    dataset = SyntheticDrivingDataset(
        SyntheticSceneConfig(
            num_scenes=2,
            history_steps=8,
            future_steps=12,
            num_agents=5,
            dt=0.2,
            seed=1,
        )
    )

    scene = dataset[0]

    assert len(dataset) == 2
    assert scene["ego_history"].shape == (8, 4)
    assert scene["agent_history"].shape == (5, 8, 4)
    assert scene["lane_centerlines"].shape == (3, 3, 2)
