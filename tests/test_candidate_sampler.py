from wam_planning.datasets.synthetic import SyntheticDrivingDataset, SyntheticSceneConfig
from wam_planning.planners import CandidateTrajectorySampler


def test_candidate_sampler_shape() -> None:
    dataset = SyntheticDrivingDataset(
        SyntheticSceneConfig(
            num_scenes=1,
            history_steps=8,
            future_steps=12,
            num_agents=5,
            dt=0.2,
            seed=1,
        )
    )
    scene = dataset[0]
    sampler = CandidateTrajectorySampler(future_steps=12, num_candidates=32, dt=0.2)

    candidates = sampler.sample(scene["ego_history"].unsqueeze(0))

    assert candidates.shape == (1, 32, 12, 4)
