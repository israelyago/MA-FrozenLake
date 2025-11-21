from pathlib import Path

import pandas as pd
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.env import ParallelPettingZooEnv
from ray.tune.registry import register_env
from tqdm import tqdm

import frozen_lake


class TrainConfig:
    full_observability = True
    with_communication = False
    slippery = False
    with_lstm = True
    reward_schedule = [10, 1, -1, -0.001, -0.1]
    artifacts: Path
    experiment_dir: Path
    run_metrics_file: Path
    run: int
    run_dir: Path


def train(config: TrainConfig) -> pd.DataFrame:
    def env_creator(c):
        success_rate = 1.0/3.0 if config.slippery else None
        env = frozen_lake.env(
            render_mode=None,
            seed=config.seed,
            flatten_observations=True,
            full_observability=config.full_observability,
            with_communication=config.with_communication,
            success_rate=success_rate,
        )
        env = ParallelPettingZooEnv(env)
        env.reset(seed=config.seed)
        return env

    register_env("ma_frozen_lake_v0", env_creator)

    iterations = 10000
    max_iterations = 2 if config.smoke else iterations

    algo_config = (
        PPOConfig()
        .env_runners(
            sample_timeout_s=10,
            rollout_fragment_length="auto",
            batch_mode="truncate_episodes",
        )
        .environment("ma_frozen_lake_v0")
        .framework("torch")
        .training(
            lr=0.00001,
            train_batch_size_per_learner=512,
            num_epochs=2,
            model={
                "fcnet_hiddens": [64, 64],
                "use_lstm": True,
                "lstm_cell_size": 64,
                "max_seq_len": 32,
            }
            if config.with_lstm
            else {
                "fcnet_hiddens": [64, 64],
            },
        )
        .multi_agent(
            policies={"shared_policy"},
            policy_mapping_fn=lambda aid, *args, **kwargs: "shared_policy",
        )
    )

    algo = algo_config.build_algo()

    rows = []
    for i in tqdm(range(1, max_iterations + 1), desc="Train iter."):
        result = algo.train()
        mean_reward = result["env_runners"]["episode_return_mean"]
        rows.append((i, mean_reward))

    df = pd.DataFrame(rows, columns=["iteration", "mean_reward"])
    df.to_csv(config.run_metrics_file, index=False)

    checkpoints_dir = config.run_dir / "checkpoints"
    algo.save(checkpoints_dir.absolute())
    return df
