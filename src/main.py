import argparse
import os
from pathlib import Path

from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.env import ParallelPettingZooEnv
from ray.tune.registry import register_env
from tqdm import tqdm

import frozen_lake

class TrainConfig:
    full_observability = True
    with_communication = False
    slippery = False
    reward_schedule = [10, 1, -1, -0.001, -0.1]
    artifacts: Path
    experiment_dir: Path


def get_args():
    parser = argparse.ArgumentParser(
        description="Experiment configuration for Multi-Agent Frozen Lake",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--artifacts",
        type=Path,
        default=Path("artifacts"),
        help="Where experiments artifacts are stored",
    )

    parser.add_argument(
        "--experiment",
        type=str,
        default="smoke",
        help="Experiment name",
    )

    parser.add_argument(
        "--full_observability",
        type=bool,
        default=False,
        help="Should the agents see the whole map?",
    )

    parser.add_argument(
        "--with_communication",
        type=bool,
        default=True,
        help="Can the agents communicate?",
    )

    parser.add_argument(
        "--reward_schedule",
        type=float,
        nargs="+",
        default=[10, 1, -1, -0.001, -0.1],
        help="Rewards schedule: (All at goal, Agent at goal, Falling at Hole, Timeout, Step)",
    )

    parser.add_argument(
        "--slippery",
        type=bool,
        default=True,
        help="Do the agents slip when walking?",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed for random operations",
    )

    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run smoke test",
    )

    args = parser.parse_args()
    return args


def parse_config() -> TrainConfig:
    config = get_args()

    # Parse configuration
    if len(config.reward_schedule) != 5:
        print(
            f"🚨 Argument reward_schedule must have 5 elements, got {len(config.reward_schedule)}"
        )
    config.artifacts = config.artifacts.absolute()
    if not config.artifacts.is_dir():
        print(f"🚨 --artifacts should be a dir, check {(config.artifacts.absolute())}")

    config.experiment_dir = config.artifacts / config.experiment
    os.makedirs(config.artifacts, exist_ok=True)
    os.makedirs(config.experiment_dir, exist_ok=True)

    if config.smoke:
        print("🚬🗿 Smoke testing")
    else:
        print("📢 Production mode")

    return config


def train(config: TrainConfig):
    def env_creator(c):
        env = frozen_lake.env(
            render_mode=None, seed=config.seed, flatten_observations=True
        )
        env = ParallelPettingZooEnv(env)
        env.reset(seed=config.seed)
        return env

    register_env("ma_frozen_lake_v0", env_creator)

    algo_config = (
        PPOConfig()
        .env_runners(
            # rollout_fragment_length=32,
            sample_timeout_s=10,
            rollout_fragment_length="auto",
            batch_mode="truncate_episodes",
        )
        .environment("ma_frozen_lake_v0")
        .framework("torch")
        .training(
            lr=0.0001,
            train_batch_size_per_learner=128,
            num_epochs=2,
            model={
                "use_lstm": True,
                "lstm_cell_size": 64,
                "max_seq_len": 32,
            },
        )
        .multi_agent(
            policies={"shared_policy"},
            policy_mapping_fn=lambda aid, *args, **kwargs: "shared_policy",
        )
    )

    algo = algo_config.build_algo()
    module = algo.get_module("shared_policy")

    total_params = sum(p.numel() for p in module.parameters() if p.requires_grad)
    print("⚛︎ Total trainable parameters:", total_params)

    print("🤖 Will train a model with PPO")
    max_iterations = 5 if config.smoke else 300

    for i in tqdm(range(1, max_iterations + 1), desc="Train iter."):
        algo.train()
        # result = algo.train()

        # mean_reward = result["env_runners"]["episode_return_mean"]
        # print(f"🧪 Iteration {i}: mean_reward={mean_reward}")
    
    checkpoints_dir = config.experiment_dir / "checkpoints"
    print(f"💾 Saving model to checkpoints to '{checkpoints_dir}'")
    algo.save(checkpoints_dir)
    print("✅ Done")


def main():
    config = parse_config()
    train(config)


if __name__ == "__main__":
    main()
