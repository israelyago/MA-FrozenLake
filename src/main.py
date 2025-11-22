import argparse
import os
from copy import deepcopy
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import pandas as pd
import ray
import yaml
from tqdm import tqdm

from metrics import plot
from trainer import TrainConfig, train

ray.init(
    runtime_env={
        "env_vars": {"PYTHONWARNINGS": "ignore::DeprecationWarning"},
    },
)

def get_args():
    parser = argparse.ArgumentParser(
        description="Experiment configuration for Multi-Agent Frozen Lake",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--runs",
        type=int,
        default=5,
        help="How many runs per experiment configuration",
    )

    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run smoke test",
    )

    args = parser.parse_args()
    return args


def get_args_from_file(config: Path):
    """
    Load experiment configuration from a YAML file.
    Returns an argparse.Namespace for drop-in compatibility.
    """
    with open(config, "r") as f:
        data = yaml.safe_load(f) or {}

    defaults = {
        "artifacts": Path("./artifacts"),
        "experiment": "smoke",
        "full_observability": False,
        "with_communication": True,
        "reward_schedule": [10, 1, -1, -0.001, -0.1],
        "slippery": True,
        "seed": 42,
        "smoke": False,
    }

    # Merge YAML with defaults
    cfg = {**defaults, **data}

    # Type corrections (YAML may not preserve types fully)
    cfg["artifacts"] = Path(cfg["artifacts"])
    cfg["reward_schedule"] = list(map(float, cfg["reward_schedule"]))
    cfg["seed"] = int(cfg["seed"])
    cfg["slippery"] = bool(cfg["slippery"])
    cfg["with_communication"] = bool(cfg["with_communication"])
    cfg["full_observability"] = bool(cfg["full_observability"])
    cfg["smoke"] = bool(cfg["smoke"])

    return argparse.Namespace(**cfg)


def parse_base_config(config_path: Path) -> TrainConfig:
    config = get_args_from_file(config_path)

    # Parse configuration
    if len(config.reward_schedule) != 5:
        print(
            f"🚨 Argument reward_schedule must have 5 elements, got {len(config.reward_schedule)}"
        )
    config.artifacts = config.artifacts
    if not config.artifacts.is_dir():
        print(f"🚨 --artifacts should be a dir, check {(config.artifacts)}")

    config.experiment_dir = config.artifacts / config.experiment
    config.runs_dir = config.experiment_dir / "runs"
    os.makedirs(config.artifacts, exist_ok=True)
    os.makedirs(config.experiment_dir, exist_ok=True)
    os.makedirs(config.runs_dir, exist_ok=True)

    if config.smoke:
        print("🚬🗿 Smoke testing")
    else:
        print("📢 Production mode")

    return config


def get_all_runs_from_config(base_config: TrainConfig, runs: int) -> List[TrainConfig]:
    configs = []
    for run in range(runs):
        config = deepcopy(base_config)
        config.run = run
        config.run_dir = config.runs_dir / f"run_{run}"  # <--- use f-string with run
        config.run_metrics_file = config.run_dir / "metrics.csv"
        config.seed = base_config.seed + run
        configs.append(config)
    return configs


def save_config_yaml(config, filepath: Path):
    filepath.parent.mkdir(parents=True, exist_ok=True)

    # Convert config (Namespace or dataclass) → dict
    if hasattr(config, "__dict__"):
        data = {k: v for k, v in config.__dict__.items()}
    else:
        raise TypeError("Unsupported config type for saving.")

    # Convert Path objects to strings (YAML cannot serialize Path directly)
    for k, v in data.items():
        if isinstance(v, Path):
            data[k] = str(v)

    with open(filepath, "w") as f:
        yaml.safe_dump(data, f, sort_keys=False)


def aggregate_metrics(dfs: List[pd.DataFrame]) -> pd.DataFrame:
    all_df = pd.concat(dfs, keys=range(len(dfs)), names=["run"])
    return (
        all_df.groupby("iteration")
        .agg(mean_reward=("mean_reward", "mean"), std_reward=("mean_reward", "std"))
        .reset_index()
    )


@ray.remote
def run_experiment(config: TrainConfig, args):
    # Inherit smoke flag
    config.smoke = config.smoke or args.smoke

    # Path: artifacts/<experiment>/runs/run_X/config.yaml
    config_path = config.run_dir / "config.yaml"

    # Save run config YAML
    save_config_yaml(config, config_path)
    df = train(config)
    return df


def main():
    args = get_args()

    # Loop over ALL config files in src/configs/*.yaml
    config_files = sorted(Path("src/configs").glob("*.yaml"))
    print(f"💡 Found {len(config_files)} experiments config files")

    for cfg_file in tqdm(config_files, desc="Experiment"):
        print(f"🔧 Using base config: {cfg_file}")

        # Load base config from file
        base_config = parse_base_config(cfg_file)

        # Create N run-configs for this base config
        run_configs = get_all_runs_from_config(base_config, args.runs)

        # --- PARALLEL LAUNCH ----
        futures = [run_experiment.remote(cfg, args) for cfg in run_configs]

        # --- WAIT & COLLECT ----
        dfs = ray.get(futures)

        aggregated = aggregate_metrics(dfs)
        aggregated_path = base_config.experiment_dir / "metrics.csv"
        aggregated.to_csv(aggregated_path, index=False)

    plot()


if __name__ == "__main__":
    main()
