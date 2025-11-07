import os
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.registry import register_env
from ray.rllib.env import PettingZooEnv
import frozen_lake
from game_engine import MovementAction
from ray.rllib.core.rl_module.rl_module import RLModule
from ray.rllib.core.rl_module.torch.torch_rl_module import TorchRLModule
from ray.rllib.algorithms.ppo.ppo_catalog import PPOCatalog
from ray.rllib.core.rl_module.default_model_config import DefaultModelConfig
from ray.rllib.algorithms.ppo.torch.default_ppo_torch_rl_module import (
    DefaultPPOTorchRLModule
)

# All go to goal
TRAJECTORY = [
    MovementAction.DOWN, MovementAction.DOWN, MovementAction.DOWN,
    MovementAction.DOWN, MovementAction.DOWN, MovementAction.DOWN,
    MovementAction.RIGHT, MovementAction.RIGHT, MovementAction.RIGHT,
    MovementAction.RIGHT, MovementAction.RIGHT, MovementAction.RIGHT,
    MovementAction.DOWN, MovementAction.DOWN, MovementAction.DOWN,
    MovementAction.RIGHT, MovementAction.RIGHT, MovementAction.DO_NOTHING, 
    MovementAction.LEFT, MovementAction.DO_NOTHING, MovementAction.DO_NOTHING, 
    MovementAction.DO_NOTHING, MovementAction.DO_NOTHING, MovementAction.RIGHT, 
    MovementAction.RIGHT, MovementAction.DO_NOTHING, MovementAction.DO_NOTHING, 
]

def main():
    seed = 42
    # RLlib environment creator
    def env_creator(config=None):
        env = frozen_lake.env(render_mode=None, seed=seed, flatten_observations=True)
        env = PettingZooEnv(env)
        env.reset(seed=seed)
        return env

    register_env("ma_frozen_lake_v0", env_creator)

    config = (
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
        )
        .multi_agent(
            policies={"shared_policy"},
            policy_mapping_fn=lambda aid, *args, **kwargs: "shared_policy",
        )
        # .debugging(log_level="DEBUG")
        # .rl_module(
        #     rl_module_spec=DefaultPPOTorchRLModule(
        #         observation_space=env.observation_space,
        #         action_space=env.action_space,
        #         model_config=DefaultModelConfig(fcnet_hiddens=[64, 64]),
        #         catalog_class=PPOCatalog,
        #     )
        # )
    )

    algo = config.build_algo()
    module = algo.get_module("shared_policy")

    total_params = sum(p.numel() for p in module.parameters() if p.requires_grad)
    print("⚛︎ Total trainable parameters:", total_params)

    print("🤖 Will train a model with PPO")
    reward_threshold = 0.0  # stop when mean reward > 0
    max_iterations = 3

    for i in range(max_iterations+1):
        result = algo.train()
        # print(result)
        # mean_reward = result["episode_reward_mean"]
        # print(f"🧪 Iteration {i}: mean_reward={mean_reward}")
        print(f"🧪 Iteration {i}")

        # if mean_reward > reward_threshold:
        #     print(f"🚀 Stopping training: mean reward {mean_reward} exceeded threshold")
        #     break
    print("💾 Saving model to ma_frozen_lake_ppo")
    save_path = os.path.abspath("./checkpoints/ma_frozen_lake_ppo")
    algo.save(save_path)
    print("✅ Done")

if __name__ == "__main__":
    main()
