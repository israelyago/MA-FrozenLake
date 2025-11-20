import os

import numpy as np
import torch
from ray.rllib.algorithms.ppo import PPO
from ray.rllib.env import ParallelPettingZooEnv
from ray.tune.registry import register_env

import frozen_lake


class SafePettingZooEnv(ParallelPettingZooEnv):
    """Patch RLlib's render() call to ignore render_mode argument."""

    def render(self, *args, **kwargs):
        # Ignore any render_mode that RLlib may try to pass
        try:
            return self.par_env.render()
        except TypeError:
            # Fallback for older PettingZoo versions
            return self.par_env.render


def main():
    seed = 42

    def env_creator(config=None):
        env = frozen_lake.env(seed=seed, flatten_observations=True, render_mode="human")
        env = SafePettingZooEnv(env)
        env.reset(seed=seed)
        return env

    register_env("ma_frozen_lake_v0", env_creator)

    checkpoint_dir = os.path.abspath("./checkpoints/ma_frozen_lake_ppo")
    algo = PPO.from_checkpoint(checkpoint_dir)
    env = env_creator()

    module = algo.get_module("shared_policy")  # single shared policy module
    N_ACTIONS = 5

    for run in range(10):
        obs, infos = env.reset()
        done = False
        while not done:
            actions = {a: (0, 0) for a in env.agents}

            for agent in env.agents:
                if agent not in obs:
                    continue
                obs_np = np.array(obs[agent], dtype=np.float32)
                obs_tensor = torch.tensor(obs_np, dtype=torch.float32).unsqueeze(
                    0
                )  # (1, obs_dim)

                # Build batch dict for forward_inference
                batch = {"obs": obs_tensor}
                with torch.no_grad():
                    out = module.forward_inference(batch)

                logits = out["action_dist_inputs"].squeeze(
                    0
                )  # shape: (N_ACTIONS + N_MESSAGES,)

                move_logits = torch.distributions.Categorical(logits=logits[:N_ACTIONS])
                msg_logits = torch.distributions.Categorical(logits=logits[N_ACTIONS:])

                move = move_logits.sample().item()
                msg = msg_logits.sample().item()

                action = (move, msg)
                actions[agent] = action

            obs, rewards, terminations, truncations, infos = env.step(actions)

            done = all(terminations.values()) or all(truncations.values())

    env.close()


if __name__ == "__main__":
    main()
