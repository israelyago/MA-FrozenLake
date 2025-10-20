from time import sleep
import numpy as np
from stable_baselines3 import PPO
from gymnasium import ObservationWrapper
import gymnasium as gym
import frozen_lake

class OneHotObsWrapper(ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        n = env.observation_space.n
        self.observation_space = gym.spaces.Box(0, 1, (n,), dtype=np.float32)

    def observation(self, obs):
        onehot = np.zeros(self.observation_space.shape[0], dtype=np.float32)
        onehot[obs] = 1.0
        return onehot

def run_ppo(env, steps: int):
    env = OneHotObsWrapper(env)
    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=steps)
    model.save("trained_model")    
    return model

def main():
    seed = 42
    env = frozen_lake.env(seed=seed, render_mode="human")
    env.reset(seed=seed)

    for agent in env.agent_iter():
        observation, reward, termination, truncation, info = env.last()
        if termination or truncation:
            action = (frozen_lake.DO_NOTHING, 0)
        else:
            # this is where you would insert your policy
            action = env.action_space(agent).sample()
        env.step(action)

        all_done = all(env.terminations[a] or env.truncations[a] for a in env.possible_agents)
        if all_done:
            break
    
    env.close()

if __name__ == "__main__":
    main()
