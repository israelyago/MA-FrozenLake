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

# All go to goal
TRAJECTORY = [
    frozen_lake.DOWN, frozen_lake.DOWN, frozen_lake.DOWN,
    frozen_lake.DOWN, frozen_lake.DOWN, frozen_lake.DOWN,
    frozen_lake.RIGHT, frozen_lake.RIGHT, frozen_lake.RIGHT,
    frozen_lake.RIGHT, frozen_lake.RIGHT, frozen_lake.RIGHT,
    frozen_lake.DOWN, frozen_lake.DOWN, frozen_lake.DOWN,
    frozen_lake.RIGHT, frozen_lake.RIGHT, frozen_lake.DO_NOTHING, 
    frozen_lake.LEFT, frozen_lake.DO_NOTHING, frozen_lake.DO_NOTHING, 
    frozen_lake.DO_NOTHING, frozen_lake.DO_NOTHING, frozen_lake.RIGHT, 
    frozen_lake.RIGHT, frozen_lake.DO_NOTHING, frozen_lake.DO_NOTHING, 
]

def main():
    seed = 42
    env = frozen_lake.env(seed=seed, render_mode="human")
    env.reset(seed=seed)

    trajectory_counter = 0
    for agent in env.agent_iter():
        obs, reward, termination, truncation, info = env.last()        
        if termination or truncation:
            action_bundle = (None, 0)
        else:
            action_bundle = env.action_space(agent).sample()        

            # DEBUG
            if trajectory_counter < len(TRAJECTORY):
                action = TRAJECTORY[trajectory_counter]
                action_bundle = (action, 0)

        env.step(action_bundle)

        trajectory_counter += 1

        all_done = all(env.terminations[a] or env.truncations[a] for a in env.agents)
        if all_done:
            for agent in env.possible_agents:
                last_reward = env.rewards[agent]
                print(f"Agent: {agent} won as last reward: {last_reward}")
            break
    
    env.close()

if __name__ == "__main__":
    main()
