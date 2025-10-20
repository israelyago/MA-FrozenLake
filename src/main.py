from time import sleep
import numpy as np
from stable_baselines3 import PPO
from frozen_lake import FrozenLakeEnv
from gymnasium import ObservationWrapper
import gymnasium as gym

class OneHotObsWrapper(ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        n = env.observation_space.n
        self.observation_space = gym.spaces.Box(0, 1, (n,), dtype=np.float32)

    def observation(self, obs):
        onehot = np.zeros(self.observation_space.shape[0], dtype=np.float32)
        onehot[obs] = 1.0
        return onehot

def visualize(model, env, games: int):
    env = OneHotObsWrapper(env)
    obs, _ = env.reset()
    while games > 0:
        actions, _ = model.predict(obs)
        obs, reward, terminated, truncated, info = env.step(int(actions))
        done = terminated or truncated
        env.render()
        sleep(1/100)
        if done:
            games -= 1
            obs, _ = env.reset()

def run_ppo(env, steps: int):
    env = OneHotObsWrapper(env)
    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=steps)
    model.save("trained_model")    
    return model

def main():
    env = FrozenLakeEnv(
            is_slippery=False,
            desc=None,
            map_name=None,
            success_rate=4/5,
            reward_schedule=(1, -1, -0.01),
        )
    model = run_ppo(env, steps=10_000)
    env.render_mode = "human"
    visualize(model, env, games=10)

if __name__ == "__main__":
    main()
