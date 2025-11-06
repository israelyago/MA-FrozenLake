import time
import frozen_lake

env = frozen_lake.env(render_mode="human", seed=42, flatten_observations=True)
# env = frozen_lake.env(render_mode=None, seed=42, flatten_observations=True)
env.reset()

N = 0
start = time.time()

for agent in env.agent_iter(max_iter=100):
    obs, reward, termination, truncation, info = env.last()
    if termination or truncation:
        action = None  # No-op for done agents
    else:
        action = env.action_space(agent).sample()

    N+=1
    env.step(action)

    all_done = all(env.terminations[a] or env.truncations[a] for a in env.possible_agents)
    if all_done:
        print("✅ All agents done — ending test loop.")
        break

end = time.time()
print(f"Average step time: {(end - start) / N:.6f} seconds per step")