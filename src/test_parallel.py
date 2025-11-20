import time
import frozen_lake

env = frozen_lake.env(render_mode="human", seed=42, flatten_observations=True)

obs, infos = env.reset()
env.render()

N = 0
start = time.time()

for _ in range(100):
    N+=1
    actions = {agent: env.action_space(agent).sample() for agent in env.agents}
    obs, rewards, terminations, truncations, infos = env.step(actions)
    
    # Example: break when all agents done
    if all(terminations.values()) or all(truncations.values()):
        break
    time.sleep(1)

env.close()
end = time.time()
print(f"Average step time: {(end - start) / N:.6f} seconds per step")