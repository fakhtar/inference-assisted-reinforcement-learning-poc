import sys
import os
import time

# Same path setup as test_env.py -- allows running from tests/ without installing the package
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gymnasium as gym
import iarl  # noqa: F401 -- triggers gymnasium.register()

env = gym.make("iarl/RaceTrack-v0")
env.reset()

N = 200
t0 = time.time()
for _ in range(N):
    obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
    if terminated or truncated:
        env.reset()
elapsed = time.time() - t0

print(f"{N} steps in {elapsed:.2f}s")
print(f"{elapsed / N * 1000:.1f} ms per step")
print(f"~{N / elapsed:.1f} steps/sec")

env.close()