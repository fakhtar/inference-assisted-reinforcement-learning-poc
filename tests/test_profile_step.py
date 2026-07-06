import sys
import os
import cProfile
import pstats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gymnasium as gym
import iarl  # noqa: F401

env = gym.make("iarl/RaceTrack-v0", render_top_down=False)
env.reset()

def run(n=200):
    for _ in range(n):
        obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
        if terminated or truncated:
            env.reset()

profiler = cProfile.Profile()
profiler.enable()
run(200)
profiler.disable()

stats = pstats.Stats(profiler)
stats.sort_stats("cumulative")
print("\n" + "=" * 70)
print("TOP 15 FUNCTIONS BY CUMULATIVE TIME")
print("=" * 70)
stats.print_stats(15)

env.close()