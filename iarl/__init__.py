"""
IARL — Inference Assisted Reinforcement Learning
=================================================
Package initialisation. Registers the RaceEnv environment with Gymnasium
so it can be instantiated via:

    import gymnasium as gym
    import iarl  # noqa: F401  (triggers registration)

    env = gym.make("iarl/RaceTrack-v0")
    env = gym.make("iarl/RaceTrack-v0", render_mode="human")
"""

from gymnasium.envs.registration import register

register(
    id="iarl/RaceTrack-v0",
    entry_point="iarl.envs.race_env:RaceEnv",
    max_episode_steps=2000,
)