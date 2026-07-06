"""
train_arm1.py
-------------
IARL Phase 2 — Experimental Arm 1: Blind RL baseline.

Trains a PPO policy on iarl/RaceTrack-v0 using ONLY the front_camera
observation. No LLM, no top_down access, no reward shaping. This is the
blind baseline that Phase 3 (LLM-assisted arm) must beat on:
  - episodes to first lap completion
  - wall-clock time to first lap completion
  - mean reward per episode
  - collision rate per episode

Usage:
    python train_arm1.py
    python train_arm1.py --total-timesteps 1000000
    python train_arm1.py --smoke-test        # 2000 steps, sanity check only

Requirements:
    pip install stable-baselines3 gymnasium
    iarl package must be importable (run from repo root, or `pip install -e .`)
"""

import argparse
import csv
import os
import time

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv

import gymnasium as gym
import iarl  # noqa: F401 -- triggers gymnasium.register("iarl/RaceTrack-v0")


# ---------------------------------------------------------------------------
# Config -- starting hyperparameters. Do not change without confirming first.
# ---------------------------------------------------------------------------

DEFAULT_TOTAL_TIMESTEPS = 500_000
N_STEPS = 2048
BATCH_SIZE = 64
N_EPOCHS = 10
LEARNING_RATE = 3e-4
GAMMA = 0.99
CLIP_RANGE = 0.2

COLLISION_REWARD_THRESHOLD = -5.0  # matches the convention used in test_env.py

OUTPUT_DIR = "arm1_outputs"
CSV_PATH = os.path.join(OUTPUT_DIR, "arm1_training_log.csv")
CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, "checkpoints")
MODEL_PATH = os.path.join(OUTPUT_DIR, "arm1_ppo_final.zip")
TENSORBOARD_DIR = os.path.join(OUTPUT_DIR, "tb")


# ---------------------------------------------------------------------------
# Callback: per-episode metric logging
# ---------------------------------------------------------------------------

class EpisodeMetricsCallback(BaseCallback):
    """
    Logs one row per completed episode to CSV:
        episode, timesteps, wall_clock_s, episode_reward, episode_length,
        collisions, lap_completed

    Disambiguates the two terminated=True cases (collision vs lap
    completion) using info["lap"], since both present identically as
    `done=True` from the VecEnv otherwise.

    Prints a milestone line the first time a lap is completed.
    """

    def __init__(self, csv_path, n_envs, enabled=True, verbose=1):
        super().__init__(verbose)
        self.csv_path = csv_path
        self.n_envs = n_envs
        self.enabled = enabled
        self.episode_num = 0
        self.episode_reward = np.zeros(n_envs, dtype=np.float64)
        self.episode_length = np.zeros(n_envs, dtype=np.int64)
        self.episode_collisions = np.zeros(n_envs, dtype=np.int64)
        self.episode_max_progress = np.zeros(n_envs, dtype=np.float64)
        self.start_time = None
        self.first_lap_logged = False
        self._csv_file = None
        self._csv_writer = None

    def _on_training_start(self) -> None:
        self.start_time = time.time()
        if not self.enabled:
            return
        os.makedirs(os.path.dirname(self.csv_path), exist_ok=True)
        # Open once, keep the handle open for the whole run instead of
        # open/write/close per episode.
        self._csv_file = open(self.csv_path, "w", newline="")
        self._csv_writer = csv.writer(self._csv_file)
        self._csv_writer.writerow([
            "episode", "timesteps", "wall_clock_s", "episode_reward",
            "episode_length", "collisions", "lap_completed", "max_track_progress",
        ])
        self._csv_file.flush()

    def _on_training_end(self) -> None:
        if self._csv_file is not None:
            self._csv_file.close()
            self._csv_file = None

    def _on_step(self) -> bool:
        # Handles n_envs >= 1. self.locals["rewards"]/["dones"] are arrays
        # of length n_envs; self.locals["infos"] is a list of length n_envs.
        # Each env slot accumulates independently and logs its own episode
        # row the moment IT finishes, regardless of what other envs are doing.
        rewards = self.locals["rewards"]
        dones = self.locals["dones"]
        infos = self.locals["infos"]

        for i in range(self.n_envs):
            self.episode_reward[i] += float(rewards[i])
            self.episode_length[i] += 1
            if rewards[i] <= COLLISION_REWARD_THRESHOLD:
                self.episode_collisions[i] += 1
            self.episode_max_progress[i] = max(
                self.episode_max_progress[i], float(infos[i].get("track_progress", 0.0))
            )

            if dones[i]:
                self.episode_num += 1
                info = infos[i]
                lap_completed = int(info.get("lap", 0) >= 1)
                elapsed = time.time() - self.start_time

                if self.enabled:
                    self._csv_writer.writerow([
                        self.episode_num, self.num_timesteps, f"{elapsed:.2f}",
                        f"{self.episode_reward[i]:.3f}", self.episode_length[i],
                        self.episode_collisions[i], lap_completed,
                        f"{self.episode_max_progress[i]:.4f}",
                    ])
                    self._csv_file.flush()

                if lap_completed and not self.first_lap_logged:
                    self.first_lap_logged = True
                    print(
                        f"\n[MILESTONE] First lap completed -- "
                        f"episode {self.episode_num}, "
                        f"timestep {self.num_timesteps}, "
                        f"wall-clock {elapsed:.1f}s\n"
                    )

                if self.verbose and self.episode_num % 20 == 0:
                    print(
                        f"[ep {self.episode_num:>5}] env={i} "
                        f"steps={self.num_timesteps:>8} "
                        f"reward={self.episode_reward[i]:>8.2f} "
                        f"len={self.episode_length[i]:>4} "
                        f"collisions={self.episode_collisions[i]:>2} "
                        f"progress={self.episode_max_progress[i]:.1%} "
                        f"lap={lap_completed}"
                    )

                self.episode_reward[i] = 0.0
                self.episode_length[i] = 0
                self.episode_collisions[i] = 0
                self.episode_max_progress[i] = 0.0

        return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="IARL Phase 2 Arm 1 training (blind PPO)")
    parser.add_argument("--total-timesteps", type=int, default=DEFAULT_TOTAL_TIMESTEPS)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--smoke-test", action="store_true",
                         help="Run only 2000 timesteps to verify the pipeline works end to end.")
    parser.add_argument("--tensorboard", action="store_true",
                         help="Enable TensorBoard logging to arm1_outputs/tb")
    parser.add_argument("--n-envs", type=int, default=4,
                         help="Number of parallel environments (SubprocVecEnv). "
                              "Set to 1 to keep the original single-process behavior. "
                              "Check `os.cpu_count()` locally and leave 1-2 cores free.")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"],
                         help="Device for the policy network. 'auto' lets SB3/PyTorch "
                              "pick. Only affects the CNN forward/backward pass during "
                              "optimization -- env.step() always runs on CPU regardless.")
    parser.add_argument("--checkpoint-freq", type=int, default=50_000,
                         help="Save an intermediate model snapshot every N timesteps "
                              "(crash insurance). Each snapshot is a few MB (policy "
                              "weights + Adam optimizer state). Default 50,000 gives "
                              "~10 snapshots over a 500k run instead of ~49.")
    parser.add_argument("--no-checkpoints", action="store_true",
                         help="Skip intermediate checkpoints entirely. The final model "
                              "is still saved at the end regardless of this flag.")
    parser.add_argument("--no-csv", action="store_true",
                         help="Disable per-episode CSV logging entirely, for isolating "
                              "whether it affects wall-clock time.")
    args = parser.parse_args()

    total_timesteps = 2000 if args.smoke_test else args.total_timesteps

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    # Monitor wrapper gives us SB3's own ep_rew_mean/ep_len_mean in
    # stdout/tensorboard; our callback gives the richer per-episode CSV.
    def make_env():
        env = gym.make("iarl/RaceTrack-v0", render_top_down=False)
        env = Monitor(env)
        return env

    n_envs = 1 if args.smoke_test else args.n_envs
    vec_env_cls = DummyVecEnv if n_envs == 1 else SubprocVecEnv
    vec_env = make_vec_env(make_env, n_envs=n_envs, seed=args.seed, vec_env_cls=vec_env_cls)

    model = PPO(
        policy="CnnPolicy",
        env=vec_env,
        n_steps=N_STEPS,
        batch_size=BATCH_SIZE,
        n_epochs=N_EPOCHS,
        learning_rate=LEARNING_RATE,
        gamma=GAMMA,
        clip_range=CLIP_RANGE,
        verbose=1,
        seed=args.seed,
        device=args.device,
        tensorboard_log=TENSORBOARD_DIR if args.tensorboard else None,
    )

    metrics_callback = EpisodeMetricsCallback(csv_path=CSV_PATH, n_envs=n_envs, enabled=not args.no_csv)
    callbacks = [metrics_callback]

    if not args.no_checkpoints:
        checkpoint_callback = CheckpointCallback(
            save_freq=max(args.checkpoint_freq // n_envs, 1),  # save_freq counts per-env steps under VecEnv
            save_path=CHECKPOINT_DIR,
            name_prefix="arm1_ppo",
        )
        callbacks.append(checkpoint_callback)
        approx_n_checkpoints = total_timesteps // args.checkpoint_freq

    print("=" * 70)
    print("IARL Phase 2 -- Arm 1 (blind PPO baseline)")
    print("=" * 70)
    print(f"Observation : front_camera only, shape (120, 160, 3) uint8")
    print(f"Action space: Discrete(5)")
    print(f"Parallel envs: {n_envs} ({'SubprocVecEnv' if n_envs > 1 else 'DummyVecEnv'})")
    print(f"Total steps : {total_timesteps:,}")
    print(f"Log CSV     : {CSV_PATH}")
    if not args.no_checkpoints:
        print(f"Checkpoints : every {args.checkpoint_freq:,} steps (~{approx_n_checkpoints} saves this run)")
    else:
        print("Checkpoints : disabled (--no-checkpoints). Only the final model will be saved.")
    print("=" * 70)

    start = time.time()
    model.learn(
        total_timesteps=total_timesteps,
        callback=callbacks,
        progress_bar=True,
    )
    elapsed = time.time() - start

    model.save(MODEL_PATH)
    print(f"\nTraining complete in {elapsed / 60:.1f} minutes.")
    print(f"Model saved to: {MODEL_PATH}")
    print(f"Per-episode metrics: {CSV_PATH}")

    if not metrics_callback.first_lap_logged:
        print(
            "\n[WARNING] No lap was completed within this training run. "
            "Consider increasing --total-timesteps and re-running."
        )

    vec_env.close()


if __name__ == "__main__":
    main()