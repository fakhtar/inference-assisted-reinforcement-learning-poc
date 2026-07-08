"""
train_arm2.py
-------------
IARL Phase 2 -- Experimental Arm 2: full-effort conventional RL baseline.

Blind to pixels entirely -- no CNN, no image observation at all. The
policy receives the 14-dim structured-state vector (StructuredStateWrapper):
position/heading/speed relative to the track, plus a tiered arc-length
lookahead of upcoming curvature and track width. This is deliberately
given every reasonable informational/architectural advantage a
conventional RL pipeline could have, with NO LLM involvement -- this is
the real baseline Arm 3 has to beat, not Arm 1.

Usage:
    python train_arm2.py --total-timesteps 503808 --n-envs 6 --device cuda --tensorboard
    python train_arm2.py --smoke-test
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
from iarl.wrappers.structured_state import StructuredStateWrapper


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DEFAULT_TOTAL_TIMESTEPS = 503_808  # closest multiple of n_steps*n_envs to 500k
N_STEPS = 2048
BATCH_SIZE = 64
N_EPOCHS = 10
LEARNING_RATE = 3e-4
GAMMA = 0.99
CLIP_RANGE = 0.2
TARGET_KL = 0.03  # caps policy-update size; Arm 1's final rollouts showed
                   # approx_kl spiking to ~0.15 (5x the healthy ~0.01-0.03
                   # range) with no early-stopping guard. This adds that
                   # guard for Arm 2 onward, since Arm 2/3's convergence
                   # curves are the ones that matter for the paper.

COLLISION_REWARD_THRESHOLD = -5.0

OUTPUT_DIR = "arm2_outputs"
CSV_PATH = os.path.join(OUTPUT_DIR, "arm2_training_log.csv")
CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, "checkpoints")
MODEL_PATH = os.path.join(OUTPUT_DIR, "arm2_ppo_final.zip")
TENSORBOARD_DIR = os.path.join(OUTPUT_DIR, "tb")


# ---------------------------------------------------------------------------
# Callback: per-episode metric logging (same design as Arm 1's, unchanged
# logic -- info["track_progress"] and info["lap"] are still populated by
# race_env.py regardless of what observation the policy actually sees)
# ---------------------------------------------------------------------------

class EpisodeMetricsCallback(BaseCallback):
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
    parser = argparse.ArgumentParser(description="IARL Phase 2 Arm 2 training (structured-state PPO baseline)")
    parser.add_argument("--total-timesteps", type=int, default=DEFAULT_TOTAL_TIMESTEPS)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--tensorboard", action="store_true")
    parser.add_argument("--n-envs", type=int, default=4)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--checkpoint-freq", type=int, default=50_000)
    parser.add_argument("--no-checkpoints", action="store_true")
    parser.add_argument("--no-csv", action="store_true")
    parser.add_argument("--target-kl", type=float, default=TARGET_KL,
                         help="Caps policy-update size; None disables. Added after Arm 1's "
                              "late-training approx_kl instability (spiked to ~0.15 with no guard).")
    args = parser.parse_args()

    total_timesteps = 2000 if args.smoke_test else args.total_timesteps

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    def make_env():
        env = gym.make("iarl/RaceTrack-v0", render_top_down=False, render_front_camera=False)
        env = StructuredStateWrapper(env)
        env = Monitor(env)
        return env

    n_envs = 1 if args.smoke_test else args.n_envs
    vec_env_cls = DummyVecEnv if n_envs == 1 else SubprocVecEnv
    vec_env = make_vec_env(make_env, n_envs=n_envs, seed=args.seed, vec_env_cls=vec_env_cls)

    model = PPO(
        policy="MlpPolicy",
        env=vec_env,
        n_steps=N_STEPS,
        batch_size=BATCH_SIZE,
        n_epochs=N_EPOCHS,
        learning_rate=LEARNING_RATE,
        gamma=GAMMA,
        clip_range=CLIP_RANGE,
        target_kl=args.target_kl,
        verbose=1,
        seed=args.seed,
        device=args.device,
        tensorboard_log=TENSORBOARD_DIR if args.tensorboard else None,
    )

    metrics_callback = EpisodeMetricsCallback(csv_path=CSV_PATH, n_envs=n_envs, enabled=not args.no_csv)
    callbacks = [metrics_callback]

    if not args.no_checkpoints:
        checkpoint_callback = CheckpointCallback(
            save_freq=max(args.checkpoint_freq // n_envs, 1),
            save_path=CHECKPOINT_DIR,
            name_prefix="arm2_ppo",
        )
        callbacks.append(checkpoint_callback)
        approx_n_checkpoints = total_timesteps // args.checkpoint_freq

    print("=" * 70)
    print("IARL Phase 2 -- Arm 2 (structured-state PPO, full-effort baseline)")
    print("=" * 70)
    print(f"Observation : 14-dim structured state (no pixels, no CNN)")
    print(f"Action space: Discrete(5)")
    print(f"Policy      : MlpPolicy")
    print(f"Parallel envs: {n_envs} ({'SubprocVecEnv' if n_envs > 1 else 'DummyVecEnv'})")
    print(f"target_kl   : {args.target_kl}")
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