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
import json
import os
import time
from collections import deque

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

# "Reliable completion" -- the operational definition we're locking in for
# the whole sweep, since we agreed to checkpoint the model at this point
# specifically (not just the final, post-convergence model). Defined as:
# the first episode where the trailing RELIABLE_WINDOW episodes have a lap
# completion rate >= RELIABLE_THRESHOLD. 20/0.90 means >=18 of the last 20
# episodes completed the lap -- strict enough to mean something, lenient
# enough to not be thrown off by one unlucky episode on harder tracks.
RELIABLE_WINDOW = 20
RELIABLE_THRESHOLD = 0.90


# ---------------------------------------------------------------------------
# Callback: per-episode metric logging (same design as Arm 1's, unchanged
# logic -- info["track_progress"] and info["lap"] are still populated by
# race_env.py regardless of what observation the policy actually sees)
# ---------------------------------------------------------------------------

class EpisodeMetricsCallback(BaseCallback):
    def __init__(self, csv_path, n_envs, reliable_checkpoint_path, enabled=True, verbose=1):
        super().__init__(verbose)
        self.csv_path = csv_path
        self.n_envs = n_envs
        self.enabled = enabled
        self.reliable_checkpoint_path = reliable_checkpoint_path
        self.episode_num = 0
        self.episode_reward = np.zeros(n_envs, dtype=np.float64)
        self.episode_length = np.zeros(n_envs, dtype=np.int64)
        self.episode_collisions = np.zeros(n_envs, dtype=np.int64)
        self.episode_max_progress = np.zeros(n_envs, dtype=np.float64)
        self.start_time = None
        self.first_lap_logged = False
        self._csv_file = None
        self._csv_writer = None
        # Rolling window of lap_completed values, in global episode-completion
        # order (not per-env) -- reliable completion is a property of the
        # policy's recent behavior overall, not any one worker's history.
        self._recent_completions = deque(maxlen=RELIABLE_WINDOW)
        self.reliable_completion_episode = None
        self.reliable_completion_timestep = None
        self.reliable_completion_wallclock = None
        self.first_lap_episode = None
        self.first_lap_timestep = None
        self.first_lap_wallclock = None

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
                    self.first_lap_episode = self.episode_num
                    self.first_lap_timestep = self.num_timesteps
                    self.first_lap_wallclock = elapsed
                    print(
                        f"\n[MILESTONE] First lap completed -- "
                        f"episode {self.episode_num}, "
                        f"timestep {self.num_timesteps}, "
                        f"wall-clock {elapsed:.1f}s\n"
                    )

                # Reliable-completion tracking: rolling window across ALL
                # envs in global episode-completion order.
                self._recent_completions.append(lap_completed)
                if (self.reliable_completion_episode is None
                        and len(self._recent_completions) == RELIABLE_WINDOW
                        and sum(self._recent_completions) / RELIABLE_WINDOW >= RELIABLE_THRESHOLD):
                    self.reliable_completion_episode = self.episode_num
                    self.reliable_completion_timestep = self.num_timesteps
                    self.reliable_completion_wallclock = elapsed
                    if self.reliable_checkpoint_path:
                        self.model.save(self.reliable_checkpoint_path)
                    print(
                        f"\n[MILESTONE] RELIABLE completion reached -- "
                        f"{sum(self._recent_completions)}/{RELIABLE_WINDOW} of last "
                        f"{RELIABLE_WINDOW} episodes completed the lap. "
                        f"episode {self.episode_num}, timestep {self.num_timesteps}, "
                        f"wall-clock {elapsed:.1f}s. Checkpoint saved.\n"
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
    parser.add_argument("--track", type=str, default="basra_loop",
                         help="Which track to train on (basra_loop, circle, oval, "
                              "rectangle, spur, bramble, sawtooth).")
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

    # Per-track, per-seed output directory -- critical for the sweep:
    # without this, every run overwrites the previous one's files.
    run_dir = os.path.join("arm2_outputs", args.track, f"seed{args.seed}")
    csv_path = os.path.join(run_dir, "arm2_training_log.csv")
    checkpoint_dir = os.path.join(run_dir, "checkpoints")
    model_path = os.path.join(run_dir, "arm2_ppo_final.zip")
    reliable_checkpoint_path = os.path.join(run_dir, "arm2_ppo_reliable_completion.zip")
    tensorboard_dir = os.path.join(run_dir, "tb")
    manifest_path = os.path.join(run_dir, "manifest.json")

    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)

    def make_env():
        env = gym.make("iarl/RaceTrack-v0", render_top_down=False, render_front_camera=False,
                        track_name=args.track)
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
        tensorboard_log=tensorboard_dir if args.tensorboard else None,
    )

    metrics_callback = EpisodeMetricsCallback(
        csv_path=csv_path, n_envs=n_envs,
        reliable_checkpoint_path=reliable_checkpoint_path, enabled=not args.no_csv,
    )
    callbacks = [metrics_callback]

    if not args.no_checkpoints:
        checkpoint_callback = CheckpointCallback(
            save_freq=max(args.checkpoint_freq // n_envs, 1),
            save_path=checkpoint_dir,
            name_prefix="arm2_ppo",
        )
        callbacks.append(checkpoint_callback)
        approx_n_checkpoints = total_timesteps // args.checkpoint_freq

    print("=" * 70)
    print("IARL Phase 2 -- Arm 2 (structured-state PPO, full-effort baseline)")
    print("=" * 70)
    print(f"Track       : {args.track}")
    print(f"Seed        : {args.seed}")
    print(f"Observation : 14-dim structured state (no pixels, no CNN)")
    print(f"Action space: Discrete(5)")
    print(f"Policy      : MlpPolicy")
    print(f"Parallel envs: {n_envs} ({'SubprocVecEnv' if n_envs > 1 else 'DummyVecEnv'})")
    print(f"target_kl   : {args.target_kl}")
    print(f"Total steps : {total_timesteps:,}")
    print(f"Log CSV     : {csv_path}")
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

    model.save(model_path)
    print(f"\nTraining complete in {elapsed / 60:.1f} minutes.")
    print(f"Model saved to: {model_path}")
    print(f"Per-episode metrics: {csv_path}")

    if not metrics_callback.first_lap_logged:
        print(
            "\n[WARNING] No lap was completed within this training run. "
            "Consider increasing --total-timesteps and re-running."
        )
    if metrics_callback.reliable_completion_episode is None:
        print(
            "\n[WARNING] Reliable completion threshold "
            f"({RELIABLE_THRESHOLD:.0%} of last {RELIABLE_WINDOW} episodes) was "
            "never reached. No reliable-completion checkpoint was saved."
        )
    else:
        print(f"Reliable completion checkpoint: {reliable_checkpoint_path}")

    # Manifest -- ties this run's results together with its config, for the
    # eventual cross-track, cross-seed analysis. One of these per run dir;
    # the aggregation script reads all of them plus each run's CSV.
    manifest = {
        "track": args.track,
        "seed": args.seed,
        "total_timesteps_requested": total_timesteps,
        "total_timesteps_actual": model.num_timesteps,
        "wall_clock_minutes": round(elapsed / 60, 2),
        "n_envs": n_envs,
        "device": args.device,
        "hyperparameters": {
            "n_steps": N_STEPS, "batch_size": BATCH_SIZE, "n_epochs": N_EPOCHS,
            "learning_rate": LEARNING_RATE, "gamma": GAMMA, "clip_range": CLIP_RANGE,
            "target_kl": args.target_kl,
        },
        "first_lap_completed": metrics_callback.first_lap_logged,
        "first_lap": {
            "episode": metrics_callback.first_lap_episode,
            "timestep": metrics_callback.first_lap_timestep,
            "wall_clock_s": metrics_callback.first_lap_wallclock,
        },
        "reliable_completion": {
            "window": RELIABLE_WINDOW,
            "threshold": RELIABLE_THRESHOLD,
            "episode": metrics_callback.reliable_completion_episode,
            "timestep": metrics_callback.reliable_completion_timestep,
            "wall_clock_s": metrics_callback.reliable_completion_wallclock,
        },
        "total_episodes": metrics_callback.episode_num,
        "csv_path": csv_path,
        "model_path": model_path,
        "reliable_checkpoint_path": (
            reliable_checkpoint_path if metrics_callback.reliable_completion_episode is not None else None
        ),
    }
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Manifest saved to: {manifest_path}")

    vec_env.close()


if __name__ == "__main__":
    main()