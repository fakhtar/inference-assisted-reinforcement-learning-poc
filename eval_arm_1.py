"""
eval_arm1.py
------------
IARL Phase 2 -- Arm 1 evaluation.

Runs the trained blind PPO policy for 20 episodes with deterministic
actions and reports the baseline that Phase 3 (LLM-assisted arm) must beat:
  - mean lap completion rate
  - mean reward per episode
  - mean wall-clock time per completed lap (crashed episodes excluded
    from this specific average, but reported separately)

Usage:
    python eval_arm1.py
    python eval_arm1.py --model arm1_outputs/arm1_ppo_final.zip --episodes 20
"""

import argparse
import csv
import os
import time

from stable_baselines3 import PPO

import gymnasium as gym
import iarl  # noqa: F401 -- triggers gymnasium.register("iarl/RaceTrack-v0")


OUTPUT_DIR = "arm1_outputs"
DEFAULT_MODEL_PATH = os.path.join(OUTPUT_DIR, "arm1_ppo_final.zip")
METRICS_CSV_PATH = "arm1_baseline_metrics.csv"
COLLISION_REWARD_THRESHOLD = -5.0


def run_episode(env, model):
    """
    Runs one deterministic episode. Returns a dict of per-episode results.
    Disambiguates collision-termination from lap-completion-termination
    via info["lap"], since both surface as terminated=True.
    """
    obs, info = env.reset()
    total_reward = 0.0
    steps = 0
    collisions = 0
    start = time.time()

    terminated = False
    truncated = False

    while not (terminated or truncated):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += float(reward)
        steps += 1
        if reward <= COLLISION_REWARD_THRESHOLD:
            collisions += 1

    elapsed = time.time() - start
    lap_completed = bool(info.get("lap", 0) >= 1)

    return {
        "reward": total_reward,
        "steps": steps,
        "collisions": collisions,
        "lap_completed": lap_completed,
        "wall_clock_s": elapsed,
    }


def main():
    parser = argparse.ArgumentParser(description="IARL Phase 2 Arm 1 evaluation")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    if not os.path.exists(args.model):
        raise FileNotFoundError(
            f"Model not found at {args.model}. Run train_arm1.py first, "
            f"or pass --model with the correct path."
        )

    env = gym.make("iarl/RaceTrack-v0")
    model = PPO.load(args.model)

    print("=" * 70)
    print("IARL Phase 2 -- Arm 1 evaluation (blind PPO baseline)")
    print("=" * 70)
    print(f"Model    : {args.model}")
    print(f"Episodes : {args.episodes}")
    print("=" * 70)

    results = []
    for ep in range(1, args.episodes + 1):
        env.reset(seed=args.seed + ep)  # vary seed per episode, reproducible run-to-run
        r = run_episode(env, model)
        results.append(r)
        status = "LAP COMPLETE" if r["lap_completed"] else "no lap"
        print(
            f"[ep {ep:>2}/{args.episodes}] reward={r['reward']:>8.2f} "
            f"steps={r['steps']:>4} collisions={r['collisions']:>2} "
            f"time={r['wall_clock_s']:>6.2f}s  -> {status}"
        )

    env.close()

    n = len(results)
    lap_results = [r for r in results if r["lap_completed"]]
    n_laps = len(lap_results)

    mean_reward = sum(r["reward"] for r in results) / n
    lap_completion_rate = n_laps / n
    mean_collisions = sum(r["collisions"] for r in results) / n
    mean_wall_clock_per_lap = (
        sum(r["wall_clock_s"] for r in lap_results) / n_laps if n_laps > 0 else float("nan")
    )

    print("\n" + "=" * 70)
    print("BASELINE SUMMARY (Arm 1 -- blind PPO)")
    print("=" * 70)
    print(f"Episodes run                    : {n}")
    print(f"Lap completion rate             : {lap_completion_rate:.2%}  ({n_laps}/{n})")
    print(f"Mean reward per episode         : {mean_reward:.3f}")
    print(f"Mean collisions per episode     : {mean_collisions:.2f}")
    if n_laps > 0:
        print(f"Mean wall-clock time per lap    : {mean_wall_clock_per_lap:.2f}s  (over {n_laps} completed laps)")
    else:
        print("Mean wall-clock time per lap    : N/A -- no laps completed in this evaluation run")
    print("=" * 70)

    # Write summary CSV
    with open(METRICS_CSV_PATH, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        writer.writerow(["episodes_run", n])
        writer.writerow(["laps_completed", n_laps])
        writer.writerow(["lap_completion_rate", f"{lap_completion_rate:.4f}"])
        writer.writerow(["mean_reward_per_episode", f"{mean_reward:.4f}"])
        writer.writerow(["mean_collisions_per_episode", f"{mean_collisions:.4f}"])
        writer.writerow([
            "mean_wall_clock_s_per_lap",
            f"{mean_wall_clock_per_lap:.4f}" if n_laps > 0 else "NA",
        ])

    # Also write full per-episode detail alongside, for anyone auditing the
    # summary numbers later.
    detail_path = os.path.join(
        os.path.dirname(METRICS_CSV_PATH) or ".", "arm1_eval_episodes.csv"
    )
    with open(detail_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["episode", "reward", "steps", "collisions", "lap_completed", "wall_clock_s"])
        for i, r in enumerate(results, start=1):
            writer.writerow([
                i, f"{r['reward']:.3f}", r["steps"], r["collisions"],
                int(r["lap_completed"]), f"{r['wall_clock_s']:.3f}",
            ])

    print(f"\nSummary written to : {METRICS_CSV_PATH}")
    print(f"Per-episode detail  : {detail_path}")


if __name__ == "__main__":
    main()