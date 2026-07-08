"""
watch_episode.py
----------------
Loads the trained Arm 1 model and plays back ONE episode with a live
pygame window (render_mode="human"), using the top-down view so you can
see the whole track and exactly where the car crashes.

This is NOT training and NOT evaluation metrics -- it's purely a visual
sanity check. Deterministic actions (no exploration noise) so you see
the policy's actual learned behavior, not random exploration.

Usage:
    python watch_episode.py
    python watch_episode.py --model arm1_outputs/arm1_ppo_final.zip
"""

import argparse
import math
import time
from collections import deque

from stable_baselines3 import PPO

import gymnasium as gym
import iarl  # noqa: F401

ACTION_NAMES = {0: "TURN_LEFT", 1: "TURN_RIGHT", 2: "ACCELERATE", 3: "BRAKE", 4: "DO_NOTHING"}


def _wrap_pi(x):
    return (x + math.pi) % (2 * math.pi) - math.pi


def _lateral_offset_px(base_env):
    """
    Signed distance from centerline in pixels, using the same convention
    as structured_state.py -- positive = left of track direction. Computed
    directly here (not via the wrapper) so this works for diagnosing ANY
    arm's model, whether or not --structured is passed.
    """
    idx = base_env._closest_idx
    c = base_env.centerline[idx]
    N = base_env.N
    prev = base_env.centerline[(idx - 1) % N]
    nxt = base_env.centerline[(idx + 1) % N]
    t = nxt - prev
    tangent = math.atan2(t[1], t[0])
    nx, ny = -math.sin(tangent), math.cos(tangent)
    dx, dy = base_env._pos[0] - c[0], base_env._pos[1] - c[1]
    return dx * nx + dy * ny


def main():
    parser = argparse.ArgumentParser(description="Watch one Arm 1/2/3 episode render live")
    parser.add_argument("--model", type=str, default="arm1_outputs/arm1_ppo_final.zip")
    parser.add_argument("--seed", type=int, default=None,
                         help="Optional seed for reproducibility across repeated viewings")
    parser.add_argument("--structured", action="store_true",
                         help="Wrap the env with StructuredStateWrapper -- use this for "
                              "Arm 2/3 models, which expect the 14-dim structured-state "
                              "vector, not the raw pixel observation Arm 1 uses.")
    parser.add_argument("--diagnostic-window", type=int, default=40,
                         help="Number of most-recent steps to print in full detail "
                              "right before the episode ends.")
    args = parser.parse_args()

    env = gym.make("iarl/RaceTrack-v0", render_mode="human")
    if args.structured:
        from iarl.wrappers.structured_state import StructuredStateWrapper
        env = StructuredStateWrapper(env)

    model = PPO.load(args.model)
    base_env = env.unwrapped

    obs, info = env.reset(seed=args.seed)
    env.render()

    steps = 0
    total_reward = 0.0
    terminated = False
    truncated = False
    prev_heading_deg = math.degrees(base_env._heading)
    diagnostic_buffer = deque(maxlen=args.diagnostic_window)

    print("Watching one episode. Close the pygame window or wait for it to end.")
    print("(Deterministic actions -- this is the policy's real learned behavior, not exploration.)\n")

    while not (terminated or truncated):
        action, _ = model.predict(obs, deterministic=True)
        action_int = int(action)
        obs, reward, terminated, truncated, info = env.step(action)
        env.render()

        total_reward += float(reward)
        steps += 1

        heading_deg = math.degrees(base_env._heading)
        heading_delta = heading_deg - prev_heading_deg
        # normalize delta to [-180,180] in case of wraparound
        while heading_delta > 180: heading_delta -= 360
        while heading_delta < -180: heading_delta += 360
        prev_heading_deg = heading_deg

        lateral_px = _lateral_offset_px(base_env)

        diagnostic_buffer.append({
            "step": steps,
            "action": ACTION_NAMES.get(action_int, str(action_int)),
            "heading_deg": heading_deg,
            "heading_delta": heading_delta,
            "speed": info["speed"],
            "lateral_px": lateral_px,
            "progress": info["track_progress"],
            "reward": reward,
        })

        if steps % 30 == 0:  # print roughly once per second at FPS=30
            print(
                f"step {steps:>4} | progress {info['track_progress']:.1%} | "
                f"speed {info['speed']:.2f} | reward so far {total_reward:.2f}"
            )

    print("\n" + "=" * 60)
    print("EPISODE ENDED")
    print("=" * 60)
    print(f"Total steps       : {steps}")
    print(f"Final track progress : {info['track_progress']:.1%} of one lap")
    print(f"Laps completed    : {info['lap']}")
    print(f"Total reward      : {total_reward:.2f}")

    print(f"\n{'='*90}")
    print(f"DIAGNOSTIC: last {len(diagnostic_buffer)} steps before episode end")
    print(f"{'='*90}")
    print(f"{'step':>5} {'action':>12} {'heading':>9} {'d_head':>8} {'speed':>6} {'lat_px':>8} {'progress':>9} {'reward':>8}")
    for row in diagnostic_buffer:
        print(
            f"{row['step']:>5} {row['action']:>12} {row['heading_deg']:>+8.1f} "
            f"{row['heading_delta']:>+7.2f} {row['speed']:>6.2f} {row['lateral_px']:>+8.2f} "
            f"{row['progress']:>8.1%} {row['reward']:>+8.3f}"
        )
    print(f"{'='*90}")
    print("action legend: TURN_LEFT decreases heading, TURN_RIGHT increases heading")
    print("(compare d_head sign against what the track's geometry requires at that point)")

    if info['lap'] == 0:
        print(
            f"\n-> Crashed at {info['track_progress']:.1%} around the track. "
            f"Run this again a few times (try --seed with different values, "
            f"or omit --seed for a fresh random start) to see if it dies in "
            f"roughly the same spot each time (a specific track feature, "
            f"most likely the hairpin) or at random locations."
        )

    # Keep the window open briefly so you can see the final frame
    time.sleep(2)
    env.close()


if __name__ == "__main__":
    main()