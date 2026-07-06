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
import time

from stable_baselines3 import PPO

import gymnasium as gym
import iarl  # noqa: F401


def main():
    parser = argparse.ArgumentParser(description="Watch one Arm 1 episode render live")
    parser.add_argument("--model", type=str, default="arm1_outputs/arm1_ppo_final.zip")
    parser.add_argument("--seed", type=int, default=None,
                         help="Optional seed for reproducibility across repeated viewings")
    args = parser.parse_args()

    env = gym.make("iarl/RaceTrack-v0", render_mode="human")
    model = PPO.load(args.model)

    obs, info = env.reset(seed=args.seed)
    env.render()

    steps = 0
    total_reward = 0.0
    terminated = False
    truncated = False

    print("Watching one episode. Close the pygame window or wait for it to end.")
    print("(Deterministic actions -- this is the policy's real learned behavior, not exploration.)\n")

    while not (terminated or truncated):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        env.render()

        total_reward += float(reward)
        steps += 1

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