"""
tests/test_env.py
=================
Verification script for the IARL RaceEnv Gymnasium environment.

What this script does
---------------------
1. Instantiates the environment with render_mode="human" (pygame window).
2. Runs 100 steps with random actions.
3. Prints reward at each step with action name and speed.
4. Renders the top-down view in a pygame window at each step.
5. Confirms both front_camera and top_down observations are returned
   with correct shapes.
6. Prints a summary at the end.

Run from the repo root:
    python tests/test_env.py

Requirements:
    pip install -r requirements.txt
    Package must be importable: run from repo root or install with pip install -e .
"""

import sys
import os
import time
import numpy as np
import pygame

# Allow running from repo root without installing the package
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gymnasium as gym
import iarl  # noqa: F401 — triggers gymnasium.register()


# ---------------------------------------------------------------------------
# Action name map for readable output
# ---------------------------------------------------------------------------

ACTION_NAMES = {
    0: "TURN_LEFT ",
    1: "TURN_RIGHT",
    2: "ACCELERATE",
    3: "BRAKE     ",
    4: "DO_NOTHING",
}


# ---------------------------------------------------------------------------
# Shape verification
# ---------------------------------------------------------------------------

def verify_shapes(obs, info, step):
    """
    Confirm front_camera and top_down observations have the expected shapes.
    Prints a warning if anything is wrong. Called once at step 0.
    """
    expected_front  = (120, 160, 3)
    expected_top    = (120, 160, 3)

    front_shape = obs.shape
    top_shape   = info["top_down"].shape

    ok = True

    if front_shape != expected_front:
        print(f"  [SHAPE ERROR] front_camera: expected {expected_front}, "
              f"got {front_shape}")
        ok = False
    else:
        print(f"  [OK] front_camera shape: {front_shape}  dtype: {obs.dtype}")

    if top_shape != expected_top:
        print(f"  [SHAPE ERROR] top_down: expected {expected_top}, "
              f"got {top_shape}")
        ok = False
    else:
        print(f"  [OK] top_down shape:    {top_shape}  "
              f"dtype: {info['top_down'].dtype}")

    if ok:
        print(f"  [OK] Both observation shapes verified at step {step}.")
    else:
        print(f"  [FAIL] Shape verification failed. Check race_env.py.")

    return ok


# ---------------------------------------------------------------------------
# Main test loop
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("IARL RaceEnv — Verification Test")
    print("=" * 60)

    # -----------------------------------------------------------------------
    # 1. Instantiate
    # -----------------------------------------------------------------------
    print("\n[1/3] Instantiating environment...")
    env = gym.make("iarl/RaceTrack-v0", render_mode="human")
    print("      Environment created successfully.")
    print(f"      Observation space : {env.observation_space}")
    print(f"      Action space      : {env.action_space}")

    # -----------------------------------------------------------------------
    # 2. Reset
    # -----------------------------------------------------------------------
    print("\n[2/3] Resetting environment...")
    obs, info = env.reset(seed=42)
    print("      Reset successful.")

    # Verify shapes immediately after reset
    print("\n      Verifying observation shapes...")
    shapes_ok = verify_shapes(obs, info, step=0)

    # -----------------------------------------------------------------------
    # 3. Run 100 steps
    # -----------------------------------------------------------------------
    print("\n[3/3] Running 100 steps with random actions...")
    print("-" * 60)
    print(f"{'Step':>4}  {'Action':>10}  {'Reward':>8}  "
          f"{'Speed':>6}  {'Progress':>8}  {'Lap':>4}  {'Done':>5}")
    print("-" * 60)

    total_reward      = 0.0
    total_collisions  = 0
    steps_run         = 0
    episode_rewards   = []

    for step in range(1, 101):
        # Sample random action
        action = env.action_space.sample()

        # Step environment
        obs, reward, terminated, truncated, info = env.step(action)

        total_reward += reward
        steps_run    += 1

        if reward <= -5.0:
            total_collisions += 1

        done = terminated or truncated

        print(f"{step:>4}  {ACTION_NAMES[action]}  {reward:>8.3f}  "
              f"{info['speed']:>6.2f}  {info['track_progress']:>8.3f}  "
              f"{info['lap']:>4}  {str(done):>5}")

        # Render top-down view to pygame window
        env.render()

        # Handle pygame quit event so the window can be closed manually
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                print("\n  [INFO] Window closed by user. Ending test early.")
                env.close()
                sys.exit(0)

        # Small delay so the window is watchable at human speed
        time.sleep(0.05)

        if done:
            print(f"\n  [EPISODE END] Terminated at step {step}. "
                  f"Reward this episode: {total_reward:.3f}")
            episode_rewards.append(total_reward)
            total_reward = 0.0

            # Reset for next episode within the 100-step budget
            obs, info = env.reset()

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    print("-" * 60)
    print("\nTest Summary")
    print("-" * 60)
    print(f"  Steps run             : {steps_run}")
    print(f"  Episodes completed    : {len(episode_rewards)}")
    print(f"  Total collisions      : {total_collisions}")

    if episode_rewards:
        print(f"  Rewards per episode   : "
              f"{[round(r, 2) for r in episode_rewards]}")
        print(f"  Mean episode reward   : {np.mean(episode_rewards):.3f}")
    else:
        print(f"  Cumulative reward     : {total_reward:.3f}  "
              f"(no episode ended cleanly in 100 steps)")

    print(f"\n  Observation shapes    : {'PASS' if shapes_ok else 'FAIL'}")
    print(f"  front_camera          : (120, 160, 3) uint8")
    print(f"  top_down              : (120, 160, 3) uint8")

    print("\n[DONE] Test complete. Close the pygame window to exit.")

    # Keep window open until user closes it
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        env.render()
        time.sleep(0.1)

    env.close()


if __name__ == "__main__":
    main()