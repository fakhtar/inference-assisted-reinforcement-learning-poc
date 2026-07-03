"""
tests/test_one_lap.py
=====================
A scripted one-lap navigation test for the IARL RaceEnv.

Instead of random actions, this test uses a hand-designed action sequence
derived from the track geometry and physics constants confirmed by the
unit tests. The purpose is to verify that a complete lap is physically
possible and that the checkpoint and lap completion logic fires correctly.

Segment action plan
-------------------
S1  Bottom straight    : ACCELERATE to build speed, then DO_NOTHING to cruise
S2  Bottom-right sweeper: BRAKE to slow, TURN_RIGHT to navigate the curve
S3  Right connector    : DO_NOTHING heading upward
S4  Hairpin            : BRAKE to MIN_SPEED, TURN_RIGHT for ~35 steps
S5  Top straight       : ACCELERATE and DO_NOTHING heading left
S6  Left sweeper       : BRAKE to slow, TURN_RIGHT to curve downward
S7  Left descent       : DO_NOTHING back to start

Each segment is defined as a list of (action, steps) pairs.
The script executes them in order and prints state at every step.
Screenshots saved at start, after each segment, and at lap end.

Run from repo root:
    python tests/test_one_lap.py
"""

import sys
import os
import math
import time
import numpy as np
import pygame

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gymnasium as gym
import iarl  # noqa: F401

# ---------------------------------------------------------------------------
# Output directory
# ---------------------------------------------------------------------------

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "test_outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Action constants (matches race_env.py)
# ---------------------------------------------------------------------------

TURN_LEFT  = 0
TURN_RIGHT = 1
ACCELERATE = 2
BRAKE      = 3
DO_NOTHING = 4

ACTION_NAMES = {
    TURN_LEFT:  "TURN_LEFT ",
    TURN_RIGHT: "TURN_RIGHT",
    ACCELERATE: "ACCELERATE",
    BRAKE:      "BRAKE     ",
    DO_NOTHING: "DO_NOTHING",
}

# ---------------------------------------------------------------------------
# Scripted lap sequence
#
# Format: (segment_name, [(action, steps), (action, steps), ...])
#
# Physics reminders from unit tests:
#   Speed 3.0 → turn_rate ≈ 0.075 rad/step  (1.8px/step net after 0.02 resist)
#   Speed 2.0 → turn_rate ≈ 0.090 rad/step  (tightest turning)
#   Speed 8.0 → turn_rate ≈ 0.000 rad/step  (cannot steer at full speed)
#   90°  turn at speed 2.0 ≈ 18 steps of TURN_RIGHT
#   180° turn at speed 2.0 ≈ 35 steps of TURN_RIGHT (hairpin)
# ---------------------------------------------------------------------------

LAP_SEQUENCE = [
    # ------------------------------------------------------------------
    # S1: Bottom straight — left to right (~430px at x=490)
    # Accelerate hard to build speed, then cruise at near-max.
    # Stay on the straight: no turning, just forward momentum.
    # ------------------------------------------------------------------
    ("S1 Bottom straight", [
        (ACCELERATE, 10),   # build speed: 3.0 -> ~7.98
        (DO_NOTHING, 35),   # cruise to end of straight at ~7.98px/step
    ]),

    # ------------------------------------------------------------------
    # S2: Bottom-right sweeper — curve upward and right (~90° right turn)
    # Must slow down before turning or turn_rate ≈ 0 at max speed.
    # Brake to ~2-3px/step, then turn right while crawling through curve.
    # ------------------------------------------------------------------
    ("S2 Bottom-right sweeper", [
        (BRAKE,      4),    # 7.98 -> ~4.0
        (BRAKE,      2),    # ~4.0 -> ~2.0 (MIN_SPEED)
        (TURN_RIGHT, 22),   # turn right ~90° at speed 2.0
        (ACCELERATE, 3),    # rebuild a little speed for the connector
    ]),

    # ------------------------------------------------------------------
    # S3: Right connector — heading upward, short vertical segment
    # Car is now pointing upward (heading ≈ -90° / 270°).
    # Just travel straight upward a short distance.
    # ------------------------------------------------------------------
    ("S3 Right connector", [
        (DO_NOTHING, 10),   # travel upward ~20px
        (BRAKE,      3),    # slow for hairpin approach
    ]),

    # ------------------------------------------------------------------
    # S4: Hairpin — tight 180° right turn at top-right
    # Must be at MIN_SPEED. Turn right for ~35 steps.
    # Car arrives heading upward, exits heading downward-left.
    # ------------------------------------------------------------------
    ("S4 Hairpin", [
        (BRAKE,      3),    # ensure MIN_SPEED
        (TURN_RIGHT, 38),   # 180° turn: 3.14rad / 0.09rad/step ≈ 35 steps
                            # extra steps to complete the arc cleanly
    ]),

    # ------------------------------------------------------------------
    # S5: Top straight — heading left (~370px)
    # Car exits hairpin heading left (heading ≈ 180°).
    # Accelerate and cruise.
    # ------------------------------------------------------------------
    ("S5 Top straight", [
        (ACCELERATE, 8),    # build speed
        (DO_NOTHING, 42),   # cruise left across the top
    ]),

    # ------------------------------------------------------------------
    # S6: Left sweeper — curve downward (~90° right turn)
    # Brake first, then turn right to curve from heading-left to heading-down.
    # ------------------------------------------------------------------
    ("S6 Left sweeper", [
        (BRAKE,      4),    # slow down
        (BRAKE,      2),    # ensure MIN_SPEED
        (TURN_RIGHT, 20),   # turn right ~90°
    ]),

    # ------------------------------------------------------------------
    # S7: Left descent — heading downward back to start (~180px)
    # Short angled straight back to (120, 490).
    # ------------------------------------------------------------------
    ("S7 Left descent", [
        (ACCELERATE, 3),    # small speed boost
        (DO_NOTHING, 20),   # coast home
    ]),
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def save_screenshot(env, label):
    surf = env.unwrapped._render_top_down_surface(w=800, h=600)
    fname = os.path.join(OUTPUT_DIR, f"lap_{label}.png")
    pygame.image.save(surf, fname)
    print(f"  [screenshot] {fname}")
    return fname


def print_state(step, action, info, reward, terminated):
    pos = env.unwrapped._pos
    print(f"  {step:>4}  {ACTION_NAMES[action]}  "
          f"spd={info['speed']:>5.2f}  "
          f"hdg={math.degrees(info['heading']):>7.2f}°  "
          f"x={pos[0]:>6.1f}  y={pos[1]:>6.1f}  "
          f"cp={info['checkpoint']:>2}  "
          f"prog={info['track_progress']:.3f}  "
          f"r={reward:>7.3f}  "
          f"{'DONE' if terminated else ''}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 72)
    print("IARL RaceEnv — Scripted One-Lap Navigation Test")
    print("=" * 72)

    env = gym.make("iarl/RaceTrack-v0", render_mode="human")
    obs, info = env.reset(seed=0)

    print(f"\nStart state:")
    pos = env.unwrapped._pos
    print(f"  pos=({pos[0]:.1f}, {pos[1]:.1f})  "
          f"speed={info['speed']:.2f}  "
          f"heading={math.degrees(info['heading']):.2f}°")

    save_screenshot(env, "00_start")

    global_step   = 0
    total_reward  = 0.0
    lap_completed = False
    terminated    = False

    print(f"\n{'Step':>4}  {'Action':>10}  {'Spd':>5}  "
          f"{'Hdg°':>7}  {'X':>6}  {'Y':>6}  "
          f"{'CP':>2}  {'Prog':>5}  {'Reward':>7}")
    print("-" * 72)

    for seg_idx, (seg_name, actions) in enumerate(LAP_SEQUENCE):
        print(f"\n--- {seg_name} ---")

        for action, steps in actions:
            for _ in range(steps):
                if terminated:
                    break

                obs, reward, terminated, truncated, info = env.step(action)
                global_step  += 1
                total_reward += reward

                print_state(global_step, action, info, reward, terminated)

                env.render()

                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        print("\n[INFO] Window closed.")
                        env.close()
                        sys.exit(0)

                time.sleep(0.04)

                if info["lap"] >= 1:
                    lap_completed = True
                    terminated    = True
                    break

            if terminated:
                break

        # Screenshot after each segment
        seg_label = f"{seg_idx+1:02d}_{seg_name.split()[0].lower()}"
        save_screenshot(env, seg_label)

        if terminated:
            break

    # Final screenshot
    save_screenshot(env, "final")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 72)
    print("LAP SUMMARY")
    print("=" * 72)
    print(f"  Total steps        : {global_step}")
    print(f"  Total reward       : {total_reward:.3f}")
    print(f"  Checkpoints passed : {info['checkpoint']}")
    print(f"  Laps completed     : {info['lap']}")
    print(f"  Track progress     : {info['track_progress']:.3f}")
    print(f"  Lap completed      : {'YES' if lap_completed else 'NO'}")

    if lap_completed:
        print("\n  [PASS] Full lap completed successfully.")
    else:
        pos = env.unwrapped._pos
        print(f"\n  [INCOMPLETE] Lap not finished.")
        print(f"  Final pos=({pos[0]:.1f}, {pos[1]:.1f})  "
              f"progress={info['track_progress']:.3f}")
        print(f"  Adjust the LAP_SEQUENCE timing and re-run.")

    print(f"\n  Screenshots saved to: {OUTPUT_DIR}")
    print("\n[DONE] Close the pygame window to exit.")

    # Hold window open
    while True:
        env.render()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                env.close()
                sys.exit(0)
        time.sleep(0.05)