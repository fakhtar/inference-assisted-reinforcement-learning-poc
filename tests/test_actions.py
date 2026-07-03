"""
tests/test_actions.py
=====================
Unit tests for each discrete action in the IARL RaceEnv.

Each test:
  - Resets to the same starting position (seed=0)
  - Applies exactly one action repeatedly for 30 steps
  - Prints speed and heading at each step
  - Saves a screenshot at step 0 (start) and step 30 (end)
  - Evaluates a measurable PASS/FAIL criterion

Screenshots saved to: tests/test_outputs/
  test_1_accelerate_start.png  /  test_1_accelerate_end.png
  test_2_brake_start.png       /  test_2_brake_end.png
  test_3_turn_left_start.png   /  test_3_turn_left_end.png
  test_4_turn_right_start.png  /  test_4_turn_right_end.png
  test_5_do_nothing_start.png  /  test_5_do_nothing_end.png

Run from repo root:
  python tests/test_actions.py
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
# Action names
# ---------------------------------------------------------------------------

ACTION_NAMES = {
    0: "TURN_LEFT",
    1: "TURN_RIGHT",
    2: "ACCELERATE",
    3: "BRAKE",
    4: "DO_NOTHING",
}

STEPS_PER_TEST = 30

# ---------------------------------------------------------------------------
# PASS/FAIL criteria
# Each returns (passed: bool, reason: str)
# ---------------------------------------------------------------------------

def criterion_accelerate(start_speed, end_speed, start_heading, end_heading,
                          start_pos, end_pos):
    """Speed must be strictly higher than starting speed at the end."""
    if end_speed > start_speed:
        return True, f"Speed increased from {start_speed:.2f} to {end_speed:.2f}"
    return False, f"Speed did NOT increase: start={start_speed:.2f} end={end_speed:.2f}"


def criterion_brake(start_speed, end_speed, start_heading, end_heading,
                    start_pos, end_pos):
    """Speed must stay at MIN_SPEED (2.0) throughout — checked via end speed."""
    from iarl.envs.race_env import MIN_SPEED
    if abs(end_speed - MIN_SPEED) < 0.01:
        return True, f"Speed held at MIN_SPEED={MIN_SPEED:.2f} as expected"
    return False, f"Speed did not stay at MIN_SPEED: end={end_speed:.2f}"


def criterion_turn_left(start_speed, end_speed, start_heading, end_heading,
                         start_pos, end_pos):
    """
    Heading must have decreased (turned left = counter-clockwise in screen coords).
    Start heading is 0.0 (pointing right). After 30 left turns heading should be
    clearly negative (or wrapped to near +pi on the other side).
    We check that the angular difference is a left turn.
    """
    # Compute signed angular delta, unwrapped
    delta = end_heading - start_heading
    # Normalise to [-pi, pi]
    delta = (delta + math.pi) % (2 * math.pi) - math.pi
    if delta < -0.1:
        return True, (f"Heading turned left by {math.degrees(abs(delta)):.1f}° "
                      f"(start={math.degrees(start_heading):.1f}° "
                      f"end={math.degrees(end_heading):.1f}°)")
    return False, (f"Heading did NOT turn left: delta={math.degrees(delta):.1f}° "
                   f"start={math.degrees(start_heading):.1f}° "
                   f"end={math.degrees(end_heading):.1f}°)")


def criterion_turn_right(start_speed, end_speed, start_heading, end_heading,
                          start_pos, end_pos):
    """Heading must have increased (turned right = clockwise in screen coords)."""
    delta = end_heading - start_heading
    delta = (delta + math.pi) % (2 * math.pi) - math.pi
    if delta > 0.1:
        return True, (f"Heading turned right by {math.degrees(delta):.1f}° "
                      f"(start={math.degrees(start_heading):.1f}° "
                      f"end={math.degrees(end_heading):.1f}°)")
    return False, (f"Heading did NOT turn right: delta={math.degrees(delta):.1f}° "
                   f"start={math.degrees(start_heading):.1f}° "
                   f"end={math.degrees(end_heading):.1f}°)")


def criterion_do_nothing(start_speed, end_speed, start_heading, end_heading,
                          start_pos, end_pos):
    """
    Heading must be unchanged (straight line travel).
    Car must have moved forward (position changed along x axis since start
    heading is 0 = pointing right).
    Speed must be at MIN_SPEED (rolling resistance brings it there quickly).
    """
    heading_delta = abs(end_heading - start_heading)
    dx = end_pos[0] - start_pos[0]
    dy = end_pos[1] - start_pos[1]
    distance = math.sqrt(dx * dx + dy * dy)

    heading_ok  = heading_delta < 0.01
    moved_right = dx > 10        # must have moved right on the straight
    lateral_ok  = abs(dy) < 5   # must not have drifted up or down

    if heading_ok and moved_right and lateral_ok:
        return True, (f"Straight line travel confirmed: dx={dx:.1f} dy={dy:.1f} "
                      f"heading unchanged={math.degrees(end_heading):.2f}°")
    reasons = []
    if not heading_ok:
        reasons.append(f"heading changed by {math.degrees(heading_delta):.2f}°")
    if not moved_right:
        reasons.append(f"did not move right (dx={dx:.1f})")
    if not lateral_ok:
        reasons.append(f"lateral drift (dy={dy:.1f})")
    return False, "FAIL: " + ", ".join(reasons)


# ---------------------------------------------------------------------------
# Test runner
# ---------------------------------------------------------------------------

TESTS = [
    (1, "ACCELERATE", 2, criterion_accelerate),
    (2, "BRAKE",      3, criterion_brake),
    (3, "TURN_LEFT",  0, criterion_turn_left),
    (4, "TURN_RIGHT", 1, criterion_turn_right),
    (5, "DO_NOTHING", 4, criterion_do_nothing),
]


def save_screenshot(env, filename):
    """Render top-down surface and save to file."""
    surf = env.unwrapped._render_top_down_surface(w=800, h=600)
    pygame.image.save(surf, filename)


def run_test(env, test_num, test_name, action_id, criterion_fn):
    print()
    print("=" * 60)
    print(f"TEST {test_num}: {test_name}  (action={action_id})")
    print("=" * 60)

    # Reset to fixed seed for reproducibility
    obs, info = env.reset(seed=0)

    start_speed   = info["speed"]
    start_heading = info["heading"]
    start_pos     = env.unwrapped._pos.copy()

    # Save start screenshot
    start_file = os.path.join(OUTPUT_DIR,
                              f"test_{test_num}_{test_name.lower()}_start.png")
    save_screenshot(env, start_file)
    print(f"  Start screenshot saved: {start_file}")
    print(f"  Start state — speed={start_speed:.3f}  "
          f"heading={math.degrees(start_heading):.2f}°  "
          f"pos=({start_pos[0]:.1f}, {start_pos[1]:.1f})")
    print()

    print(f"  {'Step':>4}  {'Speed':>7}  {'Heading°':>9}  "
          f"{'Pos X':>7}  {'Pos Y':>7}  {'Reward':>8}")
    print(f"  {'-'*55}")

    end_speed   = start_speed
    end_heading = start_heading
    end_pos     = start_pos.copy()

    terminated = False
    for step in range(1, STEPS_PER_TEST + 1):
        obs, reward, terminated, truncated, info = env.step(action_id)
        end_speed   = info["speed"]
        end_heading = info["heading"]
        end_pos     = env.unwrapped._pos.copy()

        print(f"  {step:>4}  {end_speed:>7.3f}  "
              f"{math.degrees(end_heading):>9.2f}  "
              f"{end_pos[0]:>7.1f}  {end_pos[1]:>7.1f}  "
              f"{reward:>8.3f}")

        # Render to pygame window for live viewing
        env.render()

        # Handle window close
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                print("\n  [INFO] Window closed. Ending tests.")
                env.close()
                sys.exit(0)

        time.sleep(0.05)

        if terminated:
            print(f"\n  [NOTE] Episode terminated at step {step} "
                  f"(collision or lap complete).")
            break

    # Save end screenshot
    end_file = os.path.join(OUTPUT_DIR,
                            f"test_{test_num}_{test_name.lower()}_end.png")
    save_screenshot(env, end_file)
    print(f"\n  End screenshot saved: {end_file}")
    print(f"  End state   — speed={end_speed:.3f}  "
          f"heading={math.degrees(end_heading):.2f}°  "
          f"pos=({end_pos[0]:.1f}, {end_pos[1]:.1f})")

    # Evaluate criterion
    passed, reason = criterion_fn(
        start_speed, end_speed,
        start_heading, end_heading,
        start_pos, end_pos
    )

    status = "PASS" if passed else "FAIL"
    print(f"\n  [{status}] {reason}")

    return passed, test_name, reason


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("IARL RaceEnv — Action Unit Tests")
    print("=" * 60)
    print(f"Steps per test : {STEPS_PER_TEST}")
    print(f"Screenshots    : {OUTPUT_DIR}")

    env = gym.make("iarl/RaceTrack-v0", render_mode="human")

    results = []
    for test_num, test_name, action_id, criterion_fn in TESTS:
        passed, name, reason = run_test(
            env, test_num, test_name, action_id, criterion_fn
        )
        results.append((test_num, name, passed, reason))

    env.close()

    # Final summary
    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    all_passed = True
    for test_num, name, passed, reason in results:
        status = "PASS" if passed else "FAIL"
        print(f"  TEST {test_num} {name:<12} [{status}]  {reason}")
        if not passed:
            all_passed = False

    print()
    if all_passed:
        print("  All 5 action tests PASSED.")
    else:
        print("  One or more tests FAILED. Review screenshots and log above.")

    print(f"\n  Screenshots saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()