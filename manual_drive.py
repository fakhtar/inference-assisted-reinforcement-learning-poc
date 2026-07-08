"""
manual_drive.py
----------------
Manual keyboard control of the RaceEnv. Purpose: validate whether the
hairpin (or any section of track) is actually navigable at all,
independent of whether the RL policy has learned to navigate it.

If you can clear it with careful, deliberate driving -- especially
braking well before entry -- that's strong evidence the Arm 2 plateau
(~46%) is a policy/training limitation, not a structural problem with
the environment. If even careful manual control can't get through it,
that's real evidence of a genuine geometry issue (e.g. the hairpin's
minimum turning radius not fitting its corridor width) worth fixing in
race_env.py before trusting further Arm 2/3 results.

Controls:
    Left/Right arrow (or A/D) : turn left / turn right
    Up arrow (or W)           : accelerate
    Down arrow (or S)         : brake
    (no key held)             : do nothing
    Esc or close window       : quit

Tip: the hairpin is narrow (half_width=20 vs 30-38 elsewhere). Brake
down to minimum speed well BEFORE entry (progress ~45%) and hold a
steady turn through it rather than trying to correct sharply mid-turn.

Usage:
    python manual_drive.py
"""

import math
import sys
import time
from collections import deque

import numpy as np
import pygame
import gymnasium as gym
import iarl  # noqa: F401

ACTION_NAMES = {0: "TURN_LEFT", 1: "TURN_RIGHT", 2: "ACCELERATE", 3: "BRAKE", 4: "DO_NOTHING"}

HAIRPIN_ENTRY = 0.457  # from earlier arc-length calculation
HAIRPIN_EXIT = 0.597


def _wrap_pi(x):
    return (x + math.pi) % (2 * math.pi) - math.pi


def _lateral_offset_px(base_env):
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
    env = gym.make("iarl/RaceTrack-v0", render_mode="human")
    base_env = env.unwrapped

    obs, info = env.reset()
    env.render()

    print("=" * 70)
    print("MANUAL DRIVE -- is the hairpin navigable at all?")
    print("=" * 70)
    print("Controls: Left/Right or A/D = turn, Up/W = accelerate, Down/S = brake")
    print("Esc or close window to quit early")
    print(f"Hairpin spans roughly {HAIRPIN_ENTRY:.1%} to {HAIRPIN_EXIT:.1%} of the lap.")
    print("Tip: brake to minimum speed BEFORE entry, hold a steady turn through it.")
    print("=" * 70)

    steps = 0
    total_reward = 0.0
    terminated = False
    truncated = False
    max_progress = 0.0
    entered_announced = False
    exited_announced = False
    prev_heading_deg = math.degrees(base_env._heading)
    diagnostic_buffer = deque(maxlen=40)
    closest_idx_prev = base_env._closest_idx

    while not (terminated or truncated):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                env.close()
                sys.exit(0)
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                env.close()
                sys.exit(0)

        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT] or keys[pygame.K_a]:
            action = 0
        elif keys[pygame.K_RIGHT] or keys[pygame.K_d]:
            action = 1
        elif keys[pygame.K_UP] or keys[pygame.K_w]:
            action = 2
        elif keys[pygame.K_DOWN] or keys[pygame.K_s]:
            action = 3
        else:
            action = 4

        obs, reward, terminated, truncated, info = env.step(action)
        env.render()  # paces itself to FPS internally, no extra throttling needed

        # --- Debug overlay: draw exactly what collision detection is
        # measuring, directly on top of the render. race_env's render()
        # already flipped the display; we draw more on the same surface
        # and flip again so these markers show up without needing to
        # modify race_env.py itself.
        screen = base_env._screen
        idx = base_env._closest_idx
        car_screen_pos = (int(base_env._pos[0]), int(base_env._pos[1]))
        cl_pos = base_env.centerline[idx]
        cl_screen_pos = (int(cl_pos[0]), int(cl_pos[1]))
        true_dist = float(np.linalg.norm(base_env._pos - cl_pos))
        hw_here = float(base_env.half_widths[idx])

        # Bright magenta dot = the centerline point collision is measured against
        pygame.draw.circle(screen, (255, 0, 255), cl_screen_pos, 5)
        # Line from car to that point
        pygame.draw.line(screen, (255, 0, 255), car_screen_pos, cl_screen_pos, 2)
        # White outline circle at the car's true position (sanity check vs the red dot)
        pygame.draw.circle(screen, (255, 255, 255), car_screen_pos, 8, width=2)

        if steps % 5 == 0 or reward <= -0.99:  # print often near graze/collision
            print(
                f"  [overlay] step {steps} car_pos={base_env._pos.round(1)} "
                f"closest_pt={cl_pos.round(1)} true_dist={true_dist:.2f} "
                f"hw={hw_here:.0f} collision_at={hw_here+4:.0f}"
            )

        pygame.display.flip()

        total_reward += reward
        steps += 1
        max_progress = max(max_progress, info["track_progress"])

        heading_deg = math.degrees(base_env._heading)
        heading_delta = heading_deg - prev_heading_deg
        while heading_delta > 180: heading_delta -= 360
        while heading_delta < -180: heading_delta += 360
        prev_heading_deg = heading_deg

        lat = _lateral_offset_px(base_env)
        idx_jump = base_env._closest_idx - closest_idx_prev
        closest_idx_prev = base_env._closest_idx

        diagnostic_buffer.append({
            "step": steps,
            "action": ACTION_NAMES[action],
            "closest_idx": base_env._closest_idx,
            "idx_jump": idx_jump,
            "heading_deg": heading_deg,
            "heading_delta": heading_delta,
            "speed": info["speed"],
            "lat_px": lat,
            "progress": info["track_progress"],
            "reward": reward,
        })

        if info["track_progress"] >= HAIRPIN_ENTRY and not entered_announced:
            entered_announced = True
            print(f"\n>>> ENTERED HAIRPIN at step {steps}, progress {info['track_progress']:.1%}\n")
        if info["track_progress"] >= HAIRPIN_EXIT and not exited_announced:
            exited_announced = True
            print(f"\n>>> CLEARED HAIRPIN at step {steps}, progress {info['track_progress']:.1%}\n")

        if steps % 15 == 0:
            print(
                f"step {steps:>4} | action {ACTION_NAMES[action]:>10} | "
                f"progress {info['track_progress']:.1%} | speed {info['speed']:.2f} | "
                f"lat_px {lat:+.1f} | reward {reward:+.3f}"
            )

    print("\n" + "=" * 70)
    print("DRIVE ENDED")
    print("=" * 70)
    print(f"Total steps        : {steps}")
    print(f"Max track progress : {max_progress:.1%}")
    print(f"Laps completed     : {info['lap']}")
    print(f"Total reward       : {total_reward:.2f}")

    print(f"\n{'='*100}")
    print(f"DIAGNOSTIC: last {len(diagnostic_buffer)} steps before episode end")
    print(f"{'='*100}")
    print(f"{'step':>5} {'action':>12} {'idx':>5} {'idx_jmp':>8} {'heading':>9} {'d_head':>8} {'speed':>6} {'lat_px':>8} {'progress':>9} {'reward':>8}")
    for row in diagnostic_buffer:
        flag = " <-- IDX JUMP" if abs(row['idx_jump']) > 2 and abs(row['idx_jump']) < base_env.N - 2 else ""
        print(
            f"{row['step']:>5} {row['action']:>12} {row['closest_idx']:>5} {row['idx_jump']:>+8} "
            f"{row['heading_deg']:>+8.1f} {row['heading_delta']:>+7.2f} {row['speed']:>6.2f} "
            f"{row['lat_px']:>+8.2f} {row['progress']:>8.1%} {row['reward']:>+8.3f}{flag}"
        )
    print(f"{'='*100}")

    # Save the exact crash frame to disk -- easier to share than a manual
    # screenshot, and guarantees you get the precise moment, not a frame
    # or two later.
    screenshot_path = "crash_frame.png"
    try:
        pygame.image.save(base_env._screen, screenshot_path)
        print(f"\nScreenshot saved to: {screenshot_path}")
    except Exception as e:
        print(f"\n[warning] could not save screenshot: {e}")

    if info["lap"] >= 1:
        print("\n-> LAP COMPLETED. The hairpin (and full track) IS navigable.")
    elif max_progress > HAIRPIN_EXIT:
        print("\n-> Cleared the hairpin but crashed elsewhere on this attempt.")
        print("   The hairpin itself IS navigable -- the failure point is elsewhere.")
    elif max_progress > HAIRPIN_ENTRY:
        print(f"\n-> Entered the hairpin but crashed inside it at {max_progress:.1%}.")
        print("   Try again with more deliberate braking before entry. If you consistently")
        print("   can't clear it even driving carefully, that's real evidence of a")
        print("   structural/geometry issue worth fixing in race_env.py.")
    else:
        print(f"\n-> Crashed before reaching the hairpin, at {max_progress:.1%}.")

    print("\n" + "=" * 70)
    print("PAUSED at the crash frame -- window will stay open.")
    print("Take a screenshot now if you want, or just use crash_frame.png.")
    print("Press any key in the game window (or close it) to exit.")
    print("=" * 70)

    waiting = True
    while waiting:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                waiting = False
            if event.type == pygame.KEYDOWN:
                waiting = False
        pygame.time.wait(50)  # avoid busy-spinning the CPU while paused

    env.close()


if __name__ == "__main__":
    main()