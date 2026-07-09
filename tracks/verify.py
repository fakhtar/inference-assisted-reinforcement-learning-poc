"""
verify.py
---------
Reusable track verification pipeline. Takes any hand-built track's
centerline + half_widths and runs the same four checks we developed
and proved out on the hairpin fix:
  1. Tangent continuity
  2. Analytic/discrete curvature margin vs. the car's min turning radius
  3. Autopilot driving the ACTUAL physics through a full lap
  4. Robustness under injected position noise

Not a track generator -- tracks are still hand-built, one-off, same as
race_env.py's _build_track(). This module only checks them.
"""
import math
import numpy as np

# Exact physics constants from race_env.py
MAX_SPEED, MIN_SPEED, START_SPEED = 8.0, 2.0, 3.0
ACCELERATION_STEP, BRAKE_STEP, ROLLING_RESIST = 0.5, 1.0, 0.02
BASE_TURN_RATE, TURN_SPEED_DECAY = 0.12, 0.015
GRAZE_DISTANCE = 4
TURN_LEFT, TURN_RIGHT, ACCELERATE, BRAKE, DO_NOTHING = 0, 1, 2, 3, 4

CAR_MIN_TURN_RATE = max(0.01, BASE_TURN_RATE - TURN_SPEED_DECAY * MIN_SPEED)
CAR_MIN_TURN_RADIUS = MIN_SPEED / CAR_MIN_TURN_RATE  # ~22.2px


def process_geometry(raw_points, raw_widths):
    """Same dedup + tangent + arc-length computation as race_env.py."""
    pts = np.array(raw_points, dtype=np.float64)
    hws = np.array(raw_widths, dtype=np.float64)
    keep = [0]
    for i in range(1, len(pts)):
        if np.linalg.norm(pts[i] - pts[keep[-1]]) > 0.5:
            keep.append(i)
    pts = pts[keep]; hws = hws[keep]
    N = len(pts)

    tangent_angles = np.zeros(N)
    for i in range(N):
        prev = pts[(i - 1) % N]; nxt = pts[(i + 1) % N]
        t = nxt - prev
        tangent_angles[i] = math.atan2(t[1], t[0])

    arc = np.zeros(N)
    for i in range(1, N):
        arc[i] = arc[i - 1] + np.linalg.norm(pts[i] - pts[i - 1])
    total = arc[-1] + np.linalg.norm(pts[0] - pts[-1])

    return pts, hws, tangent_angles, arc, total, N


def check_initial_heading(pts, tangent_angles):
    """
    Stage 0. race_env.py's reset() hardcodes starting heading=0.0 (east),
    regardless of track. If point[0]'s true tangent doesn't match that,
    the car starts every episode with a heading error it has to correct
    before it can even begin tracking the track properly -- on a tight
    track this can cause an immediate, confusing failure that looks like
    a geometry problem but is actually just a point-ordering mismatch.
    """
    initial_tangent_deg = math.degrees(tangent_angles[0])
    mismatch = abs(initial_tangent_deg)
    mismatch = min(mismatch, 360 - mismatch)
    return mismatch

def check_tangent_continuity(pts, tangent_angles, N, skip_idx=None):
    """
    Stage 1. skip_idx: indices to exclude (e.g. the pre-existing,
    unrelated loop-closure discontinuity at idx 0 on the original track --
    pass skip_idx={0} to reproduce that exclusion; None checks everything).
    """
    skip_idx = skip_idx or set()
    max_d, worst_i = 0, None
    for i in range(N):
        if i in skip_idx:
            continue
        d = abs(math.degrees(tangent_angles[i] - tangent_angles[(i - 1) % N]))
        d = min(d, 360 - d)
        if d > max_d:
            max_d, worst_i = d, i
    return max_d, worst_i


def check_curvature_margin(pts, hws, N):
    """Stage 2. Discrete 3-point circumradius at every point."""
    worst_radius, worst_idx = float('inf'), None
    for i in range(N):
        p_prev, p_curr, p_next = pts[(i-1) % N], pts[i], pts[(i+1) % N]
        a = np.linalg.norm(p_curr - p_prev)
        b = np.linalg.norm(p_next - p_curr)
        c = np.linalg.norm(p_next - p_prev)
        s = (a + b + c) / 2
        area = max(s * (s-a) * (s-b) * (s-c), 0) ** 0.5
        if area < 1e-6:
            continue  # straight, effectively infinite radius
        r = (a * b * c) / (4 * area)
        if r < worst_radius:
            worst_radius, worst_idx = r, i
    return worst_radius, worst_idx


def find_closest_idx_hillclimb(pos, prev_idx, pts, N, max_steps=3):
    idx = prev_idx
    cur_dist = np.linalg.norm(pts[idx] - pos)
    for _ in range(max_steps):
        idx_fwd, idx_bwd = (idx + 1) % N, (idx - 1) % N
        d_fwd = np.linalg.norm(pts[idx_fwd] - pos)
        d_bwd = np.linalg.norm(pts[idx_bwd] - pos)
        if d_fwd <= cur_dist and d_fwd <= d_bwd:
            idx, cur_dist = idx_fwd, d_fwd
        elif d_bwd < cur_dist:
            idx, cur_dist = idx_bwd, d_bwd
        else:
            break
    return idx


def _autopilot_action(pos, heading, speed, closest_idx, pts, hws, arc, total, N, local_radius):
    def wrap_pi(x): return (x + math.pi) % (2*math.pi) - math.pi

    target_arc = (arc[closest_idx] + 35) % total
    target_idx = int(np.argmin(np.abs(arc - target_arc)))
    to_target = pts[target_idx] - pos
    target_heading = math.atan2(to_target[1], to_target[0])
    heading_err = wrap_pi(target_heading - heading)

    tightest_ahead = 1e6
    for look in range(10, 221, 10):
        a2 = (arc[closest_idx] + look) % total
        i2 = int(np.argmin(np.abs(arc - a2)))
        tightest_ahead = min(tightest_ahead, local_radius[i2])

    target_speed = MIN_SPEED
    for sp in np.arange(MIN_SPEED, MAX_SPEED, 0.25):
        tr = max(0.01, BASE_TURN_RATE - TURN_SPEED_DECAY * sp)
        if (sp / tr) > tightest_ahead * 0.6:
            target_speed = max(MIN_SPEED, sp - 0.5)
            break
        target_speed = sp

    if abs(heading_err) > math.radians(6):
        return TURN_RIGHT if heading_err > 0 else TURN_LEFT
    if speed > target_speed + 0.3:
        return BRAKE
    if speed < target_speed - 0.3:
        return ACCELERATE
    return DO_NOTHING


def autopilot_lap_test(pts, hws, N, perturbation_px=0.0, seed=0, max_steps=3000):
    """Stage 3/4. Drives the ACTUAL physics equations through a full lap."""
    arc = np.zeros(N)
    for i in range(1, N):
        arc[i] = arc[i-1] + np.linalg.norm(pts[i]-pts[i-1])
    total = arc[-1] + np.linalg.norm(pts[0]-pts[-1])

    local_radius = np.full(N, 1e6)
    for i in range(N):
        p_prev, p_curr, p_next = pts[(i-1)%N], pts[i], pts[(i+1)%N]
        a = np.linalg.norm(p_curr-p_prev); b = np.linalg.norm(p_next-p_curr); c = np.linalg.norm(p_next-p_prev)
        s = (a+b+c)/2
        area = max(s*(s-a)*(s-b)*(s-c), 0)**0.5
        if area > 1e-6:
            local_radius[i] = (a*b*c)/(4*area)

    rng = np.random.default_rng(seed)
    pos = pts[0].copy()
    heading = 0.0
    speed = START_SPEED
    closest_idx = 0
    checkpoint_indices = [int(np.argmin(np.abs(arc - total*i/10))) for i in range(10)]
    next_checkpoint = 0
    laps_completed = 0
    steps = 0
    min_margin = float('inf')

    def wrap_pi(x): return (x + math.pi) % (2*math.pi) - math.pi

    while steps < max_steps:
        steps += 1
        action = _autopilot_action(pos, heading, speed, closest_idx, pts, hws, arc, total, N, local_radius)
        turn_rate = max(0.01, BASE_TURN_RATE - TURN_SPEED_DECAY*speed)
        if action == TURN_LEFT: heading -= turn_rate
        elif action == TURN_RIGHT: heading += turn_rate
        elif action == ACCELERATE: speed = min(MAX_SPEED, speed + ACCELERATION_STEP)
        elif action == BRAKE: speed = max(MIN_SPEED, speed - BRAKE_STEP)
        speed = max(MIN_SPEED, speed - ROLLING_RESIST)
        heading = wrap_pi(heading)
        pos = pos + speed * np.array([math.cos(heading), math.sin(heading)])
        if perturbation_px > 0:
            pos = pos + rng.uniform(-perturbation_px, perturbation_px, size=2)

        closest_idx = find_closest_idx_hillclimb(pos, closest_idx, pts, N)
        center = pts[closest_idx]; hw = hws[closest_idx]
        dist = np.linalg.norm(pos - center)
        margin = (hw + GRAZE_DISTANCE) - dist
        min_margin = min(min_margin, margin)
        if dist > hw + GRAZE_DISTANCE:
            return dict(success=False, steps=steps, laps=laps_completed, min_margin=min_margin, status="COLLISION")

        cp_idx = checkpoint_indices[next_checkpoint]
        idx_dist = min(abs(closest_idx-cp_idx), N-abs(closest_idx-cp_idx))
        if idx_dist <= 3:
            next_checkpoint += 1
            if next_checkpoint >= 10:
                next_checkpoint = 0
                laps_completed += 1
                if laps_completed >= 1:
                    return dict(success=True, steps=steps, laps=laps_completed, min_margin=min_margin, status="LAP COMPLETE")
    return dict(success=False, steps=steps, laps=laps_completed, min_margin=min_margin, status="TIMEOUT")


def verify_track(raw_points, raw_widths, track_name, skip_tangent_idx=None,
                  n_noise_seeds=10, noise_levels=(0.5, 1.0, 2.0, 3.0)):
    """
    Runs all 4 stages against a hand-built track and prints a full report.
    Returns a dict of the key numbers for later difficulty-profile analysis.
    """
    pts, hws, tangent_angles, arc, total, N = process_geometry(raw_points, raw_widths)

    print("=" * 70)
    print(f"VERIFYING: {track_name}")
    print("=" * 70)
    print(f"Points: {N}, total length: {total:.1f}px")

    heading_mismatch = check_initial_heading(pts, tangent_angles)
    print(f"\nStage 0 -- Initial heading match: point[0]'s tangent is "
          f"{heading_mismatch:.1f} deg off from race_env's hardcoded start heading (0/east)")
    stage0_pass = heading_mismatch <= 5
    print(f"  {'PASS' if stage0_pass else 'FAIL -- reorder/rotate points so point[0] starts heading east'} (threshold: 5 deg)")

    max_d, worst_i = check_tangent_continuity(pts, tangent_angles, N, skip_tangent_idx)
    print(f"\nStage 1 -- Tangent continuity: max discontinuity {max_d:.2f} deg at idx {worst_i}")
    stage1_pass = max_d <= 30
    print(f"  {'PASS' if stage1_pass else 'FAIL'} (threshold: 30 deg)")

    worst_r, worst_ci = check_curvature_margin(pts, hws, N)
    margin = worst_r - CAR_MIN_TURN_RADIUS
    print(f"\nStage 2 -- Curvature margin: tightest radius {worst_r:.1f}px at idx {worst_ci}")
    print(f"  Car min turning radius: {CAR_MIN_TURN_RADIUS:.1f}px, margin: {margin:+.1f}px")
    stage2_pass = margin > 5
    print(f"  {'PASS' if stage2_pass else 'FAIL'} (threshold: >5px margin)")

    clean = autopilot_lap_test(pts, hws, N, perturbation_px=0.0)
    print(f"\nStage 3 -- Autopilot clean lap: {clean['status']} after {clean['steps']} steps, "
          f"min_margin={clean['min_margin']:.1f}px")
    stage3_pass = clean['success']
    print(f"  {'PASS' if stage3_pass else 'FAIL'}")

    print(f"\nStage 4 -- Robustness under injected noise:")
    stage4_pass = True
    for noise in noise_levels:
        results = [autopilot_lap_test(pts, hws, N, perturbation_px=noise, seed=s) for s in range(n_noise_seeds)]
        n_success = sum(1 for r in results if r['success'])
        worst_margin = min(r['min_margin'] for r in results)
        print(f"  +/-{noise}px: {n_success}/{n_noise_seeds} laps completed, worst margin {worst_margin:.1f}px")
        if n_success < n_noise_seeds:
            stage4_pass = False

    overall = stage0_pass and stage1_pass and stage2_pass and stage3_pass and stage4_pass
    print(f"\n{'='*70}")
    print(f"OVERALL: {'PASS -- ready to train' if overall else 'FAIL -- needs geometry revision'}")
    print(f"{'='*70}")

    return dict(
        track_name=track_name, N=N, total_length=total,
        tangent_max_discontinuity=max_d, curvature_worst_radius=worst_r,
        curvature_margin=margin, autopilot_clean_margin=clean['min_margin'],
        overall_pass=overall, pts=pts, hws=hws,
    )