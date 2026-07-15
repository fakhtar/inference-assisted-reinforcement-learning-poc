"""
IARL Race Environment
=====================
A custom Gymnasium environment representing a 2D top-down race track.
Used as the shared environment for all three experimental arms of the
Inference Assisted Reinforcement Learning (IARL) experiment.

Observation (default): front_camera RGB array (bumper-cam, no car in frame)
Additional observation:  top_down RGB array (full track, car position marked)
Action space: Discrete(5) — turn_left, turn_right, accelerate, brake, do_nothing

v3 changes vs v2:
- Complete track geometry redesign. All 7 segment joins verified at 0.00px gap.
- Shape is now a clean stadium/teardrop: bottom straight, right sweeper,
  short right connector, tight hairpin (top-right), top straight, left
  sweeper, left descent. No self-intersections. All points within bounds.
- Rendering switched from single large polygon to per-segment quads,
  eliminating self-intersection artifacts.
"""

import math
import numpy as np
import pygame
import gymnasium as gym
from gymnasium import spaces


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Actions
TURN_LEFT   = 0
TURN_RIGHT  = 1
ACCELERATE  = 2
BRAKE       = 3
DO_NOTHING  = 4

# Physics
MAX_SPEED         = 8.0    # pixels per timestep at full throttle
MIN_SPEED         = 2.0    # car always has meaningful momentum
START_SPEED       = 3.0    # speed at episode start
ACCELERATION_STEP = 0.5    # speed gained per ACCELERATE action
BRAKE_STEP        = 1.0    # speed lost per BRAKE action
ROLLING_RESIST    = 0.02   # speed lost per timestep (rolling resistance)
BASE_TURN_RATE    = 0.12   # radians per timestep at zero speed
TURN_SPEED_DECAY  = 0.015  # effective_turn = BASE - TURN_SPEED_DECAY * speed

# Collision
GRAZE_DISTANCE    = 4      # pixels inside boundary = graze zone
GRAZE_PENALTY     = -1.0
COLLISION_PENALTY = -10.0

# Reward
CHECKPOINT_REWARD = 1.0
LAP_REWARD        = 10.0
TIME_PENALTY      = -0.01

# Rendering
SCREEN_W          = 800
SCREEN_H          = 600
FPS               = 30

# Camera
BUMPER_CAM_W      = 160
BUMPER_CAM_H      = 120
BUMPER_CAM_DIST   = 120
BUMPER_CAM_FOV    = 90
TOP_DOWN_W        = 160
TOP_DOWN_H        = 120

# Colours
COL_GRASS         = (34,  85,  34)
COL_TRACK         = (80,  80,  80)
COL_CENTERLINE    = (200, 200,  50)
COL_CAR           = (220,  50,  50)
COL_CHECKPOINT    = (50,  200, 200)
COL_STARTLINE     = (255, 255, 255)
COL_WALL_NEAR     = (220,  80,  80)


# ---------------------------------------------------------------------------
# Track definition
# ---------------------------------------------------------------------------

def _build_track_basra_loop():
    """
    Build the Basra Loop — a clean 7-segment closed circuit.

    All segment joins verified at 0.00px gap. All points within
    x=[90,730], y=[200,490] — well inside the 800x600 window.

    Traversal: counterclockwise.

    Segment map:
      S1  Bottom straight    (120,490) -> (550,490)   hw=38  WIDE
      S2  Right sweeper arc  (550,490) -> (730,310)   hw=34  medium
      S3  Right connector    (730,310) -> (730,200)   hw=30  medium
      S4  Hairpin arc        (730,200) -> (570,200)   hw=20  NARROW
      S5  Top straight       (570,200) -> (200,200)   hw=30  medium
      S6  Left sweeper arc   (200,200) -> (90,310)    hw=32  medium
      S7  Left descent       (90,310)  -> (120,490)   hw=34  medium

    Four trap types from geometry alone:
      Speed floor   — S1 long straight punishes passivity via time penalty
      Early braking — S2 sweeper can be taken faster than naive policy tries
      Hairpin stall — S4 physically impassable at full speed (hw=20, tight)
      Wall hugging  — S6 left sweeper outer wall is the tempting but slow line

    Returns
    -------
    points     : list of (x, y) floats
    half_widths: list of floats
    """
    points = []
    widths = []

    def arc(cx, cy, r, a_start_deg, a_end_deg, steps, hw):
        """Append arc points. Angles in degrees, screen coords (y down)."""
        for i in range(steps + 1):
            t = i / steps
            a = math.radians(a_start_deg + t * (a_end_deg - a_start_deg))
            points.append((cx + r * math.cos(a), cy + r * math.sin(a)))
            widths.append(hw)

    def straight(x0, y0, x1, y1, steps, hw):
        """Append straight segment points."""
        for i in range(steps + 1):
            t = i / steps
            points.append((x0 + t * (x1 - x0), y0 + t * (y1 - y0)))
            widths.append(hw)

    # S1: Bottom straight — wide, long. Car starts here heading right.
    straight(120, 490, 550, 490, steps=30, hw=38)

    # S2: Bottom-right sweeper
    # Center (550,310), r=180. Arc 90°->0°.
    # Verified: start=(550,490), end=(730,310). Gap to S1 end: 0.00px.
    arc(cx=550, cy=310, r=180, a_start_deg=90, a_end_deg=0, steps=20, hw=34)

    # S3: Right connector — shortened from the original (was 730,310 to
    # 730,200). Ends at 730,255 instead of 730,200 -- this shift is what
    # gives the hairpin arc a geometrically valid entry point. See notes
    # below for why the original 730,200 endpoint made tangent-matching
    # provably impossible.
    straight(730, 310, 730, 255, steps=6, hw=30)
 
    # S4: Hairpin — replaced. The original was a semicircle (180-degree
    # sweep), which can ONLY connect points with exactly OPPOSITE tangent
    # directions by construction. This hairpin needs to connect a NORTH-
    # heading entry to a WEST-heading exit -- 90 degrees apart, not 180.
    # No semicircle could ever have tangent-matched here, regardless of
    # radius or position. Replaced with a single genuine 90-degree arc,
    # radius 55 (2.5x the car's ~22.2px minimum turning radius), tangent-
    # matched exactly at both ends by construction.
    arc(cx=675, cy=255, r=55, a_start_deg=0, a_end_deg=-90, steps=24, hw=26)
 
    # S5: Top straight — lengthened from the original (was 570,200 to
    # 200,200). Now starts at 675,200 instead of 570,200, absorbing the
    # hairpin's new exit point. Still ends at 200,200, unchanged, so S6
    # and everything downstream needs no changes at all.
    straight(675, 200, 200, 200, steps=20, hw=30)

    # S6: Left sweeper arc
    # Center (200,310), r=110. Arc 270°->180°.
    # Verified: start=(200,200), end=(90,310). Gap to S5 end: 0.00px.
    arc(cx=200, cy=310, r=110, a_start_deg=270, a_end_deg=180, steps=16, hw=32)

    # S7: Left descent — car heading downward, closing loop to S1
    # Verified: start=(90,310), end=(120,490). Gap to S6 end: 0.00px.
    # Loop closure: S7 end=(120,490) == S1 start=(120,490). Gap: 0.00px.
    straight(90, 310, 120, 490, steps=14, hw=34)

    return points, widths

def _build_track_circle():
    """
    Track 2 of the difficulty sweep: Circle. Easiest-end anchor --
    constant curvature, no direction reversals, no tight spots.
    Point[0] positioned so its tangent points due east (0 deg),
    matching reset()'s hardcoded starting heading -- verified via
    the tracks/verify.py Stage 0 check before this was added here.
    """
    import math
    cx, cy, r, steps, hw = 400, 300, 278, 140, 32
    points, widths = [], []
    start_offset = -90  # degrees, gives east-facing tangent at point[0]
    for i in range(steps):
        a = math.radians(start_offset + 360 * i / steps)
        points.append((cx + r * math.cos(a), cy + r * math.sin(a)))
        widths.append(hw)
    return points, widths

def _build_track_oval():
    """
    Track 3 of the difficulty sweep: Oval. Easy end of the spectrum --
    two long straights connected by generous semicircular U-turns
    (R=90px, 4x the car's min turning radius). Both turns are 180-degree
    parallel-tangent connections -- the simple case a semicircle is
    built for, unlike the original hairpin's perpendicular-tangent
    problem.
    Point[0] starts on the bottom straight heading east, matching
    reset()'s hardcoded starting heading -- verified via tracks/verify.py
    Stage 0 before this was added here.
    """
    import math
 
    def arc(cx, cy, r, a_start_deg, a_end_deg, steps):
        return [(cx + r*math.cos(math.radians(a_start_deg + (i/steps)*(a_end_deg-a_start_deg))),
                 cy + r*math.sin(math.radians(a_start_deg + (i/steps)*(a_end_deg-a_start_deg))))
                for i in range(steps+1)]
 
    def straight(x0, y0, x1, y1, steps):
        return [(x0+(i/steps)*(x1-x0), y0+(i/steps)*(y1-y0)) for i in range(steps+1)]
 
    cx_left, cx_right, cy, R, hw = 105, 695, 300, 90, 30
    straight_steps, arc_steps = 40, 30
 
    points, widths = [], []
    def add(pts, w): points.extend(pts); widths.extend([w]*len(pts))
 
    add(straight(cx_left, cy+R, cx_right, cy+R, straight_steps), hw)   # bottom straight, east
    add(arc(cx_right, cy, R, 90, -90, arc_steps), hw)                   # right U-turn
    add(straight(cx_right, cy-R, cx_left, cy-R, straight_steps), hw)    # top straight, west
    add(arc(cx_left, cy, R, -90, -270, arc_steps), hw)                  # left U-turn, closes loop
 
    return points, widths 

def _build_track_rectangle():
    """
    Track 4 of the difficulty sweep: Rectangle. Moderate difficulty --
    4 sharp-but-feasible corners, minimal rounding (R=35px, margin
    +12.8px over the car's min turning radius -- real but deliberately
    tighter than Oval's). Each corner's incoming/outgoing edges are
    offset in both x and y, so unlike the original hairpin there's no
    same-coordinate degeneracy -- a single quarter-circle fillet per
    corner works cleanly.
    Point[0] starts on the bottom straight heading east, matching
    reset()'s hardcoded starting heading.
    """
    import math
 
    def arc(cx, cy, r, a_start_deg, a_end_deg, steps):
        return [(cx + r*math.cos(math.radians(a_start_deg + (i/steps)*(a_end_deg-a_start_deg))),
                 cy + r*math.sin(math.radians(a_start_deg + (i/steps)*(a_end_deg-a_start_deg))))
                for i in range(steps+1)]
 
    def straight(x0, y0, x1, y1, steps):
        return [(x0+(i/steps)*(x1-x0), y0+(i/steps)*(y1-y0)) for i in range(steps+1)]
 
    x_left, x_right, y_top, y_bottom, R, hw = 110, 690, 135, 465, 35, 28
    straight_steps, arc_steps = 40, 20
 
    points, widths = [], []
    def add(pts, w): points.extend(pts); widths.extend([w]*len(pts))
 
    add(straight(x_left+R, y_bottom, x_right-R, y_bottom, straight_steps), hw)   # bottom, east
    add(arc(x_right-R, y_bottom-R, R, 90, 0, arc_steps), hw)                      # BR fillet
    add(straight(x_right, y_bottom-R, x_right, y_top+R, straight_steps), hw)     # right, north
    add(arc(x_right-R, y_top+R, R, 0, -90, arc_steps), hw)                       # TR fillet
    add(straight(x_right-R, y_top, x_left+R, y_top, straight_steps), hw)        # top, west
    add(arc(x_left+R, y_top+R, R, 270, 180, arc_steps), hw)                      # TL fillet
    add(straight(x_left, y_top+R, x_left, y_bottom-R, straight_steps), hw)      # left, south
    add(arc(x_left+R, y_bottom-R, R, 180, 90, arc_steps), hw)                    # BL fillet, closes loop
 
    return points, widths

def _build_track_spur():
    """
    Track 5 of the difficulty sweep: Spur. Moderate difficulty -- a
    single isolated elbow/protrusion on an otherwise circular base.
    Built as ONE continuous curve (Gaussian bump in radius vs. angle)
    rather than a branching neck that separates and rejoins -- this
    guarantees tangent continuity everywhere by construction, avoiding
    the junction-matching problem the original hairpin needed fixing.
    Margin +28.0px over the car's min turning radius at the tightest
    point (the bump's peak curvature). Total length ~1749px.
    Point[0] starts heading east, matching reset()'s hardcoded start.
    """
    import math
 
    cx, cy, base_r, bump_amplitude = 400, 300, 258, 110
    bump_center_deg, bump_width_deg, steps, hw = 45, 35, 200, 30
 
    points, widths = [], []
    start_offset = -90
    for i in range(steps):
        theta_deg = start_offset + 360 * i / steps
        theta = math.radians(theta_deg)
        d = (theta_deg - bump_center_deg + 180) % 360 - 180
        bump = bump_amplitude * math.exp(-(d / (bump_width_deg / 2)) ** 2)
        r = base_r + bump
        points.append((cx + r * math.cos(theta), cy + r * math.sin(theta)))
        widths.append(hw)
 
    return points, widths

def _build_track_bramble():
    """
    Track 6 of the difficulty sweep: Bramble. Hard difficulty -- 4
    lobes with GENUINE concave necks (rose/clover curve: r(theta) =
    base_r + amplitude*cos(4*theta)), not just convex bumps on a
    circle -- matches the original sketch's distinct, narrow-waisted
    lobes. Still a single continuous function, no branching junctions
    to tangent-match. Margin +22.6px over the car's min turning radius.
    Total length ~1747px. Uses range(steps) (not steps+1) to avoid
    the loop-closure duplicate point bug found during development.
    """
    import math
 
    cx, cy, base_r, amplitude, k = 400, 300, 232, 55, 4
    steps, hw, start_offset = 280, 28, -90.0
 
    points, widths = [], []
    for i in range(steps):
        theta_deg = start_offset + 360 * i / steps
        theta = math.radians(theta_deg)
        r = base_r + amplitude * math.cos(k * theta)
        points.append((cx + r * math.cos(theta), cy + r * math.sin(theta)))
        widths.append(hw)
 
    return points, widths

def _build_track_sawtooth():
    """
    Track 7 of the difficulty sweep: Sawtooth. Hardest difficulty --
    3 peaks, 2 valleys, 5 sharp vertices in close sequence. Built as
    straight segments + fillet arcs at each vertex (generalized polygon
    fillet, each vertex's own turn angle, not assumed 90 degrees like
    Rectangle).
 
    The vertex spacing was widened from the original sketch-matched
    positions specifically to clear a fillet-collision: with R=38, the
    P1-to-V1 edge originally needed 195.3px for both endpoints' fillets
    but only had 193.1px available (a ~2px overlap) -- exactly the risk
    flagged before building this track (5 tight vertices in close
    sequence). Widening peak/valley spacing resolved it with real
    margin on every edge.
 
    Margin +15.8px over the car's min turning radius (tighter than
    every other track except Bramble/Rectangle, appropriately -- this
    is meant to be the hardest config). Total length ~1780px.
    """
    import math
 
    vertices = [
        (747.5, 457.9),  # BR -- index 0; the loop builds the edge feeding
                          # INTO each vertex, so putting BR first makes the
                          # BL->BR wrap-around edge come out as point[0],
                          # heading east -- matching reset()'s hardcoded start.
        (600.1, 78.9),   # P3
        (515.8, 284.2),  # V2
        (431.6, 78.9),   # P2
        (347.4, 284.2),  # V1
        (263.1, 78.9),   # P1
        (126.2, 489.5),  # BL -- last vertex, closes loop as edge 0
    ]
    n = len(vertices)
    hw, R, straight_steps, arc_steps = 27, 38, 25, 16
 
    fillet_starts, fillet_ends, centers, a_starts, a_ends, trims = [], [], [], [], [], []
    for i in range(n):
        p_prev = vertices[(i - 1) % n]
        p_curr = vertices[i]
        p_next = vertices[(i + 1) % n]
        v_in = (p_curr[0] - p_prev[0], p_curr[1] - p_prev[1])
        v_out = (p_next[0] - p_curr[0], p_next[1] - p_curr[1])
        len_in = math.hypot(*v_in); len_out = math.hypot(*v_out)
        u_in = (v_in[0] / len_in, v_in[1] / len_in)
        u_out = (v_out[0] / len_out, v_out[1] / len_out)
        angle_in = math.atan2(u_in[1], u_in[0])
        angle_out = math.atan2(u_out[1], u_out[0])
        turn = (angle_out - angle_in + math.pi) % (2 * math.pi) - math.pi
        interior_angle = math.pi - abs(turn)
        trim = R / math.tan(interior_angle / 2) if interior_angle > 1e-6 else R * 1000
        trims.append(trim)
 
        fs = (p_curr[0] - u_in[0] * trim, p_curr[1] - u_in[1] * trim)
        fe = (p_curr[0] + u_out[0] * trim, p_curr[1] + u_out[1] * trim)
        sign = 1 if turn > 0 else -1
        normal_in = (-u_in[1], u_in[0])
        center = (fs[0] + normal_in[0] * R * sign, fs[1] + normal_in[1] * R * sign)
        a_s = math.degrees(math.atan2(fs[1] - center[1], fs[0] - center[0]))
        a_e = math.degrees(math.atan2(fe[1] - center[1], fe[0] - center[0]))
        if sign > 0:
            while a_e < a_s: a_e += 360
            while a_e - a_s > 180: a_e -= 360
        else:
            while a_e > a_s: a_e -= 360
            while a_s - a_e > 180: a_e += 360
 
        fillet_starts.append(fs); fillet_ends.append(fe); centers.append(center)
        a_starts.append(a_s); a_ends.append(a_e)
 
    def straight(p0, p1, steps):
        return [(p0[0] + (k/steps)*(p1[0]-p0[0]), p0[1] + (k/steps)*(p1[1]-p0[1]))
                for k in range(steps + 1)]
 
    points, widths = [], []
    for i in range(n):
        prev_end = fillet_ends[(i - 1) % n]
        this_start = fillet_starts[i]
        for pt in straight(prev_end, this_start, straight_steps):
            points.append(pt); widths.append(hw)
        for j in range(arc_steps + 1):
            t = j / arc_steps
            a = math.radians(a_starts[i] + t * (a_ends[i] - a_starts[i]))
            points.append((centers[i][0] + R*math.cos(a), centers[i][1] + R*math.sin(a)))
            widths.append(hw)
 
    return points, widths

def _build_track_needle():
    """
    Track 8: Needle. Isolates the "single tight margin" mechanism --
    ONE deliberately very tight feature (margin +6.8px, tighter than
    any other track -- previous minimum was Rectangle's +12.8px), with
    everything else easy (wide base circle). Companion to Serpentine
    (track 9), which isolates the OPPOSITE mechanism: many repeated
    moderate-margin features, no single one tight. Comparing these two
    lets the sweep distinguish "hard because of raw tightness" from
    "hard because of feature repetition" instead of conflating them.
 
    Note: the static 3-point curvature check (margin +6.8px) is
    noticeably more conservative than the actual dynamic autopilot
    result (clean-lap margin +18.5px, +11.8px even under +/-3px/step
    noise) -- this gap exists on other tracks too, just more visible
    here since the margin is deliberately small.
    """
    import math
 
    cx, cy, base_r, bump_amplitude = 400, 300, 260, 130
    bump_center_deg, bump_width_deg, steps, hw, start_offset = 45, 26, 250, 24, -90
 
    points, widths = [], []
    for i in range(steps):
        theta_deg = start_offset + 360 * i / steps
        theta = math.radians(theta_deg)
        d = (theta_deg - bump_center_deg + 180) % 360 - 180
        bump = bump_amplitude * math.exp(-(d / (bump_width_deg / 2)) ** 2)
        r = base_r + bump
        points.append((cx + r * math.cos(theta), cy + r * math.sin(theta)))
        widths.append(hw)
 
    return points, widths
 
 
def _build_track_serpentine():
    """
    Track 9: Serpentine. Isolates the "repeated features" mechanism --
    7 lobes, each with MODERATE margin (+24.1px, comparable to Bramble/
    Spur, well above Rectangle/Sawtooth's ~13-16px) -- deliberately
    NOT razor-thin, so any resulting difficulty/instability is
    attributable to the sheer number of repeated precision-demanding
    decision points, not to any single one being especially tight.
    Companion to Needle (track 8), which isolates the opposite
    mechanism. Same construction as Bramble (single continuous
    r(theta) = base_r + amplitude*cos(k*theta)), just k=7 instead of
    k=4, re-tuned for this track's target margin/length.
    """
    import math
 
    cx, cy, base_r, amplitude, k = 400, 300, 246, 26, 7
    steps, hw, start_offset = 350, 26, -100.5
 
    points, widths = [], []
    for i in range(steps):
        theta_deg = start_offset + 360 * i / steps
        theta = math.radians(theta_deg)
        r = base_r + amplitude * math.cos(k * theta)
        points.append((cx + r * math.cos(theta), cy + r * math.sin(theta)))
        widths.append(hw)
 
    return points, widths
 

TRACK_REGISTRY = {
    "basra_loop": _build_track_basra_loop,
    "circle": _build_track_circle,
    "oval": _build_track_oval,
    "rectangle": _build_track_rectangle,
    "spur": _build_track_spur,
    "bramble": _build_track_bramble,
    "sawtooth": _build_track_sawtooth,
    "needle": _build_track_needle,
    "serpentine": _build_track_serpentine,
}

def _compute_track_geometry(raw_points, raw_widths):
    """
    Deduplicate, compute per-point normals, build left/right boundaries.

    Returns
    -------
    centerline    : np.ndarray (N,2)
    half_widths   : np.ndarray (N,)
    left_boundary : np.ndarray (N,2)
    right_boundary: np.ndarray (N,2)
    arc_lengths   : np.ndarray (N,)
    total_length  : float
    """
    pts = np.array(raw_points, dtype=np.float32)
    hws = np.array(raw_widths,  dtype=np.float32)

    # Deduplicate consecutive near-identical points
    keep = [0]
    for i in range(1, len(pts)):
        if np.linalg.norm(pts[i] - pts[keep[-1]]) > 0.5:
            keep.append(i)
    pts = pts[keep]
    hws = hws[keep]
    N   = len(pts)

    left_b  = np.zeros_like(pts)
    right_b = np.zeros_like(pts)

    for i in range(N):
        prev    = pts[(i - 1) % N]
        nxt     = pts[(i + 1) % N]
        tangent = nxt - prev
        length  = np.linalg.norm(tangent)
        tangent = tangent / length if length > 1e-6 else np.array([1.0, 0.0])
        # Left normal (screen coords, y down): rotate tangent 90° CCW = (-dy, dx)
        normal      = np.array([-tangent[1], tangent[0]])
        left_b[i]   = pts[i] + normal * hws[i]
        right_b[i]  = pts[i] - normal * hws[i]

    # Cumulative arc length
    arc = np.zeros(N, dtype=np.float32)
    for i in range(1, N):
        arc[i] = arc[i - 1] + np.linalg.norm(pts[i] - pts[i - 1])
    total = float(arc[-1] + np.linalg.norm(pts[0] - pts[-1]))

    return pts, hws, left_b, right_b, arc, total


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

class RaceEnv(gym.Env):
    """
    IARL Race Environment.

    Observation space : Box(BUMPER_CAM_H, BUMPER_CAM_W, 3, uint8)
    Action space      : Discrete(5)
        0 turn_left | 1 turn_right | 2 accelerate | 3 brake | 4 do_nothing

    Info dict keys (every step):
        top_down        : np.ndarray (TOP_DOWN_H, TOP_DOWN_W, 3) uint8
        speed           : float
        heading         : float  radians
        checkpoint      : int    index of next checkpoint to pass
        lap             : int    laps completed
        track_progress  : float  0..1 fraction of current lap completed
    """

    metadata = {"render_modes": ["human"], "render_fps": FPS}

    def __init__(self, render_mode=None, render_top_down=True, render_front_camera=True,
                 track_name="basra_loop"):
        super().__init__()
        self.render_mode = render_mode
        self._render_top_down_enabled = render_top_down
        self._render_front_camera_enabled = render_front_camera
        self.track_name = track_name
 
        # Build track geometry
        if track_name not in TRACK_REGISTRY:
            raise ValueError(
                f"Unknown track_name '{track_name}'. Available: {list(TRACK_REGISTRY.keys())}"
            )
        raw_pts, raw_hws = TRACK_REGISTRY[track_name]()
        (self.centerline,
         self.half_widths,
         self.left_boundary,
         self.right_boundary,
         self.arc_lengths,
         self.total_length) = _compute_track_geometry(raw_pts, raw_hws)

        self.N = len(self.centerline)

        # 10 checkpoints at equal arc-length intervals
        self.checkpoint_indices = []
        for i in range(10):
            target = self.total_length * i / 10
            self.checkpoint_indices.append(
                int(np.argmin(np.abs(self.arc_lengths - target)))
            )

        # Gymnasium spaces
        self.observation_space = spaces.Box(
            low=0, high=255,
            shape=(BUMPER_CAM_H, BUMPER_CAM_W, 3),
            dtype=np.uint8
        )
        self.action_space = spaces.Discrete(5)

        # Pygame (lazy init)
        self._screen = None
        self._clock  = None

        # Car state (populated in reset)
        self._pos     = None
        self._heading = None
        self._speed   = None

        # Episode state
        self._next_checkpoint = None
        self._laps_completed  = None
        self._steps           = None
        self._closest_idx     = None

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._pos             = self.centerline[0].copy().astype(np.float32)
        self._heading         = 0.0          # pointing right along bottom straight
        self._speed           = START_SPEED
        self._next_checkpoint = 0
        self._laps_completed  = 0
        self._steps           = 0
        self._closest_idx     = 0

        obs  = self._get_observation()
        info = self._build_info()
        return obs, info

    def step(self, action):
        self._steps += 1

        # 1. Apply action → updates heading and speed
        self._apply_action(action)

        # 2. Move car
        self._pos[0] += math.cos(self._heading) * self._speed
        self._pos[1] += math.sin(self._heading) * self._speed

        # 3. Update nearest centerline index
        self._closest_idx = self._find_closest_idx(self._pos)

        # 4. Collision check
        reward, terminated = self._check_collision()

        # 5. Checkpoint / lap (only if still alive)
        if not terminated:
            cp_r, lap_r = self._check_checkpoints()
            reward += cp_r + lap_r
            if self._laps_completed >= 1:
                terminated = True

        # 6. Time penalty
        reward += TIME_PENALTY

        obs  = self._get_observation()
        info = self._build_info()
        return obs, reward, terminated, False, info

    def render(self):
        if self.render_mode != "human":
            return
        self._ensure_pygame()
        surf = self._render_top_down_surface(w=SCREEN_W, h=SCREEN_H)
        self._screen.blit(surf, (0, 0))
        pygame.display.flip()
        self._clock.tick(FPS)

    def close(self):
        if self._screen is not None:
            pygame.quit()
            self._screen = None

    # ------------------------------------------------------------------
    # Physics
    # ------------------------------------------------------------------

    def _apply_action(self, action):
        # Turn rate degrades with speed
        turn_rate = max(0.01, BASE_TURN_RATE - TURN_SPEED_DECAY * self._speed)

        if action == TURN_LEFT:
            self._heading -= turn_rate
        elif action == TURN_RIGHT:
            self._heading += turn_rate
        elif action == ACCELERATE:
            self._speed = min(MAX_SPEED, self._speed + ACCELERATION_STEP)
        elif action == BRAKE:
            self._speed = max(MIN_SPEED, self._speed - BRAKE_STEP)
        # DO_NOTHING: no change before resistance

        # Rolling resistance every step
        self._speed = max(MIN_SPEED, self._speed - ROLLING_RESIST)

        # Wrap heading to [-π, π]
        self._heading = (self._heading + math.pi) % (2 * math.pi) - math.pi

    # ------------------------------------------------------------------
    # Collision
    # ------------------------------------------------------------------

    def _check_collision(self):
        idx    = self._closest_idx
        center = self.centerline[idx]
        hw     = self.half_widths[idx]
        dist   = float(np.linalg.norm(self._pos - center))

        if dist > hw + GRAZE_DISTANCE:
            return COLLISION_PENALTY, True
        elif dist > hw - GRAZE_DISTANCE:
            return GRAZE_PENALTY, False
        return 0.0, False

    # ------------------------------------------------------------------
    # Checkpoints and laps
    # ------------------------------------------------------------------

    def _check_checkpoints(self):
        # Index-distance check instead of raw Euclidean distance to the
        # checkpoint's position. _closest_idx is now constrained to a
        # local window each step (see _find_closest_idx), so it can only
        # reach a checkpoint's index by genuinely traveling the arc-length
        # path there -- it can no longer trigger early via spatial
        # proximity across a loop (e.g. a checkpoint sitting inside the
        # hairpin being reachable in a straight line from the approach
        # corridor before the car has actually driven the turn).
        cp_idx = self.checkpoint_indices[self._next_checkpoint]
        N = self.N
        raw_dist = abs(self._closest_idx - cp_idx)
        idx_dist = min(raw_dist, N - raw_dist)  # wraparound-aware
 
        if idx_dist <= 3:
            self._next_checkpoint += 1
            if self._next_checkpoint >= len(self.checkpoint_indices):
                self._next_checkpoint = 0
                self._laps_completed  += 1
                return CHECKPOINT_REWARD, LAP_REWARD
            return CHECKPOINT_REWARD, 0.0
        return 0.0, 0.0

    # ------------------------------------------------------------------
    # Geometry helpers
    # ------------------------------------------------------------------

    def _find_closest_idx(self, pos, max_steps=3):
        # Incremental hill-climbing instead of a windowed/global search.
        # Only compares against immediate neighbors (idx-1, idx+1) and
        # takes one step at a time, up to max_steps per call. This can
        # only reach a distant index by continuously improving through
        # every point in between -- it cannot skip across a loop the way
        # any fixed-size window search can (verified: window search still
        # jumped near the hairpin even at window=5; hill-climbing does
        # not, across the same stress tests). max_steps=3 comfortably
        # covers real per-step movement (observed max: 1 index at
        # realistic speeds) with margin to spare.
        N = self.N
        idx = self._closest_idx if self._closest_idx is not None else 0
        cur_dist = np.linalg.norm(self.centerline[idx] - pos)
        for _ in range(max_steps):
            idx_fwd = (idx + 1) % N
            idx_bwd = (idx - 1) % N
            d_fwd = np.linalg.norm(self.centerline[idx_fwd] - pos)
            d_bwd = np.linalg.norm(self.centerline[idx_bwd] - pos)
            if d_fwd <= cur_dist and d_fwd <= d_bwd:
                idx, cur_dist = idx_fwd, d_fwd
            elif d_bwd < cur_dist:
                idx, cur_dist = idx_bwd, d_bwd
            else:
                break  # neither neighbor improves -- converged
        return idx

    def _track_progress(self):
        return float(self.arc_lengths[self._closest_idx] / self.total_length)

    # ------------------------------------------------------------------
    # Rendering — top-down
    # ------------------------------------------------------------------

    def _render_top_down_surface(self, w=SCREEN_W, h=SCREEN_H):
        """
        Render the track as a series of per-segment quads rather than
        a single large polygon. This eliminates self-intersection artifacts
        that occur when the boundary polygon winds back on itself.
        """
        surf = pygame.Surface((w, h))
        surf.fill(COL_GRASS)

        sx = w / SCREEN_W
        sy = h / SCREEN_H

        def sp(pt):
            return (int(pt[0] * sx), int(pt[1] * sy))

        # Draw track as consecutive quads between adjacent centerline points
        for i in range(self.N - 1):
            quad = [
                sp(self.left_boundary[i]),
                sp(self.left_boundary[i + 1]),
                sp(self.right_boundary[i + 1]),
                sp(self.right_boundary[i]),
            ]
            pygame.draw.polygon(surf, COL_TRACK, quad)

        # Close the loop: last point back to first
        quad = [
            sp(self.left_boundary[-1]),
            sp(self.left_boundary[0]),
            sp(self.right_boundary[0]),
            sp(self.right_boundary[-1]),
        ]
        pygame.draw.polygon(surf, COL_TRACK, quad)

        # Centerline dashes
        for i in range(0, self.N - 1, 4):
            pygame.draw.line(surf, COL_CENTERLINE,
                             sp(self.centerline[i]),
                             sp(self.centerline[i + 1]), 1)

        # All checkpoint lines (cyan)
        for cp_i in self.checkpoint_indices:
            pygame.draw.line(surf, COL_CHECKPOINT,
                             sp(self.left_boundary[cp_i]),
                             sp(self.right_boundary[cp_i]), 2)

        # Next checkpoint highlighted yellow
        if self._next_checkpoint is not None:
            ncp = self.checkpoint_indices[self._next_checkpoint]
            pygame.draw.line(surf, (255, 255, 0),
                             sp(self.left_boundary[ncp]),
                             sp(self.right_boundary[ncp]), 3)

        # Start line white
        sl_i = self.checkpoint_indices[0]
        pygame.draw.line(surf, COL_STARTLINE,
                         sp(self.left_boundary[sl_i]),
                         sp(self.right_boundary[sl_i]), 3)

        # Car: red dot + white heading arrow
        if self._pos is not None:
            cx, cy = sp(self._pos)
            r      = max(5, int(7 * sx))
            pygame.draw.circle(surf, COL_CAR, (cx, cy), r)
            hx = cx + int(r * 2.0 * math.cos(self._heading))
            hy = cy + int(r * 2.0 * math.sin(self._heading))
            pygame.draw.line(surf, (255, 255, 255), (cx, cy), (hx, hy), 2)

        return surf

    def _render_top_down_array(self):
        self._ensure_pygame()
        surf = self._render_top_down_surface(w=TOP_DOWN_W, h=TOP_DOWN_H)
        return pygame.surfarray.array3d(surf).transpose(1, 0, 2)

# ------------------------------------------------------------------
    # Rendering — bumper cam (ray-cast, vectorized)
    # ------------------------------------------------------------------
    #
    # Same semantics as the original per-ray Python loop: for each of
    # BUMPER_CAM_W columns, march outward in step_sz increments up to
    # BUMPER_CAM_DIST, find the first point that falls outside the track
    # boundary (nearest centerline point's half_width), and use that
    # distance to draw sky/wall/floor for that column.
    #
    # The only change is *how* "nearest centerline point" is computed for
    # every sample point: instead of calling _find_closest_idx (a full
    # O(N) brute-force search) once per sample point in a Python loop
    # (~9,600 calls/frame), all sample points across all rays are batched
    # into a single vectorized numpy computation. Output is numerically
    # identical to the original; only the internal computation path changed.
    def _get_observation(self):
        if self._render_front_camera_enabled:
            return self._render_bumper_cam()
        # Cheap placeholder matching observation_space shape/dtype. Content
        # is never read by anything -- Arm 2/3 wrappers replace this with
        # their own structured-state observation entirely.
        return np.zeros((BUMPER_CAM_H, BUMPER_CAM_W, 3), dtype=np.uint8)
    
    def _render_bumper_cam(self):
        self._ensure_pygame()

        img       = np.zeros((BUMPER_CAM_H, BUMPER_CAM_W, 3), dtype=np.uint8)
        half_fov  = math.radians(BUMPER_CAM_FOV / 2)
        step_sz   = 2.0
        max_steps = int(BUMPER_CAM_DIST / step_sz)

        # --- Ray angles, one per column ---
        cols   = np.arange(BUMPER_CAM_W)
        fracs  = cols / (BUMPER_CAM_W - 1)
        angles = self._heading - half_fov + fracs * 2 * half_fov      # (W,)

        # --- Sample distances along each ray ---
        steps           = np.arange(1, max_steps + 1, dtype=np.float64)  # (S,)
        dists_along_ray = steps * step_sz                                # (S,)

        cos_a = np.cos(angles)   # (W,)
        sin_a = np.sin(angles)   # (W,)

        # sample_pts[col, step] = pos + d * (cos(angle_col), sin(angle_col))
        sample_x = self._pos[0] + dists_along_ray[None, :] * cos_a[:, None]   # (W, S)
        sample_y = self._pos[1] + dists_along_ray[None, :] * sin_a[:, None]   # (W, S)

        W = BUMPER_CAM_W
        S = max_steps
        flat_pts = np.stack([sample_x.ravel(), sample_y.ravel()], axis=1)     # (W*S, 2)

        # --- Batched nearest-centerline-point lookup for ALL sample points at once ---
        # Equivalent to calling _find_closest_idx once per point, but done as
        # a single vectorized computation instead of W*S separate Python calls.
        c = self.centerline                                                    # (N, 2)
        pt_sq = np.sum(flat_pts * flat_pts, axis=1, keepdims=True)             # (M, 1)
        c_sq  = np.sum(c * c, axis=1)[None, :]                                 # (1, N)
        cross = flat_pts @ c.T                                                 # (M, N)
        dist2 = pt_sq - 2.0 * cross + c_sq                                     # (M, N)
        dist2 = np.maximum(dist2, 0.0)  # guard tiny negative values from fp error

        closest_idx    = np.argmin(dist2, axis=1)                             # (M,)
        dist_to_center = np.sqrt(dist2[np.arange(dist2.shape[0]), closest_idx])  # (M,)
        hw_at_closest   = self.half_widths[closest_idx]                       # (M,)

        outside = (dist_to_center > hw_at_closest).reshape(W, S)              # (W, S)

        # First step index per column where the ray leaves the track boundary.
        # np.argmax returns 0 if no True is present, so any_hit disambiguates
        # "hit at step 0" from "never hit within max_steps".
        any_hit        = outside.any(axis=1)                                   # (W,)
        first_hit_step = np.argmax(outside, axis=1)                            # (W,)
        hit_dist = np.where(any_hit, dists_along_ray[first_hit_step], BUMPER_CAM_DIST)  # (W,)

        # --- Draw columns (same per-column drawing logic as before) ---
        for col in range(W):
            d = float(hit_dist[col])
            hit_colour = COL_WALL_NEAR if (any_hit[col] and d < 30) else COL_GRASS

            wall_h = min(int(BUMPER_CAM_H * 40 / max(d, 1)), BUMPER_CAM_H)
            sky_h  = (BUMPER_CAM_H - wall_h) // 2

            img[:sky_h, col]               = (70, 130, 180)   # sky
            img[sky_h:sky_h + wall_h, col] = hit_colour        # wall/boundary
            img[sky_h + wall_h:, col]      = COL_TRACK         # track floor

        return img

   # ------------------------------------------------------------------
    # Info dict
    # ------------------------------------------------------------------

    def _build_info(self):
        info = {
            "top_down"       : None,
            "speed"          : float(self._speed),
            "heading"        : float(self._heading),
            "checkpoint"     : self._next_checkpoint,
            "lap"            : self._laps_completed,
            "track_progress" : self._track_progress(),
        }
        if self._render_top_down_enabled:
            info["top_down"] = self._render_top_down_array()
        return info

    # ------------------------------------------------------------------
    # Pygame helpers
    # ------------------------------------------------------------------

    def _ensure_pygame(self):
        if self._screen is None:
            pygame.init()
            if self.render_mode == "human":
                self._screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
                pygame.display.set_caption("IARL Race Environment")
            else:
                self._screen = pygame.Surface((SCREEN_W, SCREEN_H))
            self._clock = pygame.time.Clock()