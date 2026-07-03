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

def _build_track():
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

    # S3: Right connector — short vertical, car heading upward
    # Verified: start=(730,310), end=(730,200). Gap to S2 end: 0.00px.
    straight(730, 310, 730, 200, steps=8, hw=30)

    # S4: Hairpin — tight, narrow (hw=20, full width 40px vs 76px on straight)
    # Center (650,200), r=80. Arc 0°->180°.
    # Verified: start=(730,200), end=(570,200). Gap to S3 end: 0.00px.
    # Car enters heading upward, exits heading downward-left.
    arc(cx=650, cy=200, r=80, a_start_deg=0, a_end_deg=180, steps=28, hw=20)

    # S5: Top straight — car heading left
    # Verified: start=(570,200). Gap to S4 end: 0.00px.
    straight(570, 200, 200, 200, steps=24, hw=30)

    # S6: Left sweeper arc
    # Center (200,310), r=110. Arc 270°->180°.
    # Verified: start=(200,200), end=(90,310). Gap to S5 end: 0.00px.
    arc(cx=200, cy=310, r=110, a_start_deg=270, a_end_deg=180, steps=16, hw=32)

    # S7: Left descent — car heading downward, closing loop to S1
    # Verified: start=(90,310), end=(120,490). Gap to S6 end: 0.00px.
    # Loop closure: S7 end=(120,490) == S1 start=(120,490). Gap: 0.00px.
    straight(90, 310, 120, 490, steps=14, hw=34)

    return points, widths


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

    def __init__(self, render_mode=None):
        super().__init__()
        self.render_mode = render_mode

        # Build track geometry
        raw_pts, raw_hws = _build_track()
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

        obs  = self._render_bumper_cam()
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

        obs  = self._render_bumper_cam()
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
        cp_idx     = self.checkpoint_indices[self._next_checkpoint]
        dist_to_cp = float(np.linalg.norm(self._pos - self.centerline[cp_idx]))

        if dist_to_cp < self.half_widths[cp_idx] * 1.5:
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

    def _find_closest_idx(self, pos):
        dists = np.linalg.norm(self.centerline - pos, axis=1)
        return int(np.argmin(dists))

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
    # Rendering — bumper cam (ray-cast)
    # ------------------------------------------------------------------

    def _render_bumper_cam(self):
        self._ensure_pygame()

        img       = np.zeros((BUMPER_CAM_H, BUMPER_CAM_W, 3), dtype=np.uint8)
        half_fov  = math.radians(BUMPER_CAM_FOV / 2)
        step_sz   = 2.0
        max_steps = int(BUMPER_CAM_DIST / step_sz)

        for col in range(BUMPER_CAM_W):
            frac  = col / (BUMPER_CAM_W - 1)
            angle = self._heading - half_fov + frac * 2 * half_fov

            hit_dist   = BUMPER_CAM_DIST
            hit_colour = COL_GRASS

            for s in range(1, max_steps + 1):
                d    = s * step_sz
                rpos = np.array([self._pos[0] + d * math.cos(angle),
                                 self._pos[1] + d * math.sin(angle)])
                ci   = self._find_closest_idx(rpos)
                dist = float(np.linalg.norm(rpos - self.centerline[ci]))

                if dist > self.half_widths[ci]:
                    hit_dist   = d
                    hit_colour = COL_WALL_NEAR if d < 30 else COL_GRASS
                    break

            wall_h = min(int(BUMPER_CAM_H * 40 / max(hit_dist, 1)),
                         BUMPER_CAM_H)
            sky_h  = (BUMPER_CAM_H - wall_h) // 2

            img[:sky_h, col]               = (70, 130, 180)   # sky
            img[sky_h:sky_h + wall_h, col] = hit_colour        # wall/boundary
            img[sky_h + wall_h:, col]      = COL_TRACK         # track floor

        return img

    # ------------------------------------------------------------------
    # Info dict
    # ------------------------------------------------------------------

    def _build_info(self):
        return {
            "top_down"       : self._render_top_down_array(),
            "speed"          : float(self._speed),
            "heading"        : float(self._heading),
            "checkpoint"     : self._next_checkpoint,
            "lap"            : self._laps_completed,
            "track_progress" : self._track_progress(),
        }

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