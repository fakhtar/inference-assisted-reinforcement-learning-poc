"""
IARL Race Environment
=====================
A custom Gymnasium environment representing a 2D top-down race track.
Used as the shared environment for all three experimental arms of the
Inference Assisted Reinforcement Learning (IARL) experiment.

Observation (default): front_camera RGB array (bumper-cam, no car in frame)
Additional observation:  top_down RGB array (full track, car position marked)
Action space: Discrete(5) — turn_left, turn_right, accelerate, brake, do_nothing
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
MIN_SPEED         = 0.5    # car never fully stops (keeps momentum)
ACCELERATION_STEP = 0.5    # speed gained per ACCELERATE action
BRAKE_STEP        = 1.0    # speed lost per BRAKE action
BASE_TURN_RATE    = 0.12   # radians per timestep at zero speed
TURN_SPEED_DECAY  = 0.015  # turn rate reduction per unit of speed
                            # effective_turn = BASE - TURN_SPEED_DECAY * speed

# Collision
GRAZE_DISTANCE    = 4      # pixels inside boundary = graze penalty
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
BUMPER_CAM_W      = 160    # front_camera output width  (pixels)
BUMPER_CAM_H      = 120    # front_camera output height (pixels)
BUMPER_CAM_DIST   = 120    # how far ahead the camera sees (world units)
BUMPER_CAM_FOV    = 90     # horizontal field of view (degrees)
TOP_DOWN_W        = 160    # top_down output width
TOP_DOWN_H        = 120    # top_down output height

# Colours
COL_BG            = (30,  30,  30)
COL_TRACK         = (80,  80,  80)
COL_GRASS         = (34,  85,  34)
COL_CENTERLINE    = (200, 200,  50)
COL_CAR           = (220,  50,  50)
COL_CHECKPOINT    = (50,  200, 200)
COL_STARTLINE     = (255, 255, 255)
COL_WALL_NEAR     = (220,  80,  80)   # bumper cam wall tint when close


# ---------------------------------------------------------------------------
# Track definition
# ---------------------------------------------------------------------------

def _build_track():
    """
    Build the Basra Loop circuit.

    Returns
    -------
    centerline : list of (x, y) floats
        Ordered points forming the closed centerline loop.
    half_widths : list of floats
        Half-width of the track at each centerline point.
        Full track width = 2 * half_width.
    """
    # We construct the track as a series of arc/line segments defined by
    # (cx, cy, radius, start_angle_deg, end_angle_deg, steps, half_width)
    # for curves, and
    # (x0,y0, x1,y1, steps, half_width)
    # for straights.
    # All angles: 0 = right (+x), 90 = up (-y) in screen coords.
    # We accumulate points and widths.

    points = []
    widths = []

    def arc(cx, cy, r, a_start, a_end, steps, hw):
        """Append an arc segment (screen coords: y increases downward)."""
        for i in range(steps + 1):
            t = i / steps
            a = math.radians(a_start + t * (a_end - a_start))
            x = cx + r * math.cos(a)
            y = cy + r * math.sin(a)
            points.append((x, y))
            widths.append(hw)

    def straight(x0, y0, x1, y1, steps, hw):
        """Append a straight segment."""
        for i in range(steps + 1):
            t = i / steps
            x = x0 + t * (x1 - x0)
            y = y0 + t * (y1 - y0)
            points.append((x, y))
            widths.append(hw)

    # ------------------------------------------------------------------
    # Basra Loop — counterclockwise traversal
    # Screen origin top-left. Track fits inside 800x600 with margins.
    #
    # Segment layout (approximate world coords):
    #
    #   [5] Descent straight   [4] Top left sweeper (medium left)
    #       left side down  ←──────────────────────────────
    #       |                                              |
    #       |  [6] Start/finish                    [3] Short connector
    #       |                                              |
    #       └──────────────────────────────────────────── [2] Hairpin (top-right, tight)
    #   [1] Long straight (bottom, left→right)    [2b] exits downward
    #   start here →
    #
    # Exact coordinates chosen so the track is visually clear at 800x600.
    # ------------------------------------------------------------------

    # Segment 1: Long straight — bottom of screen, left to right
    # Wide track (half_width=38). Car starts here heading right (+x).
    # From (100, 480) to (580, 480)
    straight(100, 480, 580, 480, steps=30, hw=38)

    # Segment 2: Bottom-right medium sweeper — curves upward (arc going from
    # 90° to 0° in standard math, which in screen coords is a right-hand
    # turn curving upward). Center at (580, 340), radius 140.
    # Enters from bottom (angle 90° screen = pointing down = 90 in pygame),
    # exits pointing upward (angle 270° screen).
    # In pygame coords (y down): center (620, 480), arc from 180° to 270°
    arc(cx=620, cy=340, r=140, a_start=90, a_end=180, steps=20, hw=34)

    # Segment 3: Short connector — right side, travelling upward
    # From approx (620, 200) to (620, 130)
    straight(620, 200, 590, 130, steps=8, hw=30)

    # Segment 4: Hairpin — top right, tight turn. Narrow track.
    # Car arrives from below heading upward, hairpin turns it back downward-left.
    # Center at (530, 110), radius 60. Arc from 0° to 180° (screen coords).
    # Narrow: half_width=20 (full width 40px vs 76px on the straight).
    arc(cx=530, cy=110, r=60, a_start=0, a_end=180, steps=24, hw=20)

    # Segment 5: Top sweeper — top of screen, right to left (medium radius).
    # Car exits hairpin heading downward-left, sweeper carries it leftward.
    # Center at (310, 200), radius 110. Arc from 330° to 210° (screen coords).
    arc(cx=320, cy=160, r=130, a_start=330, a_end=210, steps=28, hw=30)

    # Segment 6: Descent straight — left side, travelling downward
    # From approx (190, 200) to (100, 480)
    straight(190, 260, 100, 480, steps=20, hw=32)

    return points, widths


def _compute_track_geometry(raw_points, raw_widths):
    """
    Deduplicate consecutive identical points, compute cumulative arc length,
    and build left/right boundary point lists.

    Returns
    -------
    centerline : np.ndarray (N, 2)
    half_widths : np.ndarray (N,)
    left_boundary : np.ndarray (N, 2)
    right_boundary : np.ndarray (N, 2)
    arc_lengths : np.ndarray (N,)   cumulative distance along centerline
    total_length : float
    """
    pts = np.array(raw_points, dtype=np.float32)
    hws = np.array(raw_widths, dtype=np.float32)

    # Deduplicate
    keep = [0]
    for i in range(1, len(pts)):
        if np.linalg.norm(pts[i] - pts[keep[-1]]) > 0.5:
            keep.append(i)
    pts = pts[keep]
    hws = hws[keep]

    N = len(pts)

    # Normals (perpendicular to tangent, pointing left of travel direction)
    left_b  = np.zeros_like(pts)
    right_b = np.zeros_like(pts)

    for i in range(N):
        prev = pts[(i - 1) % N]
        nxt  = pts[(i + 1) % N]
        tangent = nxt - prev
        length  = np.linalg.norm(tangent)
        if length < 1e-6:
            tangent = np.array([1.0, 0.0])
        else:
            tangent /= length
        # Normal: rotate tangent 90° left (in screen coords: (-dy, dx))
        normal = np.array([-tangent[1], tangent[0]])
        left_b[i]  = pts[i] + normal * hws[i]
        right_b[i] = pts[i] - normal * hws[i]

    # Arc lengths
    arc = np.zeros(N, dtype=np.float32)
    for i in range(1, N):
        arc[i] = arc[i-1] + np.linalg.norm(pts[i] - pts[i-1])
    total = arc[-1] + np.linalg.norm(pts[0] - pts[-1])

    return pts, hws, left_b, right_b, arc, total


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

class RaceEnv(gym.Env):
    """
    IARL Race Environment.

    Observation space : Box(BUMPER_CAM_H, BUMPER_CAM_W, 3, uint8)
        front_camera bumper-cam RGB image (default observation).

    Action space : Discrete(5)
        0 turn_left | 1 turn_right | 2 accelerate | 3 brake | 4 do_nothing

    Info dict keys (every step):
        top_down        : np.ndarray (TOP_DOWN_H, TOP_DOWN_W, 3) uint8
        speed           : float
        heading         : float  radians
        checkpoint      : int    index of last passed checkpoint
        lap             : int    laps completed
        track_progress  : float  0..1 fraction of current lap completed
    """

    metadata = {"render_modes": ["human"], "render_fps": FPS}

    def __init__(self, render_mode=None):
        super().__init__()

        self.render_mode = render_mode

        # Build track
        raw_pts, raw_hws = _build_track()
        (self.centerline,
         self.half_widths,
         self.left_boundary,
         self.right_boundary,
         self.arc_lengths,
         self.total_length) = _compute_track_geometry(raw_pts, raw_hws)

        self.N = len(self.centerline)  # number of centerline points

        # Checkpoints: evenly spaced by arc length (every ~10% of lap)
        n_checkpoints = 10
        cp_arc_targets = [self.total_length * i / n_checkpoints
                          for i in range(n_checkpoints)]
        self.checkpoint_indices = []
        for target in cp_arc_targets:
            idx = int(np.argmin(np.abs(self.arc_lengths - target)))
            self.checkpoint_indices.append(idx)

        # Gymnasium spaces
        self.observation_space = spaces.Box(
            low=0, high=255,
            shape=(BUMPER_CAM_H, BUMPER_CAM_W, 3),
            dtype=np.uint8
        )
        self.action_space = spaces.Discrete(5)

        # Pygame state (lazy init)
        self._screen      = None
        self._clock       = None
        self._font        = None
        self._surf_track  = None   # pre-rendered track surface

        # Car state (set in reset)
        self._pos     = None   # np.array [x, y]
        self._heading = None   # radians; 0 = right, π/2 = down (screen)
        self._speed   = None

        # Episode state
        self._next_checkpoint = None
        self._laps_completed  = None
        self._steps           = None
        self._closest_idx     = None   # index on centerline nearest to car

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        # Place car at the start of the centerline (index 0), heading right
        start = self.centerline[0].copy()
        self._pos     = start.astype(np.float32)
        self._heading = 0.0   # pointing right (+x), along the straight
        self._speed   = MIN_SPEED

        self._next_checkpoint = 0
        self._laps_completed  = 0
        self._steps           = 0
        self._closest_idx     = 0

        obs  = self._render_bumper_cam()
        info = self._build_info()
        return obs, info

    def step(self, action):
        self._steps += 1

        # 1. Apply action
        self._apply_action(action)

        # 2. Move car
        dx = math.cos(self._heading) * self._speed
        dy = math.sin(self._heading) * self._speed
        self._pos[0] += dx
        self._pos[1] += dy

        # 3. Update closest centerline index (used for width lookup + progress)
        self._closest_idx = self._find_closest_idx(self._pos)

        # 4. Collision detection
        reward, terminated = self._check_collision()

        # 5. Checkpoint / lap logic (only if not already terminated)
        if not terminated:
            cp_reward, lap_reward = self._check_checkpoints()
            reward += cp_reward + lap_reward
            if self._laps_completed >= 1:
                terminated = True

        # 6. Time penalty
        reward += TIME_PENALTY

        truncated = False
        obs  = self._render_bumper_cam()
        info = self._build_info()

        return obs, reward, terminated, truncated, info

    def render(self):
        """Display top-down view in a pygame window."""
        if self.render_mode != "human":
            return
        self._ensure_pygame()
        surf = self._render_top_down_surface(scale=1.0,
                                              w=SCREEN_W, h=SCREEN_H)
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
        turn_rate = max(0.01,
                        BASE_TURN_RATE - TURN_SPEED_DECAY * self._speed)

        if action == TURN_LEFT:
            self._heading -= turn_rate
        elif action == TURN_RIGHT:
            self._heading += turn_rate
        elif action == ACCELERATE:
            self._speed = min(MAX_SPEED, self._speed + ACCELERATION_STEP)
        elif action == BRAKE:
            self._speed = max(MIN_SPEED, self._speed - BRAKE_STEP)
        # DO_NOTHING: no change

        # Speed bleeds off slightly every step (rolling resistance)
        self._speed = max(MIN_SPEED, self._speed - 0.05)

        # Wrap heading to [-π, π]
        self._heading = (self._heading + math.pi) % (2 * math.pi) - math.pi

    # ------------------------------------------------------------------
    # Collision
    # ------------------------------------------------------------------

    def _check_collision(self):
        """
        Returns (reward_delta, terminated).
        Uses the half-width at the nearest centerline point to define
        the track boundary at the car's current location.
        """
        idx = self._closest_idx
        center = self.centerline[idx]
        hw     = self.half_widths[idx]

        dist = np.linalg.norm(self._pos - center)

        if dist > hw + GRAZE_DISTANCE:
            # Full breach — episode ends
            return COLLISION_PENALTY, True
        elif dist > hw - GRAZE_DISTANCE:
            # Graze — penalty, episode continues
            return GRAZE_PENALTY, False
        else:
            return 0.0, False

    # ------------------------------------------------------------------
    # Checkpoints and laps
    # ------------------------------------------------------------------

    def _check_checkpoints(self):
        cp_reward  = 0.0
        lap_reward = 0.0

        cp_idx = self.checkpoint_indices[self._next_checkpoint]
        dist_to_cp = np.linalg.norm(
            self._pos - self.centerline[cp_idx]
        )

        if dist_to_cp < self.half_widths[cp_idx] * 1.5:
            cp_reward = CHECKPOINT_REWARD
            self._next_checkpoint += 1

            if self._next_checkpoint >= len(self.checkpoint_indices):
                # Completed all checkpoints = one lap
                self._next_checkpoint = 0
                self._laps_completed  += 1
                lap_reward = LAP_REWARD

        return cp_reward, lap_reward

    # ------------------------------------------------------------------
    # Geometry helpers
    # ------------------------------------------------------------------

    def _find_closest_idx(self, pos):
        """Return index of centerline point nearest to pos."""
        dists = np.linalg.norm(self.centerline - pos, axis=1)
        return int(np.argmin(dists))

    def _track_progress(self):
        """Fraction 0..1 of current lap completed."""
        return self.arc_lengths[self._closest_idx] / self.total_length

    # ------------------------------------------------------------------
    # Rendering — top-down
    # ------------------------------------------------------------------

    def _render_top_down_surface(self, scale=1.0, w=SCREEN_W, h=SCREEN_H):
        """
        Render a full top-down view of the track with car position.
        Returns a pygame Surface of size (w, h).
        """
        surf = pygame.Surface((w, h))
        surf.fill(COL_GRASS)

        sx = w / SCREEN_W
        sy = h / SCREEN_H

        def sp(pt):
            """Scale a world point to surface coords."""
            return (int(pt[0] * sx), int(pt[1] * sy))

        # Draw filled track polygon (left boundary + reversed right boundary)
        left_pts  = [sp(p) for p in self.left_boundary]
        right_pts = [sp(p) for p in self.right_boundary]
        track_poly = left_pts + list(reversed(right_pts))
        if len(track_poly) >= 3:
            pygame.draw.polygon(surf, COL_TRACK, track_poly)

        # Draw centerline dashes
        for i in range(0, self.N - 1, 4):
            pygame.draw.line(surf, COL_CENTERLINE,
                             sp(self.centerline[i]),
                             sp(self.centerline[i+1]), 1)

        # Draw checkpoints
        for cp_i in self.checkpoint_indices:
            lp = sp(self.left_boundary[cp_i])
            rp = sp(self.right_boundary[cp_i])
            pygame.draw.line(surf, COL_CHECKPOINT, lp, rp, 2)

        # Highlight next checkpoint
        if self._next_checkpoint is not None:
            ncp = self.checkpoint_indices[self._next_checkpoint]
            lp  = sp(self.left_boundary[ncp])
            rp  = sp(self.right_boundary[ncp])
            pygame.draw.line(surf, (255, 255, 0), lp, rp, 3)

        # Draw start line
        sl_i = self.checkpoint_indices[0]
        pygame.draw.line(surf, COL_STARTLINE,
                         sp(self.left_boundary[sl_i]),
                         sp(self.right_boundary[sl_i]), 3)

        # Draw car
        if self._pos is not None:
            cx, cy = sp(self._pos)
            car_r  = max(4, int(6 * sx))
            pygame.draw.circle(surf, COL_CAR, (cx, cy), car_r)
            # Heading indicator
            hx = cx + int(car_r * 1.8 * math.cos(self._heading))
            hy = cy + int(car_r * 1.8 * math.sin(self._heading))
            pygame.draw.line(surf, (255, 255, 255), (cx, cy), (hx, hy), 2)

        return surf

    def _render_top_down_array(self):
        """Return top-down view as (TOP_DOWN_H, TOP_DOWN_W, 3) uint8 array."""
        self._ensure_pygame()
        surf = self._render_top_down_surface(scale=1.0,
                                              w=TOP_DOWN_W, h=TOP_DOWN_H)
        return pygame.surfarray.array3d(surf).transpose(1, 0, 2)

    # ------------------------------------------------------------------
    # Rendering — bumper cam
    # ------------------------------------------------------------------

    def _render_bumper_cam(self):
        """
        Render a forward-facing bumper-cam view.

        Strategy: ray-cast from the car's position in a fan of directions
        spanning BUMPER_CAM_FOV degrees ahead. For each ray, find where it
        exits the track (hits a boundary). Map hit distance to a pixel column.
        Render sky (top half) and track surface (bottom half) with boundary
        walls coloured by proximity.

        Returns np.ndarray (BUMPER_CAM_H, BUMPER_CAM_W, 3) uint8.
        """
        self._ensure_pygame()

        img = np.zeros((BUMPER_CAM_H, BUMPER_CAM_W, 3), dtype=np.uint8)

        half_fov = math.radians(BUMPER_CAM_FOV / 2)
        n_cols   = BUMPER_CAM_W

        for col in range(n_cols):
            # Ray angle relative to heading
            frac  = col / (n_cols - 1)          # 0 = leftmost, 1 = rightmost
            angle = self._heading - half_fov + frac * 2 * half_fov

            # March ray outward until it leaves the track or hits max dist
            hit_dist   = BUMPER_CAM_DIST
            hit_colour = COL_GRASS

            step_size = 2.0
            max_steps = int(BUMPER_CAM_DIST / step_size)

            for s in range(1, max_steps + 1):
                d   = s * step_size
                rx  = self._pos[0] + d * math.cos(angle)
                ry  = self._pos[1] + d * math.sin(angle)
                ray = np.array([rx, ry])

                # Find nearest centerline point
                ci   = self._find_closest_idx(ray)
                dist = np.linalg.norm(ray - self.centerline[ci])

                if dist > self.half_widths[ci]:
                    # Ray has left the track
                    hit_dist   = d
                    hit_colour = COL_WALL_NEAR if d < 30 else COL_GRASS
                    break

            # Perspective projection: closer wall = taller column
            wall_height = int(BUMPER_CAM_H * 40 / max(hit_dist, 1))
            wall_height = min(wall_height, BUMPER_CAM_H)

            sky_height   = (BUMPER_CAM_H - wall_height) // 2
            floor_height = BUMPER_CAM_H - sky_height - wall_height

            # Sky
            img[:sky_height, col] = (70, 130, 180)
            # Wall / boundary
            img[sky_height:sky_height + wall_height, col] = hit_colour
            # Track floor
            img[sky_height + wall_height:, col] = COL_TRACK

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
        """Initialise pygame lazily."""
        if self._screen is None and self.render_mode == "human":
            pygame.init()
            self._screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
            pygame.display.set_caption("IARL Race Environment")
            self._clock = pygame.time.Clock()
        elif self._screen is None:
            # Headless: init display-less (needed for surfarray)
            pygame.init()
            self._screen = pygame.Surface((SCREEN_W, SCREEN_H))
            self._clock  = pygame.time.Clock()