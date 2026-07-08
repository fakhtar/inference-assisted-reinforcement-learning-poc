"""
structured_state.py
--------------------
Arm 2 / Arm 3 observation wrapper.

Replaces the RaceEnv's pixel observation with a compact, egocentric,
relative structured-state vector -- position/heading/speed relative to
the track, plus a tiered arc-length lookahead of upcoming curvature and
track width. Every feature is relative to the car's own frame of
reference and normalized by THIS track's own geometry (not hardcoded
pixel constants), so the representation is not tied to this specific
track's absolute layout or scale -- the design goal being that a policy
trained on one track's structured state has a chance of transferring to
a different track's geometry, unlike a representation using absolute
world coordinates or this-track-specific fractions (e.g. checkpoint
index, total lap progress).

Feature vector (14-dim, float32):
    [0]  lateral_offset      -- signed distance from centerline / local
                                 half-width. 0 = centered, +/-1 = at the
                                 track edge.
    [1]  heading_error       -- (car heading - track tangent) / pi.
                                 0 = aligned with track direction.
    [2]  speed_norm          -- speed / MAX_SPEED.
    [3]  current_hw_norm     -- half-width at car's position / this
                                 track's own max half-width.
    [4:14] 5 x (heading_delta, hw_norm) at arc-length lookahead offsets
           of 15/35/65/110/175px ahead of the car's current position,
           geometrically spaced (dense near-term, sparse far-term) --
           dense spacing near the car for precise immediate steering,
           sparse spacing further out for early warning of upcoming
           turns (e.g. the hairpin) that a narrow-FOV camera can't see
           in time to react to.

Requires race_env.py's render_front_camera=False (set via env_kwargs
when constructing the base env) to avoid paying for an unused pixel
render every step -- this wrapper never reads the base env's returned
observation at all.
"""

import math
import numpy as np
import gymnasium as gym
from gymnasium import spaces


LOOKAHEAD_OFFSETS = [15, 35, 65, 110, 175]  # pixels ahead, arc-length
FEATURE_DIM = 4 + 2 * len(LOOKAHEAD_OFFSETS)  # 14


def _wrap_pi(x):
    return (x + math.pi) % (2 * math.pi) - math.pi


class StructuredStateWrapper(gym.ObservationWrapper):
    """
    Wraps iarl/RaceTrack-v0 (or any env exposing the same internal track
    geometry attributes) to replace the pixel observation with the
    structured-state vector described above.

    Reads track geometry (centerline, half_widths, arc_lengths,
    total_length) and live car state (_pos, _heading, _speed,
    _closest_idx) directly from the unwrapped base environment. These
    are read-only reads of existing internal state -- nothing about the
    base environment's physics, reward, or action space is touched.
    """

    def __init__(self, env):
        super().__init__(env)

        base = self.env.unwrapped
        self._centerline = base.centerline
        self._half_widths = base.half_widths
        self._arc_lengths = base.arc_lengths
        self._total_length = base.total_length
        self._N = base.N

        # Precompute tangent angle at every centerline point once, up
        # front -- same tangent formula race_env.py uses internally for
        # boundary normals, just expressed as an angle instead of a
        # normal vector, since we need signed heading comparisons.
        self._tangent_angles = np.zeros(self._N, dtype=np.float64)
        for i in range(self._N):
            prev = self._centerline[(i - 1) % self._N]
            nxt  = self._centerline[(i + 1) % self._N]
            t = nxt - prev
            self._tangent_angles[i] = math.atan2(t[1], t[0])

        self._max_hw = float(self._half_widths.max())

        # Pull MAX_SPEED from the race_env module so this wrapper doesn't
        # hardcode a value that could drift out of sync with the env.
        import iarl.envs.race_env as race_env_module
        self._max_speed = race_env_module.MAX_SPEED

        self.observation_space = spaces.Box(
            low=-10.0, high=10.0, shape=(FEATURE_DIM,), dtype=np.float32
        )

    def observation(self, obs):
        # `obs` (the base env's pixel array or zero-placeholder) is
        # intentionally ignored -- we read live state directly from the
        # unwrapped env instead.
        base = self.env.unwrapped

        closest_idx = base._closest_idx
        pos     = base._pos
        heading = base._heading
        speed   = base._speed

        c   = self._centerline[closest_idx]
        hw  = self._half_widths[closest_idx]
        tangent = self._tangent_angles[closest_idx]

        # Signed lateral offset: project (pos - centerline_point) onto
        # the left-normal direction (same convention as race_env's
        # boundary construction: rotate tangent 90 deg CCW).
        nx, ny = -math.sin(tangent), math.cos(tangent)
        dx, dy = pos[0] - c[0], pos[1] - c[1]
        lateral = dx * nx + dy * ny
        lateral_norm = lateral / hw if hw > 1e-6 else 0.0

        heading_err = _wrap_pi(heading - tangent) / math.pi
        speed_norm  = speed / self._max_speed
        hw_norm     = hw / self._max_hw

        feats = [lateral_norm, heading_err, speed_norm, hw_norm]

        cur_arc = self._arc_lengths[closest_idx]
        for off in LOOKAHEAD_OFFSETS:
            target = (cur_arc + off) % self._total_length
            idx = int(np.argmin(np.abs(self._arc_lengths - target)))
            hd = _wrap_pi(self._tangent_angles[idx] - heading) / math.pi
            hw_l = self._half_widths[idx] / self._max_hw
            feats.extend([hd, hw_l])

        return np.array(feats, dtype=np.float32)