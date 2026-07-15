"""
narrow.py
---------
Track 10 (new): Narrow. Isolates corridor-WIDTH difficulty specifically
-- the axis Needle was originally supposed to test but accidentally
didn't, since Needle kept a wide, constant corridor while only
tightening centerline curvature.

Centerline is a PURE CIRCLE (same r=278 as the Circle track, zero
curvature variation anywhere) -- this makes turning difficulty as
close to zero as possible, isolating width as the only variable.
half_width narrows smoothly (Gaussian dip) from 32px down to 10px in
one region, then widens back -- a genuine "narrow passage," not a
sharp turn.

Verified: Stage 2 (curvature margin) trivially passes with huge margin
(+255.8px, identical to Circle) since it only checks centerline
bending -- it does NOT check width. That's not a bug; it's exactly why
this track needed a distinct reported metric. See min_half_width in
the verification report. The autopilot (Stage 3/4) passes cleanly too
(worst-case margin +3.7px even at 8px min-width and +/-3px/step noise),
but this confirms PHYSICAL drivability only -- it does not predict
whether an RL policy, starting from unstructured exploration, will
easily discover that tight centering is required here. That's the
actual open question this track exists to test.
"""
import math

def build_narrow_track(cx=400, cy=300, r=278, base_hw=32, min_hw=10,
                        narrow_center_deg=45, narrow_width_deg=40,
                        steps=200, start_offset=-90):
    points, widths = [], []
    for i in range(steps):
        theta_deg = start_offset + 360 * i / steps
        theta = math.radians(theta_deg)
        points.append((cx + r * math.cos(theta), cy + r * math.sin(theta)))
        d = (theta_deg - narrow_center_deg + 180) % 360 - 180
        dip = (base_hw - min_hw) * math.exp(-(d / (narrow_width_deg / 2)) ** 2)
        widths.append(base_hw - dip)
    return points, widths