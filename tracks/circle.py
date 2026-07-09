"""
circle.py
---------
Track 2: Circle. Easiest-end anchor for the difficulty sweep -- constant
curvature, no direction reversals, no tight spots by design.

Point[0] is positioned so its tangent points due east (0 degrees) --
race_env.py's reset() hardcodes the starting heading to 0.0 regardless
of track, so every new track's point ordering needs this to hold, or
the car starts every episode with an uncorrected heading error (caught
here after it broke the first version of this track -- see Stage 0 in
verify.py).
"""
import math

def build_circle_track(cx=400, cy=300, r=278, steps=140, hw=32):
    """
    Circumference = 2*pi*r = ~1746px at r=278, close to the target ~1714px.
    Starting angle offset (-90 deg / 270 deg) chosen so point[0]'s tangent
    for increasing-angle (CCW) traversal points due east -- verified
    numerically, not just derived by hand, given how easy it is to get
    an East/West or North/South sign flipped in screen coordinates.
    """
    points, widths = [], []
    start_offset = -90  # degrees; verified below to give east-facing tangent at point[0]
    for i in range(steps + 1):
        a = math.radians(start_offset + 360 * i / steps)
        points.append((cx + r * math.cos(a), cy + r * math.sin(a)))
        widths.append(hw)
    return points, widths