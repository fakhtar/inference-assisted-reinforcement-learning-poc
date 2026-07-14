"""
bramble.py
----------
Track 6: Bramble. Hard difficulty -- 4 lobes with GENUINE concave necks
between them (a rose/clover-style curve), not just gentle outward bumps
on a circle -- an earlier version used strictly-positive Gaussian bumps
to avoid ever dipping below the base radius, which was safer to verify
but produced a shape that looked more like "a lumpy circle" than the
distinctly-lobed, narrow-waisted shape in the original sketch. This
version uses r(theta) = base_r + amplitude*cos(4*theta), which oscillates
symmetrically above AND below base_r -- true outward lobes and true
inward pinches -- while remaining a single continuous function (still
no branching junctions to tangent-match, same safety property as
before, just now genuinely non-convex at the necks).

Verified: margin +22.6px over the car's min turning radius (BETTER
than the old smooth version's +13.7px, despite being visually more
extreme), total length 1746.8px (on target), 10/10 lap completion at
every tested noise level up to +/-3px/step.

Uses range(steps), not range(steps+1) -- avoids the loop-closure
duplicate point bug found during this track's development (see git
history / conversation notes -- also backported to circle.py/spur.py).
"""
import math

def build_bramble_track(cx=400, cy=300, base_r=232, amplitude=55, k=4,
                         steps=280, hw=28, start_offset=-90.0):
    """
    k=4 -- four lobes. base_r=232, amplitude=55 tuned together for
    length ~1747px and margin +22.6px. start_offset=-90.0 makes
    point[0]'s tangent match reset()'s hardcoded east heading exactly.
    """
    points, widths = [], []
    for i in range(steps):
        theta_deg = start_offset + 360 * i / steps
        theta = math.radians(theta_deg)
        r = base_r + amplitude * math.cos(k * theta)
        points.append((cx + r * math.cos(theta), cy + r * math.sin(theta)))
        widths.append(hw)
    return points, widths