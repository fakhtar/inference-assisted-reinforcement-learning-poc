"""
spur.py
-------
Track 5: Spur. Moderate difficulty -- a single isolated elbow/protrusion
feature on an otherwise circular base, closest in structure to the
hairpin (one tight feature, rest of the track easy).

Built as ONE continuous curve with a smooth radial bulge (Gaussian bump
in radius as a function of angle) rather than a branching neck that
separates and rejoins -- this guarantees tangent continuity everywhere
by construction, avoiding the same class of junction-matching problem
the original hairpin needed fixing.
"""
import math

def build_spur_track(cx=400, cy=300, base_r=258, bump_amplitude=110,
                      bump_center_deg=45, bump_width_deg=35, steps=200, hw=30):
    """
    base_r=258 -- tuned so total length lands at ~1749px, matching the
    other configs (an earlier base_r=230 came in short at ~1578px).
    bump_amplitude=110 -- a clear, visible protrusion.
    bump_width_deg=35 -- controls how tight the bump's curvature gets;
    verified via the pipeline (margin +28.0px over car's min radius).

    Uses range(steps), NOT range(steps+1) -- avoids duplicating point[0]
    at the loop closure (see circle.py for the full explanation; caused
    a real failure on Bramble, harmless here but fixed at the source).
    """
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