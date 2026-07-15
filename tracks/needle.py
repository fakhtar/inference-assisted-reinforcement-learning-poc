"""
track8_needle.py
------------------
Track 8 (new, added to isolate the "single tight margin" mechanism):
ONE deliberately very tight feature -- tighter than any other track in
the set (margin +6.8px vs. Rectangle's previous minimum of +12.8px) --
with everything else easy (a wide base circle), so any resulting
difficulty is attributable to raw curvature tightness, not to feature
repetition.

Companion to Track 9 (Serpentine), which isolates the opposite
mechanism: 7 repeated moderate-margin features, no single one tight.

Same construction as Spur (single continuous Gaussian bump on a
circle) -- guarantees tangent continuity by construction, no branching
junctions. Note: the STATIC 3-point curvature estimate (Stage 2) is
noticeably more conservative than the actual dynamic autopilot result
here (Stage 2 margin +6.8px vs. Stage 3 clean-lap margin +18.5px, and
+11.1px even under +/-3px/step injected noise) -- worth remembering
this static/dynamic gap exists and isn't unique to this track, it's
just more visible here because the margin is deliberately small.
"""
import math

def build_needle_track(cx=400, cy=300, base_r=260, bump_amplitude=130,
                        bump_center_deg=45, bump_width_deg=26, steps=250, hw=24,
                        start_offset=-90):
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