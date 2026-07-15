"""
slalom.py
---------
Track 11: Slalom. The hardest track in the set, by design -- built
specifically to maximize TOTAL ACCUMULATED TURNING, the single
strongest predictor of training difficulty found across the whole
sweep (r=0.820 against mean steps-to-reliable-completion, beating both
raw curvature margin and simple feature-count).

total_turn = 1404 degrees over one lap -- ~29% more than Serpentine
(the previous highest, 1075deg), and nearly 4x a full circle's 360deg.
Margin (+12.9px) deliberately kept comparably tight to Sawtooth's
(+15.8px), not loosened -- unlike Serpentine, which intentionally kept
margin generous to isolate repetition from tightness. Here the goal is
the opposite: combine both known-relevant factors rather than isolate
one, since the corridor-WIDTH axis (tested via Narrow) was found to
add essentially zero difficulty in this observation/reward design and
was deliberately excluded as a lever (hw held constant at 26px,
identical to Sawtooth/Serpentine/Bramble).

9-lobe periodic curve (r(theta) = base_r + amplitude*cos(9*theta)),
same proven construction as Bramble (k=4) and Serpentine (k=7), just
pushed further. Name: "slalom" is the actual motorsport/skiing term
for a course of rapid alternating turns -- describes the mechanism
this track was built to test, not just "hard" in the abstract.

Verified: all 5 pipeline stages pass, including autopilot robustness
down to a still-positive 6.0px margin under +/-3px/step injected
noise -- tight, but not broken.
"""
import math

def build_slalom_track(cx=400, cy=300, base_r=255, amplitude=22, k=9,
                        steps=600, hw=26, start_offset=-115.5):
    points, widths = [], []
    for i in range(steps):
        theta_deg = start_offset + 360 * i / steps
        theta = math.radians(theta_deg)
        r = base_r + amplitude * math.cos(k * theta)
        points.append((cx + r * math.cos(theta), cy + r * math.sin(theta)))
        widths.append(hw)
    return points, widths