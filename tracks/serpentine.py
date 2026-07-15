"""
track9_serpentine.py
---------------------
Track 9 (new, added to isolate the "repeated features" mechanism): 7
lobes, MODERATE margin per lobe (not razor-thin like Rectangle/
Sawtooth) -- deliberately designed so any resulting difficulty is
attributable to the SHEER NUMBER of repeated precision-demanding
decision points, not to any single one being especially tight.

Companion to Track 8 (Needle), which isolates the opposite mechanism:
one single very tight feature, everything else easy.

Same construction as Bramble (single continuous r(theta) = base_r +
amplitude*cos(k*theta)), just k=7 instead of k=4, re-tuned so per-lobe
margin lands around +24px (comparable to Bramble/Spur's margin, well
above Rectangle/Sawtooth's ~13-16px) while total length still lands
close to the ~1750px target shared across the whole track set.
"""
import math

def build_serpentine_track(cx=400, cy=300, base_r=246, amplitude=26, k=7,
                            steps=350, hw=26, start_offset=-100.5):
    points, widths = [], []
    for i in range(steps):
        theta_deg = start_offset + 360 * i / steps
        theta = math.radians(theta_deg)
        r = base_r + amplitude * math.cos(k * theta)
        points.append((cx + r * math.cos(theta), cy + r * math.sin(theta)))
        widths.append(hw)
    return points, widths