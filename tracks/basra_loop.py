"""
basra_loop.py
--------------
Track 1 (existing): the fixed Basra Loop hairpin track. Reconstructed
here so it can run through the SAME verify.py pipeline as the other 6
tracks, for a true apples-to-apples difficulty comparison -- its
original verification used an earlier version of this pipeline, before
the loop-closure duplicate-point fix and extended braking lookahead
existed.
"""
import math

def arc(cx, cy, r, a_start_deg, a_end_deg, steps):
    return [(cx + r*math.cos(math.radians(a_start_deg + (i/steps)*(a_end_deg-a_start_deg))),
             cy + r*math.sin(math.radians(a_start_deg + (i/steps)*(a_end_deg-a_start_deg))))
            for i in range(steps+1)]

def straight(x0, y0, x1, y1, steps):
    return [(x0+(i/steps)*(x1-x0), y0+(i/steps)*(y1-y0)) for i in range(steps+1)]

def build_basra_loop_track():
    points, widths = [], []
    def add(pts, hw): points.extend(pts); widths.extend([hw]*len(pts))

    add(straight(120, 490, 550, 490, steps=30), 38)
    add(arc(cx=550, cy=310, r=180, a_start_deg=90, a_end_deg=0, steps=20), 34)
    add(straight(730, 310, 730, 255, steps=6), 30)          # S3, shortened (fixed hairpin)
    add(arc(cx=675, cy=255, r=55, a_start_deg=0, a_end_deg=-90, steps=24), 26)  # fixed hairpin
    add(straight(675, 200, 200, 200, steps=20), 30)          # S5, lengthened
    add(arc(cx=200, cy=310, r=110, a_start_deg=270, a_end_deg=180, steps=16), 32)
    add(straight(90, 310, 120, 490, steps=14), 34)

    return points, widths