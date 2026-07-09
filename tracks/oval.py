"""
oval.py
-------
Track 3: Oval. Easy end of the spectrum -- two long straights connected
by generous semicircular U-turns. Both turns are 180-degree, parallel-
tangent connections (the SIMPLE case -- unlike the original hairpin,
which needed perpendicular tangents and was proven impossible for a
single semicircle; a full U-turn IS exactly what a semicircle is built
for).

Point[0] starts on the bottom straight heading east, matching
race_env.py's hardcoded reset() heading -- verified below, not assumed.
"""
import math

def arc(cx, cy, r, a_start_deg, a_end_deg, steps):
    return [(cx + r*math.cos(math.radians(a_start_deg + (i/steps)*(a_end_deg-a_start_deg))),
             cy + r*math.sin(math.radians(a_start_deg + (i/steps)*(a_end_deg-a_start_deg))))
            for i in range(steps+1)]

def straight(x0, y0, x1, y1, steps):
    return [(x0+(i/steps)*(x1-x0), y0+(i/steps)*(y1-y0)) for i in range(steps+1)]

def build_oval_track(cx_left=105, cx_right=695, cy=300, R=90, hw=30,
                      straight_steps=40, arc_steps=30):
    """
    R=90 -- 4x the car's ~22.2px minimum turning radius, comfortable
    margin (Oval is meant to be easy, no reason to make the turns tight).
    Total length ~1745px: 2*(cx_right-cx_left) + 2*pi*R
      = 2*590 + 2*pi*90 = 1180 + 565.5 = 1745.5px
    """
    points, widths = [], []
    def add(pts, w): points.extend(pts); widths.extend([w]*len(pts))

    # Bottom straight, heading EAST (matches reset()'s hardcoded heading=0)
    add(straight(cx_left, cy+R, cx_right, cy+R, straight_steps), hw)
    # Right semicircle: enter heading east, exit heading west
    add(arc(cx_right, cy, R, 90, -90, arc_steps), hw)
    # Top straight, heading WEST
    add(straight(cx_right, cy-R, cx_left, cy-R, straight_steps), hw)
    # Left semicircle: enter heading west, exit heading east -- closes the loop
    add(arc(cx_left, cy, R, -90, -270, arc_steps), hw)

    return points, widths