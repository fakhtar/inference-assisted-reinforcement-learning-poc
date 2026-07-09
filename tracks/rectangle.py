"""
rectangle.py
------------
Track 4: Rectangle. Moderate difficulty -- 4 sharp-but-feasible corners,
minimal rounding (as specified). Unlike the original hairpin, each 90-deg
corner here has adjacent edges offset in BOTH x and y, so there's no
same-coordinate degeneracy -- a single quarter-circle fillet per corner
works cleanly at any radius.

W=580, H=330, R=35 (minimal rounding -- margin over car's min turning
radius is real but deliberately tighter than Oval's, per "minimal
rounding" spec). Target total length ~1750px, matching other configs.

Point[0] starts on the bottom straight heading east, matching
reset()'s hardcoded starting heading.
"""
import math

def arc(cx, cy, r, a_start_deg, a_end_deg, steps):
    return [(cx + r*math.cos(math.radians(a_start_deg + (i/steps)*(a_end_deg-a_start_deg))),
             cy + r*math.sin(math.radians(a_start_deg + (i/steps)*(a_end_deg-a_start_deg))))
            for i in range(steps+1)]

def straight(x0, y0, x1, y1, steps):
    return [(x0+(i/steps)*(x1-x0), y0+(i/steps)*(y1-y0)) for i in range(steps+1)]

def build_rectangle_track(x_left=110, x_right=690, y_top=135, y_bottom=465, R=35, hw=28,
                           straight_steps=40, arc_steps=20):
    points, widths = [], []
    def add(pts, w): points.extend(pts); widths.extend([w]*len(pts))

    # Bottom straight: heading east (matches reset()'s hardcoded heading=0)
    add(straight(x_left+R, y_bottom, x_right-R, y_bottom, straight_steps), hw)
    # Bottom-right fillet: east -> north
    add(arc(x_right-R, y_bottom-R, R, 90, 0, arc_steps), hw)
    # Right straight: heading north
    add(straight(x_right, y_bottom-R, x_right, y_top+R, straight_steps), hw)
    # Top-right fillet: north -> west
    add(arc(x_right-R, y_top+R, R, 0, -90, arc_steps), hw)
    # Top straight: heading west
    add(straight(x_right-R, y_top, x_left+R, y_top, straight_steps), hw)
    # Top-left fillet: west -> south
    add(arc(x_left+R, y_top+R, R, 270, 180, arc_steps), hw)
    # Left straight: heading south
    add(straight(x_left, y_top+R, x_left, y_bottom-R, straight_steps), hw)
    # Bottom-left fillet: south -> east, closes the loop
    add(arc(x_left+R, y_bottom-R, R, 180, 90, arc_steps), hw)

    return points, widths