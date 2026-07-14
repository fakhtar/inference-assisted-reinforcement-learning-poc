"""
sawtooth.py
-----------
Track 7: Sawtooth. Hardest difficulty -- 3 peaks, 2 valleys, 5 sharp
vertices in close sequence. Built as straight segments + fillet arcs at
each vertex (same approach as Rectangle, generalized to N vertices,
each with its own turn angle rather than assuming 90 degrees).
"""
import math


def build_sawtooth_track(hw=27, R=38, straight_steps=25, arc_steps=16, verbose=False):
    vertices = [
        (747.5, 457.9),  # BR -- index 0
        (600.1, 78.9),   # P3
        (515.8, 284.2),  # V2
        (431.6, 78.9),   # P2
        (347.4, 284.2),  # V1
        (263.1, 78.9),   # P1
        (126.2, 489.5),  # BL -- last vertex, closes loop as edge 0
    ]
    n = len(vertices)

    fillet_starts, fillet_ends, centers, a_starts, a_ends = [], [], [], [], []
    trims = []

    for i in range(n):
        p_prev = vertices[(i - 1) % n]
        p_curr = vertices[i]
        p_next = vertices[(i + 1) % n]

        v_in = (p_curr[0] - p_prev[0], p_curr[1] - p_prev[1])
        v_out = (p_next[0] - p_curr[0], p_next[1] - p_curr[1])
        len_in = math.hypot(*v_in)
        len_out = math.hypot(*v_out)
        u_in = (v_in[0] / len_in, v_in[1] / len_in)
        u_out = (v_out[0] / len_out, v_out[1] / len_out)

        angle_in = math.atan2(u_in[1], u_in[0])
        angle_out = math.atan2(u_out[1], u_out[0])
        # turn: signed angle from incoming to outgoing direction, in (-pi, pi]
        turn = (angle_out - angle_in + math.pi) % (2 * math.pi) - math.pi
        # Standard polygon fillet formula: trim distance back from the
        # corner along each edge = R / tan(interior_angle / 2), where
        # interior_angle is the angle BETWEEN the two edges (not the
        # turn angle) = pi - abs(turn).
        interior_angle = math.pi - abs(turn)
        trim = R / math.tan(interior_angle / 2) if interior_angle > 1e-6 else R * 1000
        trims.append(trim)

        fs = (p_curr[0] - u_in[0] * trim, p_curr[1] - u_in[1] * trim)
        fe = (p_curr[0] + u_out[0] * trim, p_curr[1] + u_out[1] * trim)

        sign = 1 if turn > 0 else -1
        normal_in = (-u_in[1], u_in[0])  # left-normal of incoming direction
        center = (fs[0] + normal_in[0] * R * sign, fs[1] + normal_in[1] * R * sign)

        a_s = math.degrees(math.atan2(fs[1] - center[1], fs[0] - center[0]))
        a_e = math.degrees(math.atan2(fe[1] - center[1], fe[0] - center[0]))
        # Sweep the SHORT way, in the direction matching `sign`
        if sign > 0:
            while a_e < a_s: a_e += 360
            while a_e - a_s > 180: a_e -= 360
        else:
            while a_e > a_s: a_e -= 360
            while a_s - a_e > 180: a_e += 360

        if verbose:
            print(f"  vertex {i} {p_curr}: turn={math.degrees(turn):+.1f} sign={sign} "
                  f"trim={trim:.1f} sweep={a_e-a_s:+.1f}deg center={tuple(round(c,1) for c in center)}")

        fillet_starts.append(fs); fillet_ends.append(fe); centers.append(center)
        a_starts.append(a_s); a_ends.append(a_e)

    # Collision check
    collisions = []
    for i in range(n):
        p_curr = vertices[i]
        p_next = vertices[(i + 1) % n]
        edge_len = math.hypot(p_next[0] - p_curr[0], p_next[1] - p_curr[1])
        needed = trims[i] + trims[(i + 1) % n]
        if needed > edge_len:
            collisions.append((i, edge_len, needed))
    if collisions:
        for i, edge_len, needed in collisions:
            print(f"  [WARNING] fillet collision on edge {i}->{(i+1)%n}: "
                  f"edge_len={edge_len:.1f}px, fillets need {needed:.1f}px")

    def straight(p0, p1, steps):
        return [(p0[0] + (k/steps)*(p1[0]-p0[0]), p0[1] + (k/steps)*(p1[1]-p0[1]))
                for k in range(steps + 1)]

    points, widths = [], []
    for i in range(n):
        prev_end = fillet_ends[(i - 1) % n]
        this_start = fillet_starts[i]
        for pt in straight(prev_end, this_start, straight_steps):
            points.append(pt); widths.append(hw)
        for j in range(arc_steps + 1):
            t = j / arc_steps
            a = math.radians(a_starts[i] + t * (a_ends[i] - a_starts[i]))
            points.append((centers[i][0] + R*math.cos(a), centers[i][1] + R*math.sin(a)))
            widths.append(hw)

    return points, widths