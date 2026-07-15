"""
generate_difficulty_profile.py
--------------------------------
Runs the verification pipeline against all 7 tracks and writes a single
CSV summarizing each one's difficulty descriptors. This is the file the
post-training analysis script joins against training results, so the
difficulty-vs-training-speed correlation can actually be tested rather
than eyeballed.

Usage:
    python generate_difficulty_profile.py
"""
import csv

from basra_loop import build_basra_loop_track
from circle import build_circle_track
from oval import build_oval_track
from rectangle import build_rectangle_track
from spur import build_spur_track
from bramble import build_bramble_track
from sawtooth import build_sawtooth_track
from track8_needle import build_needle_track
from track9_serpentine import build_serpentine_track
from narrow import build_narrow_track
from slalom import build_slalom_track
from verify import verify_track

TRACKS = [
    ("basra_loop", build_basra_loop_track, {0}),  # skip_tangent_idx: pre-existing,
    ("circle", build_circle_track, None),          # unrelated loop-closure issue
    ("oval", build_oval_track, None),
    ("rectangle", build_rectangle_track, None),
    ("spur", build_spur_track, None),
    ("bramble", build_bramble_track, None),
    ("sawtooth", build_sawtooth_track, None),
    ("needle", build_needle_track, None),
    ("serpentine", build_serpentine_track, None),
    ("narrow", build_narrow_track, None),
    ("slalom", build_slalom_track, None),
]

rows = []
for name, builder, skip_idx in TRACKS:
    points, widths = builder()
    result = verify_track(points, widths, name, skip_tangent_idx=skip_idx)
    rows.append({
        "track": name,
        "total_length_px": round(result["total_length"], 1),
        "curvature_margin_px": round(result["curvature_margin"], 1),
        "tightest_radius_px": round(result["curvature_worst_radius"], 1),
        "min_half_width_px": round(result["min_half_width"], 1),
        "total_turn_deg": round(result["total_turn_deg"], 1),
        "autopilot_clean_margin_px": round(result["autopilot_clean_margin"], 1),
        "verified_pass": result["overall_pass"],
    })
    print()

out_path = "difficulty_profile.csv"
with open(out_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)

print("=" * 70)
print(f"Difficulty profile written to {out_path}")
print("=" * 70)
# Sorted by margin, easiest to hardest -- a quick visual sanity check
for r in sorted(rows, key=lambda r: -r["curvature_margin_px"]):
    print(f"  {r['track']:>12}: margin={r['curvature_margin_px']:>+7.1f}px  length={r['total_length_px']:>8.1f}px")