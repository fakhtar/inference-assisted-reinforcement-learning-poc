"""
run_all_checks.py
------------------
Runs the full verification pipeline (Stages 0-4) against every track
in the difficulty sweep, in one go. Use this any time you want to
re-confirm all 7 tracks are still safely drivable -- e.g. after
tweaking any track's parameters, or just as a periodic sanity check
before a training run.

Usage:
    python run_all_checks.py
    python run_all_checks.py --track circle       # just one track
"""
import argparse
import sys

from verify import verify_track

from circle import build_circle_track
from oval import build_oval_track
from rectangle import build_rectangle_track
from spur import build_spur_track
from bramble import build_bramble_track
from sawtooth import build_sawtooth_track

TRACKS = {
    "circle": build_circle_track,
    "oval": build_oval_track,
    "rectangle": build_rectangle_track,
    "spur": build_spur_track,
    "bramble": build_bramble_track,
    "sawtooth": build_sawtooth_track,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--track", type=str, default=None,
                         help="Check only this track (default: check all 6)")
    args = parser.parse_args()

    to_check = {args.track: TRACKS[args.track]} if args.track else TRACKS

    results = {}
    for name, builder in to_check.items():
        points, widths = builder()
        result = verify_track(points, widths, name.capitalize())
        results[name] = result
        print()

    print("=" * 70)
    print("SUMMARY -- all tracks checked")
    print("=" * 70)
    print(f"{'track':>12} {'length':>10} {'margin':>10} {'result':>10}")
    all_pass = True
    for name, r in results.items():
        status = "PASS" if r["overall_pass"] else "FAIL"
        if not r["overall_pass"]:
            all_pass = False
        print(f"{name:>12} {r['total_length']:>10.1f} {r['curvature_margin']:>+10.1f} {status:>10}")
    print("=" * 70)

    if not all_pass:
        print("At least one track FAILED -- do not train against it until fixed.")
        sys.exit(1)
    else:
        print("All tracks pass. Safe to train.")


if __name__ == "__main__":
    main()