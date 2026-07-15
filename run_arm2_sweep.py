"""
run_arm2_sweep.py
------------------
Batch runner for the full Arm 2 difficulty sweep: 7 tracks x 3 fixed
seeds = 21 training runs. Run this from your repo root (same level as
train_arm2.py).

Seeds are fixed at [0, 1, 2] across every track, for apples-to-apples
comparison -- NOT re-randomized per track.

Usage:
    python run_arm2_sweep.py                     # all 21 runs
    python run_arm2_sweep.py --tracks circle oval # just these 2 tracks (x3 seeds = 6 runs)
    python run_arm2_sweep.py --dry-run            # print commands without running them
    python run_arm2_sweep.py --resume             # skip runs whose manifest.json already exists
"""
import argparse
import os
import subprocess
import sys
import time

ALL_TRACKS = ["basra_loop", "circle", "oval", "rectangle", "spur", "bramble", "sawtooth",
              "needle", "serpentine", "narrow", "slalom"]
SEEDS = [0, 1, 2]

TOTAL_TIMESTEPS = 503_808
N_ENVS = 6
DEVICE = "cpu"  # confirmed faster than cuda for this small MLP (see conversation history)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tracks", nargs="+", default=ALL_TRACKS, choices=ALL_TRACKS)
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true",
                         help="Skip any (track, seed) combo whose manifest.json already exists.")
    args = parser.parse_args()

    runs = [(t, s) for t in args.tracks for s in args.seeds]
    print(f"Sweep plan: {len(runs)} runs ({len(args.tracks)} tracks x {len(args.seeds)} seeds)")
    print(f"Tracks: {args.tracks}")
    print(f"Seeds : {args.seeds}")
    print()

    total_start = time.time()
    completed, skipped, failed = 0, 0, 0

    for i, (track, seed) in enumerate(runs, 1):
        manifest_path = os.path.join("arm2_outputs", track, f"seed{seed}", "manifest.json")
        if args.resume and os.path.exists(manifest_path):
            print(f"[{i}/{len(runs)}] SKIP (already done): track={track} seed={seed}")
            skipped += 1
            continue

        cmd = [
            sys.executable, "train_arm2.py",
            "--track", track,
            "--seed", str(seed),
            "--total-timesteps", str(TOTAL_TIMESTEPS),
            "--n-envs", str(N_ENVS),
            "--device", DEVICE,
        ]
        print(f"[{i}/{len(runs)}] track={track} seed={seed}")
        print(f"  $ {' '.join(cmd)}")

        if args.dry_run:
            continue

        run_start = time.time()
        result = subprocess.run(cmd)
        run_elapsed = time.time() - run_start

        if result.returncode != 0:
            print(f"  [FAILED] exit code {result.returncode} after {run_elapsed/60:.1f} min")
            failed += 1
        else:
            print(f"  [OK] completed in {run_elapsed/60:.1f} min")
            completed += 1
        print()

    if not args.dry_run:
        total_elapsed = time.time() - total_start
        print("=" * 70)
        print(f"SWEEP DONE in {total_elapsed/60:.1f} minutes total")
        print(f"  completed: {completed}, skipped: {skipped}, failed: {failed}")
        print("=" * 70)
        if failed > 0:
            print("Some runs failed -- check output above, then re-run with --resume "
                  "to retry only the missing ones.")


if __name__ == "__main__":
    main()