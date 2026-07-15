"""
analyze_arm2_sweep.py
-----------------------
Post-training analysis for the Arm 2 difficulty sweep. Walks
arm2_outputs/{track}/seed{N}/, reads each run's manifest.json + CSV,
joins against difficulty_profile.csv, and produces:
  - per-run summary (one row per track/seed)
  - per-track aggregate (mean/std across the 3 seeds)
  - a difficulty-vs-training-speed correlation check

Run this from your repo root, with difficulty_profile.csv either in
the same directory or passed via --difficulty-csv.

Usage:
    python analyze_arm2_sweep.py
    python analyze_arm2_sweep.py --difficulty-csv tracks/difficulty_profile.csv
"""
import argparse
import csv
import json
import os
import statistics


def load_difficulty_profile(path):
    profile = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            profile[row["track"]] = row
    return profile


def load_run(track, seed):
    run_dir = os.path.join("arm2_outputs", track, f"seed{seed}")
    manifest_path = os.path.join(run_dir, "manifest.json")
    csv_path = os.path.join(run_dir, "arm2_training_log.csv")

    if not os.path.exists(manifest_path):
        return None

    with open(manifest_path) as f:
        manifest = json.load(f)

    # Also compute a couple of things directly from the CSV, as a
    # cross-check against the manifest (e.g. final collision rate over
    # the last N episodes, which the manifest doesn't record directly)
    final_collision_rate = None
    if os.path.exists(csv_path):
        with open(csv_path, newline="") as f:
            episodes = list(csv.DictReader(f))
        if episodes:
            last_n = episodes[-20:] if len(episodes) >= 20 else episodes
            final_collision_rate = sum(
                1 for e in last_n if int(e["lap_completed"]) == 0
            ) / len(last_n)

    return manifest, final_collision_rate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--difficulty-csv", type=str, default="difficulty_profile.csv")
    parser.add_argument("--tracks", nargs="+",
                         default=["basra_loop", "circle", "oval", "rectangle", "spur", "bramble",
                                  "sawtooth", "needle", "serpentine"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    args = parser.parse_args()

    if not os.path.exists(args.difficulty_csv):
        print(f"[WARNING] {args.difficulty_csv} not found -- difficulty correlation will be skipped. "
              f"Run generate_difficulty_profile.py first, or pass --difficulty-csv.")
        difficulty = {}
    else:
        difficulty = load_difficulty_profile(args.difficulty_csv)

    all_runs = []
    missing = []

    for track in args.tracks:
        for seed in args.seeds:
            result = load_run(track, seed)
            if result is None:
                missing.append((track, seed))
                continue
            manifest, final_collision_rate = result
            fl = manifest.get("first_lap", {})
            rc = manifest["reliable_completion"]
            all_runs.append({
                "track": track,
                "seed": seed,
                "first_lap_timestep": fl.get("timestep"),
                "first_lap_wallclock_s": fl.get("wall_clock_s"),
                "reliable_completion_reached": rc["episode"] is not None,
                "reliable_completion_episode": rc["episode"],
                "reliable_completion_timestep": rc["timestep"],
                "reliable_completion_wallclock_s": rc["wall_clock_s"],
                "total_episodes": manifest["total_episodes"],
                "final_collision_rate_last20": final_collision_rate,
                "wall_clock_minutes_total": manifest["wall_clock_minutes"],
            })

    if missing:
        print(f"[WARNING] {len(missing)} run(s) not found (not yet trained, or failed):")
        for t, s in missing:
            print(f"  track={t} seed={s}")
        print()

    if not all_runs:
        print("No completed runs found. Run the sweep first.")
        return

    # --- Per-run table ---
    print("=" * 100)
    print("PER-RUN RESULTS")
    print("=" * 100)
    print(f"{'track':>12} {'seed':>5} {'1st_lap':>9} {'reliable?':>10} {'ep':>6} {'steps':>8} {'wall_s':>8} {'coll_rate':>10}")
    for r in all_runs:
        first_lap = r["first_lap_timestep"] if r["first_lap_timestep"] is not None else "-"
        reached = "yes" if r["reliable_completion_reached"] else "NO"
        ep = r["reliable_completion_episode"] if r["reliable_completion_episode"] is not None else "-"
        steps = r["reliable_completion_timestep"] if r["reliable_completion_timestep"] is not None else "-"
        wc = f"{r['reliable_completion_wallclock_s']:.0f}" if r["reliable_completion_wallclock_s"] is not None else "-"
        cr = f"{r['final_collision_rate_last20']:.0%}" if r["final_collision_rate_last20"] is not None else "-"
        print(f"{r['track']:>12} {r['seed']:>5} {first_lap!s:>9} {reached:>10} {ep!s:>6} {steps!s:>8} {wc:>8} {cr:>10}")

    # --- Per-track aggregate (mean/std across seeds) ---
    print()
    print("=" * 100)
    print("PER-TRACK AGGREGATE (mean +/- stdev across seeds, reliable-completion runs only)")
    print("=" * 100)
    print(f"{'track':>12} {'n_seeds':>8} {'reached':>8} {'mean_steps':>12} {'std_steps':>10} {'margin_px':>10}")

    track_aggregates = {}
    for track in args.tracks:
        track_runs = [r for r in all_runs if r["track"] == track]
        reached_runs = [r for r in track_runs if r["reliable_completion_reached"]]
        if not track_runs:
            continue
        steps_list = [r["reliable_completion_timestep"] for r in reached_runs]
        mean_steps = statistics.mean(steps_list) if steps_list else None
        std_steps = statistics.stdev(steps_list) if len(steps_list) > 1 else 0.0 if steps_list else None
        margin = difficulty.get(track, {}).get("curvature_margin_px", "?")

        track_aggregates[track] = {
            "n_seeds": len(track_runs), "n_reached": len(reached_runs),
            "mean_steps": mean_steps, "std_steps": std_steps, "margin_px": margin,
        }

        mean_str = f"{mean_steps:,.0f}" if mean_steps is not None else "-"
        std_str = f"{std_steps:,.0f}" if std_steps is not None else "-"
        print(f"{track:>12} {len(track_runs):>8} {len(reached_runs)}/{len(track_runs):>6} "
              f"{mean_str:>12} {std_str:>10} {margin!s:>10}")

    # --- Difficulty correlation (simple, honest -- just Pearson r, no overclaiming) ---
    if difficulty:
        pairs = [
            (float(track_aggregates[t]["margin_px"]), track_aggregates[t]["mean_steps"])
            for t in track_aggregates
            if track_aggregates[t]["mean_steps"] is not None and track_aggregates[t]["margin_px"] != "?"
        ]
        if len(pairs) >= 3:
            margins = [p[0] for p in pairs]
            steps = [p[1] for p in pairs]
            n = len(pairs)
            mean_m, mean_s = sum(margins)/n, sum(steps)/n
            cov = sum((m-mean_m)*(s-mean_s) for m, s in pairs) / n
            std_m = (sum((m-mean_m)**2 for m in margins)/n) ** 0.5
            std_s = (sum((s-mean_s)**2 for s in steps)/n) ** 0.5
            r = cov / (std_m * std_s) if std_m > 0 and std_s > 0 else None
            print()
            print("=" * 100)
            print("DIFFICULTY CORRELATION (margin_px vs mean steps-to-reliable-completion)")
            print("=" * 100)
            if r is not None:
                print(f"Pearson r = {r:+.3f}  (n={n} tracks)")
                print("Negative r = tighter margin (harder track) associates with more steps needed (expected direction).")
                print("This is descriptive, not a significance test -- n is small (<=7 tracks). Treat as suggestive, not conclusive.")
            else:
                print("Could not compute (insufficient variance).")
        else:
            print("\nNot enough completed tracks yet for a correlation check (need >=3).")


if __name__ == "__main__":
    main()