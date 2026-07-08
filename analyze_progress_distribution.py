"""
analyze_progress_distribution.py
---------------------------------
Reads arm2_training_log.csv and summarizes the distribution of
max_track_progress across ALL episodes -- not just the handful printed
to console during training. Answers: is the ~46.2% plateau a hard,
consistent wall, or does it have real spread that the console output's
sampling (every 20th episode) doesn't show?

Usage:
    python analyze_progress_distribution.py
    python analyze_progress_distribution.py --csv arm2_outputs/arm2_training_log.csv
"""

import argparse
import csv
import statistics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default="arm2_outputs/arm2_training_log.csv")
    parser.add_argument("--last-n", type=int, default=200,
                         help="Also show stats for just the last N episodes "
                              "(the converged/plateaued portion), not the whole run.")
    args = parser.parse_args()

    episodes = []
    with open(args.csv, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            episodes.append({
                "episode": int(row["episode"]),
                "progress": float(row["max_track_progress"]),
                "reward": float(row["episode_reward"]),
                "lap": int(row["lap_completed"]),
                "length": int(row["episode_length"]),
            })

    print(f"Total episodes: {len(episodes)}")
    print()

    def summarize(subset, label):
        progresses = [e["progress"] for e in subset]
        laps = sum(e["lap"] for e in subset)
        print(f"--- {label} (n={len(subset)}) ---")
        print(f"  progress: min={min(progresses):.4f}  max={max(progresses):.4f}  "
              f"mean={statistics.mean(progresses):.4f}  median={statistics.median(progresses):.4f}")
        if len(progresses) > 1:
            print(f"  progress stdev: {statistics.stdev(progresses):.4f}")
        print(f"  laps completed: {laps}")
        # Histogram-ish: how many episodes land in each 1%-wide bucket near the plateau
        buckets = {}
        for p in progresses:
            bucket = round(p * 100)  # nearest integer percent
            buckets[bucket] = buckets.get(bucket, 0) + 1
        print(f"  top 10 most common progress% values (rounded to nearest 1%):")
        for val, count in sorted(buckets.items(), key=lambda x: -x[1])[:10]:
            pct_of_total = 100 * count / len(subset)
            print(f"    {val:>3}%: {count:>5} episodes ({pct_of_total:.1f}%)")
        print()

    summarize(episodes, "ALL episodes")
    if len(episodes) > args.last_n:
        summarize(episodes[-args.last_n:], f"LAST {args.last_n} episodes (converged portion)")


if __name__ == "__main__":
    main()