#!/usr/bin/env python
"""Merge per-shard K-battery CSVs into compare_results_k/results_k.csv.

- Concatenates results_shard*.csv, deduplicates by run key keeping the LAST
  row (so re-runs overwrite stale rows, same convention as the vertex battery).
- Merges completed_shard*.txt into completed_runs.txt.
- Prints coverage per (K, method, nr): completed / expected.

Usage:
    python server/aggregate_k.py [--out-dir compare_results_k]
"""
from __future__ import annotations
import sys, os, csv, glob, argparse


def run_key(row):
    return (row["num_regions"], row["num_heights"], row["seed"],
            row["endurance"], row["k"], row["method"])


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--out-dir", default="compare_results_k")
    args = p.parse_args()

    shards = sorted(glob.glob(os.path.join(args.out_dir,
                                           "results_shard*.csv")))
    if not shards:
        print("No shard CSVs in %s" % args.out_dir)
        return 1
    print("Merging %d shard files" % len(shards))

    rows = {}
    fields = None
    for path in shards:
        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            if fields is None:
                fields = reader.fieldnames
            n = 0
            for row in reader:
                rows[run_key(row)] = row  # last wins
                n += 1
        print("  %s: %d rows" % (os.path.basename(path), n))

    out_csv = os.path.join(args.out_dir, "results_k.csv")
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for key in sorted(rows):
            writer.writerow(rows[key])
    print("Wrote %s (%d unique runs)" % (out_csv, len(rows)))

    done = set()
    for path in sorted(glob.glob(os.path.join(args.out_dir,
                                              "completed_shard*.txt"))):
        with open(path) as f:
            done.update(line.strip() for line in f if line.strip())
    with open(os.path.join(args.out_dir, "completed_runs.txt"), "w") as f:
        for key in sorted(done):
            f.write(key + "\n")
    print("Merged %d completed markers" % len(done))

    # coverage per (K, method, nr)
    cov = {}
    for (nr, nh, seed, E, K, method) in rows:
        cov[(K, method, nr)] = cov.get((K, method, nr), 0) + 1
    print("Coverage (K, method, nr) -> runs:")
    for key in sorted(cov):
        print("  %s -> %d" % (key, cov[key]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
