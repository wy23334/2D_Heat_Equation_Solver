# -*- coding: utf-8 -*-
"""Benchmark and plot the four implementations required by Phase 3."""

import argparse
import csv
import os
import re
import statistics
import subprocess
import sys

import matplotlib.pyplot as plt


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
CSV_PATH = os.path.join(DATA_DIR, "phase_3_four_versions.csv")
PNG_PATH = os.path.join(DATA_DIR, "phase_3_four_versions.png")


def build_code():
    result = subprocess.run(
        ["make", "serial", "serial_tiled", "omp_static", "omp_tiled"],
        cwd=SRC_DIR, capture_output=True, text=True
    )
    if result.returncode != 0:
        raise RuntimeError(f"Build failed:\n{result.stderr}")


def run_case(executable, arguments, time_pattern, env):
    result = subprocess.run(
        [os.path.join(SRC_DIR, executable), *map(str, arguments)],
        capture_output=True, text=True, env=env
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"{executable} failed ({result.returncode}):\n{result.stderr}"
        )
    match = re.search(time_pattern, result.stdout)
    if not match:
        raise RuntimeError(f"Cannot parse time from {executable}:\n{result.stdout}")
    return float(match.group(1))


def plot(rows, n, threads, iters, runs, serial_b, omp_b):
    labels = ["Serial\nbaseline", f"Serial tiled\nB={serial_b}",
              "OpenMP\nbaseline", f"OpenMP tiled\nB={omp_b}"]
    medians = [row["median"] for row in rows]
    stddevs = [row["stddev"] for row in rows]
    colors = ["#4C78A8", "#72B7B2", "#F58518", "#E45756"]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(labels, medians, yerr=stddevs, capsize=5, color=colors,
                  width=0.66, edgecolor="#333333", linewidth=0.7)
    serial_base = medians[0]
    for bar, elapsed in zip(bars, medians):
        speedup = serial_base / elapsed
        ax.text(bar.get_x() + bar.get_width() / 2,
                elapsed + max(medians) * 0.035,
                f"{elapsed:.3f} s\n{speedup:.2f}x",
                ha="center", va="bottom", fontsize=10)

    ax.set_title(
        "Phase 3: Four-Version Performance Comparison\n"
        f"N={n}, Threads={threads}, Iterations={iters}, Median of {runs} runs"
    )
    ax.set_ylabel("Execution time (seconds, lower is better)")
    ax.set_ylim(0, max(medians) * 1.22)
    ax.grid(axis="y", linestyle=":", alpha=0.45)
    fig.tight_layout()
    fig.savefig(PNG_PATH, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=1024)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--iters", type=int, default=20000)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--serial-block", type=int, default=256)
    parser.add_argument("--omp-block", type=int, default=256)
    args = parser.parse_args()
    if args.n < 3 or min(args.threads, args.iters, args.runs,
                         args.serial_block, args.omp_block) < 1:
        parser.error("All parameters must be positive and N >= 3")

    build_code()
    env = os.environ.copy()
    env.setdefault("OMP_PROC_BIND", "close")
    env.setdefault("OMP_PLACES", "cores")
    env.setdefault("OMP_DYNAMIC", "false")
    serial_pattern = r"\[指标 1\] 串行耗时\s*:\s*([\d.]+)\s*秒"
    cases = [
        ("Serial", "jacobi_serial", [args.n, args.iters, 0], serial_pattern),
        ("Serial_Tiled", "jacobi_serial_tiled",
         [args.n, args.serial_block, args.iters], serial_pattern),
        ("OpenMP", "jacobi_omp_static",
         [args.n, args.threads, args.iters], r"Time=([\d.]+)"),
        ("OpenMP_Tiled", "jacobi_omp_tiled",
         [args.n, args.threads, args.iters, args.omp_block], r"Time=([\d.]+)"),
    ]
    samples = {name: [] for name, *_ in cases}

    # Rotate order to reduce systematic thermal/turbo bias.
    for run in range(args.runs):
        print(f"Run {run + 1}/{args.runs}", flush=True)
        ordered = cases[run % len(cases):] + cases[:run % len(cases)]
        for name, executable, arguments, pattern in ordered:
            samples[name].append(run_case(executable, arguments, pattern, env))

    rows = []
    for name, *_ in cases:
        values = samples[name]
        rows.append({
            "name": name,
            "median": statistics.median(values),
            "mean": statistics.mean(values),
            "stddev": statistics.stdev(values) if len(values) > 1 else 0.0,
            "samples": values,
        })

    os.makedirs(DATA_DIR, exist_ok=True)
    with open(CSV_PATH, "w", newline="", encoding="utf-8") as output:
        writer = csv.writer(output)
        writer.writerow(["Version", "N", "Threads", "Iters", "Runs",
                         "Median_s", "Mean_s", "Stddev_s", "Samples_s"])
        for row in rows:
            writer.writerow([
                row["name"], args.n,
                1 if row["name"].startswith("Serial") else args.threads,
                args.iters, args.runs, row["median"], row["mean"],
                row["stddev"], ";".join(map(str, row["samples"]))
            ])

    plot(rows, args.n, args.threads, args.iters, args.runs,
         args.serial_block, args.omp_block)
    print("\nVersion          Median(s)   Speedup vs Serial")
    for row in rows:
        print(f"{row['name']:<16} {row['median']:<11.4f} "
              f"{rows[0]['median'] / row['median']:.2f}x")
    print(f"CSV: {CSV_PATH}")
    print(f"PNG: {PNG_PATH}")


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as error:
        print(f"[!] {error}", file=sys.stderr)
        sys.exit(1)
