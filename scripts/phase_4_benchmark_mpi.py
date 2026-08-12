# -*- coding: utf-8 -*-
"""Phase 4: 运行 MPI 强标实验并生成 CSV 与加速比图。"""

import argparse
import csv
import os
import re
import statistics
import subprocess
import sys

import matplotlib.pyplot as plt

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC = os.path.join(ROOT, "src")
DATA = os.path.join(ROOT, "data")
CSV_PATH = os.path.join(DATA, "phase_4_mpi_scaling.csv")
PNG_PATH = os.path.join(DATA, "phase_4_mpi_scaling.png")


def build():
    result = subprocess.run(["make", "mpi"], cwd=SRC,
                            capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(result.stderr)


def run_once(processes, n, iters, env):
    command = [
        "mpirun", "--bind-to", "core", "--map-by", "core",
        "-np", str(processes), os.path.join(SRC, "jacobi_mpi"),
        str(n), str(iters)
    ]
    result = subprocess.run(command, capture_output=True, text=True, env=env)
    if result.returncode:
        raise RuntimeError(
            f"mpirun -np {processes} failed:\n{result.stdout}\n{result.stderr}"
        )
    match = re.search(
        r"RESULT,N=(\d+),Processes=(\d+),Iters=(\d+),Time=([\d.]+),"
        r"Center=([\d.]+),MaxDiff=([\deE+.-]+)", result.stdout
    )
    if not match:
        raise RuntimeError(f"Cannot parse MPI output:\n{result.stdout}")
    return {
        "iters": int(match.group(3)),
        "time": float(match.group(4)),
        "center": float(match.group(5)),
        "max_diff": float(match.group(6)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=2048)
    parser.add_argument("--iters", type=int, default=2000)
    parser.add_argument("--runs", type=int, default=3)
    args = parser.parse_args()
    if args.n < 10 or min(args.iters, args.runs) < 1:
        parser.error("Invalid benchmark parameters")

    build()
    os.makedirs(DATA, exist_ok=True)
    env = os.environ.copy()
    process_counts = [1, 2, 4, 8]
    samples = {p: [] for p in process_counts}

    for run in range(args.runs):
        print(f"Run {run + 1}/{args.runs}", flush=True)
        offset = run % len(process_counts)
        order = process_counts[offset:] + process_counts[:offset]
        for processes in order:
            samples[processes].append(run_once(
                processes, args.n, args.iters, env
            ))

    medians = {
        p: statistics.median(item["time"] for item in samples[p])
        for p in process_counts
    }
    baseline = medians[1]
    rows = []
    for processes in process_counts:
        times = [item["time"] for item in samples[processes]]
        centers = [item["center"] for item in samples[processes]]
        rows.append({
            "Processes": processes,
            "N": args.n,
            "Requested_Iters": args.iters,
            "Actual_Iters": samples[processes][0]["iters"],
            "Runs": args.runs,
            "Median_Time_s": medians[processes],
            "Mean_Time_s": statistics.mean(times),
            "Stddev_s": statistics.stdev(times) if len(times) > 1 else 0.0,
            "Speedup": baseline / medians[processes],
            "Efficiency_pct": 100.0 * baseline / medians[processes] / processes,
            "Center": statistics.median(centers),
            "Time_Samples_s": ";".join(map(str, times)),
        })

    with open(CSV_PATH, "w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    speedups = [row["Speedup"] for row in rows]
    fig, ax = plt.subplots(figsize=(8.5, 5.8))
    ax.plot(process_counts, process_counts, "k--", label="Ideal speedup")
    ax.plot(process_counts, speedups, color="#4C78A8", marker="o",
            linewidth=2.2, markersize=7, label="Measured MPI speedup")
    # [Phase 4 - Req 4] 验收下界：1 进程为 1x，8 进程至少达到 5x。
    # 与 Phase 2 的 Minimum Requirement 画法保持一致。
    ax.plot([1, 8], [1.0, 5.0], color="#C44E52", linestyle="--",
            linewidth=2.0, label="Minimum requirement (8 processes >= 5.0x)")
    for p, speedup in zip(process_counts, speedups):
        ax.annotate(f"{speedup:.2f}x", (p, speedup),
                    xytext=(0, 9), textcoords="offset points", ha="center")
    ax.set_title(
        f"MPI Strong Scaling (N={args.n}, Iterations={args.iters}, "
        f"Median of {args.runs} runs)"
    )
    ax.set_xlabel("MPI processes")
    ax.set_ylabel("Speedup over 1 process")
    ax.set_xticks(process_counts)
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend()
    fig.tight_layout()
    fig.savefig(PNG_PATH, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print("\nProcesses  Median(s)  Speedup  Efficiency")
    for row in rows:
        print(f"{row['Processes']:<9}  {row['Median_Time_s']:<9.4f}  "
              f"{row['Speedup']:<7.2f}x  {row['Efficiency_pct']:.1f}%")
    print(f"CSV: {CSV_PATH}")
    print(f"PNG: {PNG_PATH}")


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as error:
        print(f"[ERROR] {error}", file=sys.stderr)
        sys.exit(1)
