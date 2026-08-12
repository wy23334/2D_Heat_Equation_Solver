# -*- coding: utf-8 -*-
"""Phase 3: 测试 tile size 和四种版本，并生成最终 CSV 与柱状图。"""

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
TIME_CN = r"\[指标 1\] 串行耗时\s*:\s*([\d.]+)\s*秒"
TIME_RESULT = r"Time=([\d.]+)"


def build():
    result = subprocess.run(
        ["make", "serial_colmajor", "serial_tiled", "omp_colmajor", "omp_tiled"],
        cwd=SRC, capture_output=True, text=True
    )
    if result.returncode:
        raise RuntimeError(result.stderr)


def execute(executable, arguments, pattern, env):
    result = subprocess.run(
        [os.path.join(SRC, executable), *map(str, arguments)],
        capture_output=True, text=True, env=env
    )
    if result.returncode:
        raise RuntimeError(f"{executable} failed:\n{result.stderr}")
    match = re.search(pattern, result.stdout)
    if not match:
        raise RuntimeError(f"Cannot parse {executable}:\n{result.stdout}")
    return float(match.group(1))


def summary(values):
    return {
        "median": statistics.median(values),
        "mean": statistics.mean(values),
        "stddev": statistics.stdev(values) if len(values) > 1 else 0.0,
        "samples": values,
    }


def save_csv(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def plot_tile_sizes(serial_stats, omp_stats, n, threads, iters, runs, path):
    blocks = list(serial_stats)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.8))
    panels = [
        (axes[0], serial_stats, "Serial tiled"),
        (axes[1], omp_stats, f"OpenMP tiled ({threads} threads)"),
    ]
    for ax, stats, title in panels:
        medians = [stats[b]["median"] for b in blocks]
        errors = [stats[b]["stddev"] for b in blocks]
        best = min(range(len(blocks)), key=medians.__getitem__)
        colors = ["#A8B0B8"] * len(blocks)
        colors[best] = "#E1812C"
        bars = ax.bar([f"B={b}" for b in blocks], medians, yerr=errors,
                      capsize=4, color=colors, edgecolor="#333333", width=0.68)
        for index, (bar, value) in enumerate(zip(bars, medians)):
            suffix = "\nBEST" if index == best else ""
            ax.text(bar.get_x() + bar.get_width() / 2,
                    value + max(medians) * 0.035, f"{value:.3f} s{suffix}",
                    ha="center", va="bottom", fontsize=9)
        ax.set_title(title)
        ax.set_ylabel("Median execution time (s)")
        ax.set_ylim(0, max(medians) * 1.22)
        ax.grid(axis="y", linestyle=":", alpha=0.45)
    fig.suptitle(
        f"Phase 3: Tile-Size Selection (N={n}, Iterations={iters}, "
        f"Median of {runs} runs)", fontsize=15
    )
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_four_versions(stats, serial_b, omp_b, n, threads, iters, runs, path):
    names = ["Column-major\nserial baseline", f"Serial tiled\nB={serial_b}",
             "Column-major\nOpenMP baseline", f"OpenMP tiled\nB={omp_b}"]
    medians = [item["median"] for item in stats]
    errors = [item["stddev"] for item in stats]
    colors = ["#B279A2", "#72B7B2", "#4C78A8", "#F58518"]
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(names, medians, yerr=errors, capsize=5, color=colors,
                  edgecolor="#333333", width=0.66)
    # 串行版本相对于列优先串行基准计算加速比；OpenMP 版本相对于
    # 列优先 OpenMP 基准计算加速比，避免跨线程数比较造成误读。
    baselines = [medians[0], medians[0], medians[2], medians[2]]
    for bar, value, baseline in zip(bars, medians, baselines):
        ax.text(bar.get_x() + bar.get_width() / 2,
                value + max(medians) * 0.035,
                f"{value:.3f} s\n{baseline / value:.2f}x",
                ha="center", va="bottom", fontsize=10)
    ax.set_title(
        "Phase 3: Four-Version Performance Comparison\n"
        f"N={n}, Threads={threads}, Iterations={iters}, Median of {runs} runs"
    )
    ax.set_ylabel("Execution time (seconds, lower is better)")
    ax.set_ylim(0, max(medians) * 1.22)
    ax.grid(axis="y", linestyle=":", alpha=0.45)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=1024)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--iters", type=int, default=2000)
    parser.add_argument("--runs", type=int, default=3)
    args = parser.parse_args()
    if args.n < 3 or min(args.threads, args.iters, args.runs) < 1:
        parser.error("Invalid benchmark parameters")

    build()
    os.makedirs(DATA, exist_ok=True)
    env = os.environ.copy()
    env.setdefault("OMP_PROC_BIND", "close")
    env.setdefault("OMP_PLACES", "cores")
    env.setdefault("OMP_DYNAMIC", "false")
    # 按题目要求的候选集合选择最佳 tile。B=1024 等于整个网格
    # 宽度，会退化成不分块，因此不作为有效 tile size 候选。
    blocks = [32, 64, 128, 256]
    serial_samples = {b: [] for b in blocks}
    omp_samples = {b: [] for b in blocks}
    colmajor_samples = []
    colmajor_omp_samples = []

    for run in range(args.runs):
        print(f"Run {run + 1}/{args.runs}", flush=True)
        colmajor_samples.append(execute(
            "jacobi_serial_colmajor", [args.n, args.iters], TIME_RESULT, env))
        colmajor_omp_samples.append(execute(
            "jacobi_omp_colmajor", [args.n, args.threads, args.iters],
            TIME_RESULT, env))
        offset = run % len(blocks)
        for block in blocks[offset:] + blocks[:offset]:
            serial_samples[block].append(execute(
                "jacobi_serial_tiled", [args.n, block, args.iters], TIME_CN, env))
            omp_samples[block].append(execute(
                "jacobi_omp_tiled", [args.n, args.threads, args.iters, block],
                TIME_RESULT, env))

    serial_stats = {b: summary(serial_samples[b]) for b in blocks}
    omp_stats = {b: summary(omp_samples[b]) for b in blocks}
    serial_best = min(blocks, key=lambda b: serial_stats[b]["median"])
    omp_best = min(blocks, key=lambda b: omp_stats[b]["median"])
    four_stats = [summary(colmajor_samples), serial_stats[serial_best],
                  summary(colmajor_omp_samples), omp_stats[omp_best]]

    tile_rows = []
    for version, stats in (("Serial_Tiled", serial_stats),
                           ("OpenMP_Tiled", omp_stats)):
        for block in blocks:
            item = stats[block]
            tile_rows.append({
                "Version": version, "Block_Size": block, "N": args.n,
                "Threads": 1 if version == "Serial_Tiled" else args.threads,
                "Iters": args.iters, "Runs": args.runs,
                "Median_s": item["median"], "Mean_s": item["mean"],
                "Stddev_s": item["stddev"],
                "Samples_s": ";".join(map(str, item["samples"])),
            })
    save_csv(os.path.join(DATA, "phase_3_tile_size_selection.csv"), tile_rows)

    labels = ["Column_Major_Serial", f"Serial_Tiled_B{serial_best}",
              "Column_Major_OpenMP", f"OpenMP_Tiled_B{omp_best}"]
    four_rows = []
    for label, item in zip(labels, four_stats):
        four_rows.append({
            "Version": label, "N": args.n,
            "Threads": 1 if "OpenMP" not in label else args.threads,
            "Iters": args.iters, "Runs": args.runs,
            "Median_s": item["median"], "Mean_s": item["mean"],
            "Stddev_s": item["stddev"],
            "Samples_s": ";".join(map(str, item["samples"])),
        })
    save_csv(os.path.join(DATA, "phase_3_four_versions_final.csv"), four_rows)

    plot_tile_sizes(serial_stats, omp_stats, args.n, args.threads, args.iters,
                    args.runs, os.path.join(DATA, "phase_3_tile_size_selection.png"))
    plot_four_versions(four_stats, serial_best, omp_best, args.n, args.threads,
                       args.iters, args.runs,
                       os.path.join(DATA, "phase_3_four_versions_final.png"))
    print(f"Best serial tile size: B={serial_best}")
    print(f"Best OpenMP tile size: B={omp_best}")


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as error:
        print(f"[ERROR] {error}", file=sys.stderr)
        sys.exit(1)
