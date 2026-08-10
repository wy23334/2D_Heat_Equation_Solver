# -*- coding: utf-8 -*-
"""Plot the Phase 3 OpenMP tiling benchmark from its CSV report."""

import csv
import os

import matplotlib.pyplot as plt


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
CSV_PATH = os.path.join(DATA_DIR, "omp_tiling_benchmark.csv")
PNG_PATH = os.path.join(DATA_DIR, "omp_tiling_comparison.png")


def load_results():
    with open(CSV_PATH, newline="", encoding="utf-8") as source:
        rows = list(csv.DictReader(source))
    if not rows:
        raise RuntimeError(f"No benchmark data found in {CSV_PATH}")

    baseline = float(rows[0]["Baseline_Median_s"])
    labels = ["OpenMP\nbaseline"]
    times = [baseline]
    improvements = [0.0]
    for row in rows:
        labels.append(f"Tiled\nB={row['Block_Size']}")
        times.append(float(row["Tiled_Median_s"]))
        improvements.append(float(row["Improvement_pct"]))
    return rows[0], labels, times, improvements


def main():
    config, labels, times, improvements = load_results()
    baseline = times[0]
    target_time = baseline / 1.2
    best_index = min(range(1, len(times)), key=times.__getitem__)

    colors = ["#3274A1"] + ["#A8B0B8"] * (len(times) - 1)
    colors[best_index] = "#E1812C"

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(labels, times, color=colors, width=0.68, edgecolor="#333333",
                  linewidth=0.7)
    ax.axhline(target_time, color="#C44E52", linestyle="--", linewidth=1.6,
               label=f"20% speedup target ({target_time:.3f} s)")

    for index, (bar, elapsed, improvement) in enumerate(
            zip(bars, times, improvements)):
        if index == 0:
            annotation = f"{elapsed:.3f} s\nbaseline"
        else:
            annotation = f"{elapsed:.3f} s\n{improvement:+.1f}%"
        ax.text(bar.get_x() + bar.get_width() / 2, elapsed + max(times) * 0.018,
                annotation, ha="center", va="bottom", fontsize=10)

    ax.set_title(
        "OpenMP Jacobi Tiling Performance\n"
        f"N={config['N']}, Threads={config['Threads']}, "
        f"Iterations={config['Iters']}, Median of {config['Runs']} runs"
    )
    ax.set_ylabel("Execution time (seconds, lower is better)")
    ax.set_ylim(0, max(times) * 1.24)
    ax.grid(axis="y", linestyle=":", alpha=0.45)
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(PNG_PATH, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Chart saved to: {PNG_PATH}")


if __name__ == "__main__":
    main()
