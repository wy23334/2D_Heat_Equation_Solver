# -*- coding: utf-8 -*-
import argparse
import csv
import os
import re
import statistics
import subprocess
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
DATA_DIR = os.path.join(PROJECT_ROOT, "data")


def build_code():
    result = subprocess.run(
        ["make", "omp_naive", "omp_static", "omp_tiled"], cwd=SRC_DIR,
        capture_output=True, text=True
    )
    if result.returncode != 0:
        print(f"[!] 编译失败:\n{result.stderr}")
        sys.exit(1)


def run_executable(exe_name, n, threads, iters, block_size=None):
    cmd = [os.path.join(SRC_DIR, exe_name), str(n), str(threads), str(iters)]
    if block_size is not None:
        cmd.append(str(block_size))

    env = os.environ.copy()
    env.setdefault("OMP_PROC_BIND", "close")
    env.setdefault("OMP_PLACES", "cores")
    env.setdefault("OMP_DYNAMIC", "false")
    result = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if result.returncode != 0:
        print(f"[!] {exe_name} 运行失败 ({result.returncode}):\n{result.stderr}")
        sys.exit(1)

    match = re.search(r"Time=([\d.]+)", result.stdout)
    if not match:
        print(f"[!] 无法解析运行时间:\n{result.stdout}")
        sys.exit(1)
    return float(match.group(1))


def main():
    parser = argparse.ArgumentParser(description="OpenMP Jacobi tiling benchmark")
    parser.add_argument("--n", type=int, default=1024)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--iters", type=int, default=20000,
                        help="正式验收可设为 200000")
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--baseline", choices=("optimized", "naive"),
                        default="optimized")
    args = parser.parse_args()
    if args.n < 3 or min(args.threads, args.iters, args.runs) < 1:
        parser.error("参数必须为正数，且 N >= 3")

    build_code()
    block_sizes = [32, 64, 128, 256]
    baseline_exe = ("jacobi_omp_naive" if args.baseline == "naive"
                    else "jacobi_omp_static")
    base_times = []
    tiled_times = {b: [] for b in block_sizes}

    print(f"=== OpenMP + 分块测试 (N={args.n}, Threads={args.threads}, "
          f"Iters={args.iters}, Runs={args.runs}) ===")

    # 每轮重测 baseline，并旋转 tile 顺序，降低温度和睿频偏差。
    for run in range(args.runs):
        print(f"Run {run + 1}/{args.runs}", flush=True)
        base_times.append(run_executable(
            baseline_exe, args.n, args.threads, args.iters
        ))
        offset = run % len(block_sizes)
        order = block_sizes[offset:] + block_sizes[:offset]
        for block_size in order:
            tiled_times[block_size].append(run_executable(
                "jacobi_omp_tiled", args.n, args.threads, args.iters, block_size
            ))

    base_median = statistics.median(base_times)
    rows = []
    print("\n" + "=" * 88)
    print(f"OpenMP {args.baseline} 基线中位耗时: {base_median:.6f} s")
    print(f"{'Block':<10} | {'Median(s)':<12} | {'Mean(s)':<12} | "
          f"{'Stddev':<12} | {'Speedup':<10} | {'Improvement'}")
    print("-" * 88)

    for block_size in block_sizes:
        samples = tiled_times[block_size]
        median_t = statistics.median(samples)
        mean_t = statistics.mean(samples)
        std_t = statistics.stdev(samples) if len(samples) > 1 else 0.0
        speedup = base_median / median_t
        improvement = (speedup - 1.0) * 100.0
        marker = "达标" if improvement >= 20.0 else "未达20%"
        print(f"B={block_size:<8} | {median_t:<12.6f} | {mean_t:<12.6f} | "
              f"{std_t:<12.6f} | {speedup:<10.3f} | {improvement:+.1f}% {marker}")
        rows.append([
            args.n, args.threads, args.iters, args.runs, block_size,
            base_median, median_t, mean_t, std_t, speedup, improvement
        ])
    print("=" * 88)

    os.makedirs(DATA_DIR, exist_ok=True)
    suffix = "naive" if args.baseline == "naive" else "optimized"
    report = os.path.join(DATA_DIR, f"omp_tiling_benchmark_{suffix}.csv")
    with open(report, "w", newline="", encoding="utf-8") as output:
        writer = csv.writer(output)
        writer.writerow([
            "N", "Threads", "Iters", "Runs", "Block_Size",
            "Baseline_Median_s", "Tiled_Median_s", "Tiled_Mean_s",
            "Tiled_Stddev_s", "Speedup", "Improvement_pct"
        ])
        writer.writerows(rows)
    print(f"数据已保存至: {report}")


if __name__ == "__main__":
    main()
