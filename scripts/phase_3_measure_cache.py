# -*- coding: utf-8 -*-
"""Phase 3: 使用 perf 收集最终四种版本的缓存统计数据。"""

import argparse
import csv
import os
import statistics
import subprocess
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC = os.path.join(ROOT, "src")
DATA = os.path.join(ROOT, "data")
OUTPUT = os.path.join(DATA, "phase_3_cache_miss.csv")


def build():
    result = subprocess.run(
        ["make", "serial_colmajor", "serial_tiled", "omp_colmajor", "omp_tiled"],
        cwd=SRC, capture_output=True, text=True
    )
    if result.returncode:
        raise RuntimeError(result.stderr)


def perf_stat(executable, arguments, env):
    command = [
        "perf", "stat", "-x,", "-e", "cache-references,cache-misses",
        os.path.join(SRC, executable), *map(str, arguments)
    ]
    result = subprocess.run(command, stdout=subprocess.DEVNULL,
                            stderr=subprocess.PIPE, text=True, env=env)
    if result.returncode:
        raise RuntimeError(f"perf failed for {executable}:\n{result.stderr}")

    references = 0
    misses = 0
    for line in result.stderr.splitlines():
        fields = line.split(",")
        if len(fields) < 3 or fields[0] in ("", "<not counted>", "<not supported>"):
            continue
        try:
            value = int(fields[0])
        except ValueError:
            continue
        event = fields[2]
        if "cache-references" in event:
            references += value
        elif "cache-misses" in event:
            misses += value
    if references == 0:
        raise RuntimeError(f"No hardware cache counters parsed:\n{result.stderr}")
    return references, misses


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=1024)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--serial-block", type=int, default=256)
    parser.add_argument("--omp-block", type=int, default=256)
    args = parser.parse_args()
    build()

    env = os.environ.copy()
    env.setdefault("OMP_PROC_BIND", "close")
    env.setdefault("OMP_PLACES", "cores")
    env.setdefault("OMP_DYNAMIC", "false")
    cases = [
        ("Column_Major_Serial", "jacobi_serial_colmajor",
         [args.n, args.iters]),
        (f"Serial_Tiled_B{args.serial_block}", "jacobi_serial_tiled",
         [args.n, args.serial_block, args.iters]),
        ("Column_Major_OpenMP", "jacobi_omp_colmajor",
         [args.n, args.threads, args.iters]),
        (f"OpenMP_Tiled_B{args.omp_block}", "jacobi_omp_tiled",
         [args.n, args.threads, args.iters, args.omp_block]),
    ]
    samples = {name: [] for name, *_ in cases}
    for run in range(args.runs):
        print(f"Run {run + 1}/{args.runs}", flush=True)
        offset = run % len(cases)
        ordered = cases[offset:] + cases[:offset]
        for name, executable, arguments in ordered:
            samples[name].append(perf_stat(executable, arguments, env))

    os.makedirs(DATA, exist_ok=True)
    with open(OUTPUT, "w", newline="", encoding="utf-8") as output:
        writer = csv.writer(output)
        writer.writerow([
            "Version", "N", "Threads", "Iters", "Runs",
            "Cache_References_Median", "Cache_Misses_Median", "Miss_Rate_pct",
            "Reference_Samples", "Miss_Samples"
        ])
        for name, *_ in cases:
            refs = [item[0] for item in samples[name]]
            misses = [item[1] for item in samples[name]]
            median_refs = statistics.median(refs)
            median_misses = statistics.median(misses)
            miss_rate = 100.0 * median_misses / median_refs
            writer.writerow([
                name, args.n, 1 if "OpenMP" not in name else args.threads,
                args.iters, args.runs, median_refs, median_misses, miss_rate,
                ";".join(map(str, refs)), ";".join(map(str, misses))
            ])
            print(f"{name:<26} refs={median_refs:>12,.0f}  "
                  f"misses={median_misses:>12,.0f}  rate={miss_rate:>6.2f}%")
    print(f"CSV: {OUTPUT}")


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as error:
        print(f"[ERROR] {error}", file=sys.stderr)
        sys.exit(1)
