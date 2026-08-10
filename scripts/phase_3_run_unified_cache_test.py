# -*- coding: utf-8 -*-
import os
import subprocess
import re
import sys

# ==========================================
# 自动解析项目目录结构
# ==========================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))
SRC_DIR = os.path.join(PROJECT_ROOT, 'src')
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')


def create_unified_files():
    """统一提取并生成三份基准完全一致的 C 代码"""
    serial_src = os.path.join(SRC_DIR, "jacobi_serial.c")
    tiled_src = os.path.join(SRC_DIR, "jacobi_serial_tiled.c")

    for f in [serial_src, tiled_src]:
        if not os.path.exists(f):
            print(f"[!] 找不到源文件: {f}")
            sys.exit(1)

    with open(serial_src, 'r', encoding='utf-8') as f:
        serial_code = f.read()
    with open(tiled_src, 'r', encoding='utf-8') as f:
        tiled_code = f.read()

    # 🌟 统一将 MAX_ITER 强制设为 1000 次，确保绝对公平的比较基准
    serial_code = re.sub(r'#define MAX_ITER \d+', '#define MAX_ITER 1000', serial_code)
    serial_code = re.sub(r'FILE \*fp = fopen\(.*?\);', 'FILE *fp = NULL;', serial_code)

    tiled_code = re.sub(r'#define MAX_ITER \d+', '#define MAX_ITER 1000', tiled_code)
    tiled_code = re.sub(r'FILE \*fp = fopen\(.*?\);', 'FILE *fp = NULL;', tiled_code)

    # 1. 正常的行优先基准版本
    base_file = os.path.join(SCRIPT_DIR, 'unified_baseline.c')
    with open(base_file, 'w', encoding='utf-8') as f:
        f.write(serial_code)

    # 2. 糟糕的列优先版本
    bad_code = serial_code.replace(
        "for (int i = 1; i < N - 1; i++) {\n            for (int j = 1; j < N - 1; j++) {",
        "for (int j = 1; j < N - 1; j++) {\n            for (int i = 1; i < N - 1; i++) {"
    )
    bad_file = os.path.join(SCRIPT_DIR, 'unified_bad.c')
    with open(bad_file, 'w', encoding='utf-8') as f:
        f.write(bad_code)

    # 3. 分块优化版本
    tiled_file = os.path.join(SCRIPT_DIR, 'unified_tiled.c')
    with open(tiled_file, 'w', encoding='utf-8') as f:
        f.write(tiled_code)

    return base_file, bad_file, tiled_file


def compile_code(c_file, exe_name):
    """使用完全相同的极限编译选项进行编译"""
    exe_path = os.path.join(SCRIPT_DIR, exe_name)
    print(f"[*] 正在编译 {c_file} -> {exe_path} ...")

    # 统一赋予 -O3 -march=native -ffast-math，保证编译器优化待遇一致
    cmd = ["gcc", "-O3", "-Wall", "-march=native", "-ffast-math", c_file, "-o", exe_path, "-lm"]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"[!] 编译失败:\n{result.stderr}")
        sys.exit(1)
    return exe_path


def run_perf(executable_path, N, B=None):
    """运行 perf 工具提取 Cache 指标"""
    cmd = ["perf", "stat", "-e", "cache-misses,cache-references", executable_path, str(N)]
    if B is not None:
        cmd.append(str(B))
        print(f"[*] 正在运行 {os.path.basename(executable_path)} (N={N}, B={B}) ...")
    else:
        print(f"[*] 正在运行 {os.path.basename(executable_path)} (N={N}) ...")

    result = subprocess.run(cmd, stderr=subprocess.PIPE, stdout=subprocess.DEVNULL, text=True)
    output = result.stderr

    misses = 0
    refs = 0

    if "<not supported>" in output:
        print(f"\n[!] 严重警告: perf 报告硬件计数器不可用！原始输出:\n{output}")
        sys.exit(1)

    for line in output.split('\n'):
        if 'cache-misses' in line:
            match = re.search(r'^\s*([\d,]+)\s+', line)
            if match:
                misses += int(match.group(1).replace(',', ''))
        elif 'cache-references' in line:
            match = re.search(r'^\s*([\d,]+)\s+', line)
            if match:
                refs += int(match.group(1).replace(',', ''))

    if misses == 0 and refs == 0:
        print(f"\n[!] 警告: 无法解析出数据，原始输出:\n{output}")

    return refs, misses


def main():
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)

    N = 1024
    B = 64

    print("=== 阶段 1: 统一代码基准 (MAX_ITER=1000) ===")
    base_c, bad_c, tiled_c = create_unified_files()

    print("\n=== 阶段 2: 公平编译所有版本 ===")
    base_exe = compile_code(base_c, "unified_baseline_exe")
    bad_exe = compile_code(bad_c, "unified_bad_exe")
    tiled_exe = compile_code(tiled_c, "unified_tiled_exe")

    print("\n=== 阶段 3: 执行硬件性能监控 (perf) ===")
    base_refs, base_miss = run_perf(base_exe, N)
    bad_refs, bad_miss = run_perf(bad_exe, N)
    tiled_refs, tiled_miss = run_perf(tiled_exe, N, B)

    # ---------------- 终端可视化输出 ----------------
    terminal_report = (
            f"=== 🎯 终极大一统 Cache 性能对比表 (N={N}, MAX_ITER=1000) ===\n"
            + "-" * 85 + "\n"
            + f"{'访问/优化模式 (Version)':<25} | {'Cache Refs (缓存引用)':<20} | {'Cache Misses (未命中)':<20}\n"
            + "-" * 85 + "\n"
            + f"{'1. 原版基准 (Row-Major)':<25} | {base_refs:<20,} | {base_miss:<20,}\n"
            + f"{'2. 反面教材 (Col-Major)':<25} | {bad_refs:<20,} | {bad_miss:<20,}\n"
            + f"{'3. 终极形态 (Tiled B=64)':<25} | {tiled_refs:<20,} | {tiled_miss:<20,}\n"
            + "-" * 85 + "\n"
    )
    print("\n" + terminal_report)

    # ---------------- 写入 CSV 文件 ----------------
    csv_filename = "unified_cache_report.csv"
    csv_report_path = os.path.join(DATA_DIR, csv_filename)

    with open(csv_report_path, "w", encoding="utf-8") as f:
        f.write("Version,Cache_Refs,Cache_Misses\n")
        f.write(f"1_Baseline_Row_Major,{base_refs},{base_miss}\n")
        f.write(f"2_Bad_Col_Major,{bad_refs},{bad_miss}\n")
        f.write(f"3_Optimized_Tiled_B64,{tiled_refs},{tiled_miss}\n")

    print(f"[*] 报告已成功保存至: {csv_report_path}")

    # 清理所有的临时文件
    for f in [base_c, bad_c, tiled_c, base_exe, bad_exe, tiled_exe]:
        if os.path.exists(f):
            os.remove(f)
    print("[*] 测试完成，已清理临时文件。")


if __name__ == "__main__":
    main()