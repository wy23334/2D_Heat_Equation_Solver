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


def build_code():
    """调用 Makefile 确保可执行文件是最新的"""
    print("[*] 正在通过 Makefile 编译 C 代码...")
    result = subprocess.run(["make", "serial", "serial_tiled"], cwd=SRC_DIR, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"[!] 编译失败:\n{result.stderr}")
        sys.exit(1)
    print("[*] 编译成功！\n")


def run_perf(exe_name, N, B=None):
    """运行 perf 工具并兼容大小核架构提取硬件缓存数据"""
    exe_path = os.path.join(SRC_DIR, exe_name)
    cmd = ["perf", "stat", "-e", "cache-misses,cache-references", exe_path, str(N)]

    if B is not None:
        cmd.append(str(B))
        print(f"[*] 正在运行 {exe_name} (N={N}, B={B}) 并收集 perf 数据 (需要两三分钟)...")
    else:
        print(f"[*] 正在运行 {exe_name} (N={N}, 基准版) 并收集 perf 数据 (需要两三分钟)...")

    # 执行命令并捕获输出
    result = subprocess.run(cmd, stderr=subprocess.PIPE, stdout=subprocess.DEVNULL, text=True)
    output = result.stderr

    misses = 0
    refs = 0

    # 检查硬件限制
    if "<not supported>" in output:
        print(f"\n[!] 严重警告: perf 报告硬件计数器不可用！原始输出:\n{output}")
        sys.exit(1)

    # 解析 perf 数据，匹配大小核累加
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
        print(f"\n[!] 警告: 无法从 perf 解析出数据！原始输出:\n{output}")

    return refs, misses


def main():
    # 1. 编译代码并准备环境
    build_code()
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)

    N = 1024
    B = 64  # 使用上一轮测出的最佳分块大小

    # 2. 执行性能监控
    base_refs, base_miss = run_perf("jacobi_serial", N)
    tiled_refs, tiled_miss = run_perf("jacobi_serial_tiled", N, B)

    # 3. 终端可视化输出
    terminal_report = (
            f"=== 🎯 分块优化 Cache 性能对比表 (N={N}) ===\n"
            + "-" * 75 + "\n"
            + f"{'版本 (Version)':<25} | {'Cache Refs (缓存引用)':<20} | {'Cache Misses (未命中)':<20}\n"
            + "-" * 75 + "\n"
            + f"{'原版串行 (Baseline)':<25} | {base_refs:<20,} | {base_miss:<20,}\n"
            + f"{'分块优化 (Tiled B=64)':<25} | {tiled_refs:<20,} | {tiled_miss:<20,}\n"
            + "-" * 75 + "\n"
    )
    print("\n" + terminal_report)

    # 4. 写入 CSV 文件
    csv_filename = "tiling_cache_report.csv"
    csv_report_path = os.path.join(DATA_DIR, csv_filename)

    with open(csv_report_path, "w", encoding="utf-8") as f:
        f.write("Version,Cache_Refs,Cache_Misses\n")
        f.write(f"Baseline,{base_refs},{base_miss}\n")
        f.write(f"Tiled_B64,{tiled_refs},{tiled_miss}\n")

    print(f"[*] 报告已成功保存至: {csv_report_path}")


if __name__ == "__main__":
    main()