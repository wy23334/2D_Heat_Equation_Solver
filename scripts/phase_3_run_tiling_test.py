# -*- coding: utf-8 -*-
import os
import subprocess
import re
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))
SRC_DIR = os.path.join(PROJECT_ROOT, 'src')


def build_code():
    print("[*] 正在通过 Makefile 编译 C 代码...")
    result = subprocess.run(["make", "serial", "serial_tiled"], cwd=SRC_DIR, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"[!] 编译失败:\n{result.stderr}")
        sys.exit(1)
    print("[*] 编译成功！\n")


def run_executable(exe_name, N, B=None):
    exe_path = os.path.join(SRC_DIR, exe_name)

    cmd = [exe_path, str(N)]
    if B is not None:
        cmd.append(str(B))
        print(f"[-] 正在运行 {exe_name} (N={N}, B={B})...")
    else:
        print(f"[-] 正在运行 {exe_name} (N={N}, 基准版)...")

    result = subprocess.run(cmd, capture_output=True, text=True)

    time_match = re.search(r'\[指标 1\] 串行耗时\s*:\s*([\d\.]+)\s*秒', result.stdout)
    if time_match:
        return float(time_match.group(1))
    else:
        print(f"[!] 无法解析运行时间:\n{result.stdout}")
        sys.exit(1)


def main():
    build_code()

    N = 1024
    block_sizes = [32, 64, 128, 256]
    results = {}

    print(f"=== 🚀 开始分块与 SIMD 优化极限测试 (N={N}) ===")

    # 1. 测基准
    time_baseline = run_executable("jacobi_serial", N)

    # 2. 循环测各个 B 值
    for B in block_sizes:
        time_tiled = run_executable("jacobi_serial_tiled", N, B)
        speedup = time_baseline / time_tiled
        results[B] = (time_tiled, speedup)

    # 3. 打印报告
    print("\n" + "=" * 55)
    print(" 📊 多级分块测试结果报告 (N=1024)")
    print("=" * 55)
    print(f"基准串行耗时 (无分块) : {time_baseline:.4f} 秒\n")
    print(f"{'Block Size (B)':<15} | {'耗时 (秒)':<15} | {'加速比 (Speedup)':<15}")
    print("-" * 55)

    all_passed = True
    for B in block_sizes:
        t, s = results[B]
        marker = "✅" if s >= 1.5 else "⚠️"
        if s < 1.5: all_passed = False
        print(f"B = {B:<11} | {t:<13.4f} | {s:.2f}x {marker}")

    print("=" * 55)
    if all_passed:
        print("🎉 恭喜！所有分块大小均成功突破 1.5 倍加速目标！")
    else:
        print("💡 提示：虽然部分未达到 1.5x，但 SIMD 优化已显著降低了分块带来的额外开销。")


if __name__ == "__main__":
    main()