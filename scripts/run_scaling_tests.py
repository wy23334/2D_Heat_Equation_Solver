# /home/wy/Projects/2D_Heat_Equation_Solver/scripts/run_scaling_tests.py
# phase 2
import subprocess
import os

# 配置 C 语言可执行文件的路径
EXECUTABLE = "../src/jacobi_omp"

def run_c_program(N, threads, max_iters):
    """调用 C 程序并解析返回的耗时和实际迭代次数"""
    cmd = [EXECUTABLE, str(N), str(threads), str(max_iters)]
    result = subprocess.run(cmd, capture_output=True, text=True)

    for line in result.stdout.split('\n'):
        if line.startswith("RESULT"):
            # 格式: RESULT,N=1024,Threads=4,Iters=200000,Time=1.234,Center=56.25
            parts = line.split(',')
            actual_iters = int(parts[3].split('=')[1])
            time_val = float(parts[4].split('=')[1])
            return time_val, actual_iters

    print(f"❌ 运行失败 (N={N}, Threads={threads}):\n{result.stdout}")
    return None, None

print("🚀 开始进行 OpenMP 性能自动化评测...\n")

# 🌟 核心修复：设置统一的测试迭代次数，避免小网格提前收敛！
# 5000 次足够测出稳定的时间，同时能让整个测试在十几秒内飞速跑完
BENCHMARK_ITERS = 5000

# ==========================================
# [Req 4] 强标扩展性测试 (Strong Scaling)
# ==========================================
print("--- 📊 Req 4: 强标测试 (固定 N=1024) ---")
N_strong = 1024
threads_list = [1, 2, 4, 8]
t1_strong = None

print(f"{'Threads':<8} | {'Iters':<8} | {'Time (s)':<10} | {'Speedup':<10}")
print("-" * 42)

for t in threads_list:
    time_val, iters = run_c_program(N_strong, t, BENCHMARK_ITERS)
    if time_val is None: continue

    if t == 1:
        t1_strong = time_val
        speedup = 1.0
    else:
        speedup = t1_strong / time_val

    print(f"{t:<8} | {iters:<8} | {time_val:<10.4f} | {speedup:<10.2f}")

print("\n")

# ==========================================
# [Req 5] 弱标扩展性测试 (Weak Scaling)
# ==========================================
print("--- 📊 Req 5: 弱标测试 (人均任务固定) ---")
weak_scaling_pairs = [
    (1, 256),
    (2, 362),
    (4, 512),
    (8, 724)
]
t1_weak = None

print(f"{'Threads':<8} | {'Grid (N)':<10} | {'Iters':<8} | {'Time (s)':<10} | {'Efficiency'}")
print("-" * 56)

for t, N_weak in weak_scaling_pairs:
    time_val, iters = run_c_program(N_weak, t, BENCHMARK_ITERS)
    if time_val is None: continue

    if t == 1:
        t1_weak = time_val
        efficiency = 100.0
    else:
        efficiency = (t1_weak / time_val) * 100.0

    print(f"{t:<8} | {N_weak:<10} | {iters:<8} | {time_val:<10.4f} | {efficiency:.2f}%")