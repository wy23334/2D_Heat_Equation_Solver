# /home/wy/Projects/2D_Heat_Equation_Solver/scripts/phase_2_run_scaling_tests.py
# phase 2 - 批量运行求均值版
import subprocess
import os
import csv
import matplotlib.pyplot as plt

# ==========================================
# 🌟 评测参数配置
# ==========================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "../data"))
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

SCHEDULES = ["static", "dynamic", "guided"]
BENCHMARK_ITERS = 5000

# 🌟 新增：批量运行次数（你可以修改为 10、50 或 100）
NUM_RUNS = 10


def run_c_program(executable, N, threads, max_iters):
    """调用 C 程序并解析返回的耗时和实际迭代次数（单次基础函数）"""
    if not os.path.exists(executable):
        return None, None

    cmd = [executable, str(N), str(threads), str(max_iters)]
    result = subprocess.run(cmd, capture_output=True, text=True)

    for line in result.stdout.split('\n'):
        if line.startswith("RESULT"):
            parts = line.split(',')
            actual_iters = int(parts[3].split('=')[1])
            time_val = float(parts[4].split('=')[1])
            return time_val, actual_iters
    return None, None


def get_average_time(executable, N, threads, max_iters, runs):
    """🌟 核心新增：连续运行指定次数，计算平均耗时"""
    times = []
    actual_iters = 0

    # 打印一个简单的进度条，防止 100 次看着像卡死了
    print(f"      [测算 {runs} 次] ", end="", flush=True)

    for i in range(runs):
        t, iters = run_c_program(executable, N, threads, max_iters)
        if t is not None:
            times.append(t)
            actual_iters = iters

        # 每跑完 10% 打印一个点
        if runs >= 10 and (i + 1) % (runs // 10) == 0:
            print(".", end="", flush=True)

    if not times:
        print(" ❌ 全部失败")
        return None, None

    avg_t = sum(times) / len(times)
    print(" 完成", flush=True)
    return avg_t, actual_iters


print(f"🚀 开始进行 OpenMP 综合性能自动化评测 (批量 {NUM_RUNS} 次求均值版)...\n")

print("🔥 正在进行全局 CPU 预热 (唤醒硬件最高性能状态)...")
warmup_exe = os.path.join(SCRIPT_DIR, f"../src/jacobi_omp_static")
if os.path.exists(warmup_exe):
    subprocess.run([warmup_exe, "1024", "8", "1000"], capture_output=True)
print("🔥 预热完成，正式开始评测！(预计耗时可能较长，请耐心等待)\n")

plot_data = {
    sched: {
        'strong_threads': [], 'strong_speedup': [],
        'weak_threads': [], 'weak_efficiency': []
    } for sched in SCHEDULES
}
csv_rows = []

N_strong = 1024
threads_list = [1, 2, 4, 8]
weak_scaling_pairs = [(1, 256), (2, 362), (4, 512), (8, 724)]

for sched in SCHEDULES:
    executable = os.path.join(SCRIPT_DIR, f"../src/jacobi_omp_{sched}")
    if not os.path.exists(executable):
        print(f"❌ 找不到可执行文件: {executable}，请确保已经 make all\n")
        continue

    print(f"==========================================")
    print(f"   正在评测调度策略: {sched.upper()}")
    print(f"==========================================")

    # --- 1. 强标测试 ---
    print(f"--- 📊 Req 4: 强标测试 (固定 N={N_strong}) ---")
    t1_strong = None
    for t in threads_list:
        print(f"Threads: {t:<2}", end="")
        # 🌟 使用平均耗时函数
        avg_time, iters = get_average_time(executable, N_strong, t, BENCHMARK_ITERS, NUM_RUNS)
        if avg_time is None: continue

        if t == 1:
            t1_strong = avg_time
            speedup = 1.0
        else:
            speedup = t1_strong / avg_time

        print(f"    Avg Time: {avg_time:<7.4f}s | Speedup: {speedup:.2f}")

        plot_data[sched]['strong_threads'].append(t)
        plot_data[sched]['strong_speedup'].append(speedup)
        csv_rows.append([sched, "Strong", N_strong, t, BENCHMARK_ITERS, round(avg_time, 4), round(speedup, 2), "N/A"])

    print("")

    # --- 2. 弱标测试 ---
    print("--- 📊 Req 5: 弱标测试 (人均任务固定 N=256) ---")
    t1_weak = None
    for t, N_weak in weak_scaling_pairs:
        print(f"Threads: {t:<2} | Grid: {N_weak:<4}", end="")
        # 🌟 使用平均耗时函数
        avg_time, iters = get_average_time(executable, N_weak, t, BENCHMARK_ITERS, NUM_RUNS)
        if avg_time is None: continue

        if t == 1:
            t1_weak = avg_time
            efficiency = 100.0
        else:
            efficiency = (t1_weak / avg_time) * 100.0

        print(f"    Avg Time: {avg_time:<7.4f}s | Eff: {efficiency:.2f}%")

        plot_data[sched]['weak_threads'].append(t)
        plot_data[sched]['weak_efficiency'].append(efficiency)
        csv_rows.append([sched, "Weak", N_weak, t, BENCHMARK_ITERS, round(avg_time, 4), "N/A", round(efficiency, 2)])

    print("\n")

# ==========================================
# 导出统一 CSV 和绘制多线图
# ==========================================
csv_filename = os.path.join(DATA_DIR, "scaling_comparison_avg.csv")
with open(csv_filename, mode='w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["Schedule", "Test_Type", "Grid_N", "Threads", "Iters", "Avg_Time_s", "Speedup", "Efficiency_pct"])
    writer.writerows(csv_rows)
print(f"✅ 综合均值测试数据已保存至: {csv_filename}")

style_map = {
    'static': {'color': 'blue', 'marker': 'o', 'label': 'Static (Avg)'},
    'dynamic': {'color': 'orange', 'marker': '^', 'label': 'Dynamic (Avg)'},
    'guided': {'color': 'green', 'marker': 's', 'label': 'Guided (Avg)'}
}

plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
for sched in SCHEDULES:
    if not plot_data[sched]['strong_threads']: continue
    s = style_map[sched]
    plt.plot(plot_data[sched]['strong_threads'], plot_data[sched]['strong_speedup'],
             marker=s['marker'], color=s['color'], linewidth=2, label=s['label'])

plt.plot(threads_list, threads_list, 'k--', alpha=0.6, label='Ideal Speedup')
plt.plot([1, 8], [1.0, 4.0], 'r--', linewidth=2, label='Min Requirement (8 threads >= 4.0x)')

plt.title(f"Strong Scaling Comparison\n(Fixed N=1024, {NUM_RUNS} Runs Avg)")
plt.xlabel("Number of Threads")
plt.ylabel("Speedup")
plt.xticks(threads_list)
plt.grid(True, linestyle=':', alpha=0.7)
plt.legend()

plt.subplot(1, 2, 2)
for sched in SCHEDULES:
    if not plot_data[sched]['weak_threads']: continue
    s = style_map[sched]
    plt.plot(plot_data[sched]['weak_threads'], plot_data[sched]['weak_efficiency'],
             marker=s['marker'], color=s['color'], linewidth=2, label=s['label'])

plt.axhline(y=100.0, color='k', linestyle='--', alpha=0.6, label='Ideal Efficiency (100%)')
plt.axhline(y=70.0, color='r', linestyle='--', linewidth=2, label='Min Requirement (>=70%)')

plt.title(f"Weak Scaling Comparison\n(Task per thread fixed, Base N=256, {NUM_RUNS} Runs Avg)")
plt.xlabel("Number of Threads")
plt.ylabel("Efficiency (%)")
plt.ylim(0, 110)
plt.xticks([1, 2, 4, 8])
plt.grid(True, linestyle=':', alpha=0.7)
plt.legend()

img_filename = os.path.join(DATA_DIR, "scaling_comparison_avg.png")
plt.savefig(img_filename, dpi=300, bbox_inches='tight')
plt.close()
print(f"🎉 均值性能折线图渲染完毕，已保存至 -> {img_filename}")