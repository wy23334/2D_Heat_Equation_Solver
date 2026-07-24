# /home/wy/Projects/2D_Heat_Equation_Solver/scripts/run_scaling_tests.py
# phase 2
import subprocess
import os
import csv
import matplotlib.pyplot as plt

# ==========================================
# 🌟 路径管理与输出配置
# ==========================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "../data"))
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

# 调度策略列表，必须与 Makefile 编译出的后缀一致
SCHEDULES = ["static", "dynamic", "guided"]


# 🌟 修改：让函数接收具体的 executable 路径
def run_c_program(executable, N, threads, max_iters):
    """调用 C 程序并解析返回的耗时和实际迭代次数"""
    if not os.path.exists(executable):
        print(f"❌ 找不到可执行文件: {executable}，请确保已经 make all")
        return None, None

    cmd = [executable, str(N), str(threads), str(max_iters)]
    result = subprocess.run(cmd, capture_output=True, text=True)

    for line in result.stdout.split('\n'):
        if line.startswith("RESULT"):
            parts = line.split(',')
            actual_iters = int(parts[3].split('=')[1])
            time_val = float(parts[4].split('=')[1])
            return time_val, actual_iters

    print(f"❌ 运行失败 (N={N}, Threads={threads} in {executable}):\n{result.stdout}")
    return None, None


print("🚀 开始进行 OpenMP 综合性能自动化评测 (Static vs Dynamic vs Guided)...\n")

BENCHMARK_ITERS = 5000

# 🌟 升级：使用字典分类存储画图数据
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

# ==========================================
# 核心测试循环：遍历三个调度策略
# ==========================================
for sched in SCHEDULES:
    executable = os.path.join(SCRIPT_DIR, f"../src/jacobi_omp_{sched}")
    print(f"==========================================")
    print(f"   正在评测调度策略: {sched.upper()}")
    print(f"==========================================")

    # --- 1. 强标测试 ---
    print("--- 📊 Req 4: 强标测试 (固定 N=1024) ---")
    t1_strong = None
    for t in threads_list:
        time_val, iters = run_c_program(executable, N_strong, t, BENCHMARK_ITERS)
        if time_val is None: continue

        if t == 1:
            t1_strong = time_val
            speedup = 1.0
        else:
            speedup = t1_strong / time_val

        print(f"Threads: {t:<2} | Time: {time_val:<7.4f}s | Speedup: {speedup:.2f}")

        plot_data[sched]['strong_threads'].append(t)
        plot_data[sched]['strong_speedup'].append(speedup)
        csv_rows.append([sched, "Strong", N_strong, t, BENCHMARK_ITERS, round(time_val, 4), round(speedup, 2), "N/A"])

    print("")

    # --- 2. 弱标测试 ---
    print("--- 📊 Req 5: 弱标测试 (人均任务固定 N=256) ---")
    t1_weak = None
    for t, N_weak in weak_scaling_pairs:
        time_val, iters = run_c_program(executable, N_weak, t, BENCHMARK_ITERS)
        if time_val is None: continue

        if t == 1:
            t1_weak = time_val
            efficiency = 100.0
        else:
            efficiency = (t1_weak / time_val) * 100.0

        print(f"Threads: {t:<2} | Grid: {N_weak:<4} | Time: {time_val:<7.4f}s | Eff: {efficiency:.2f}%")

        plot_data[sched]['weak_threads'].append(t)
        plot_data[sched]['weak_efficiency'].append(efficiency)
        csv_rows.append([sched, "Weak", N_weak, t, BENCHMARK_ITERS, round(time_val, 4), "N/A", round(efficiency, 2)])

    print("\n")

# ==========================================
# 🌟 导出统一 CSV 和绘制多线图
# ==========================================
csv_filename = os.path.join(DATA_DIR, "scaling_comparison.csv")
with open(csv_filename, mode='w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["Schedule", "Test_Type", "Grid_N", "Threads", "Iters", "Time_s", "Speedup", "Efficiency_pct"])
    writer.writerows(csv_rows)
print(f"✅ 综合测试数据已保存至: {csv_filename}")

# 为不同策略分配图例样式
style_map = {
    'static': {'color': 'blue', 'marker': 'o', 'label': 'Static'},
    'dynamic': {'color': 'orange', 'marker': '^', 'label': 'Dynamic'},
    'guided': {'color': 'green', 'marker': 's', 'label': 'Guided'}
}

plt.figure(figsize=(14, 6))

# --------------------------
# 图 1：强标加速比对比
# --------------------------
plt.subplot(1, 2, 1)
# 遍历绘制三个策略的实线
for sched in SCHEDULES:
    s = style_map[sched]
    plt.plot(plot_data[sched]['strong_threads'], plot_data[sched]['strong_speedup'],
             marker=s['marker'], color=s['color'], linewidth=2, label=s['label'])

# 理想线与及格线
plt.plot(threads_list, threads_list, 'k--', alpha=0.6, label='Ideal Speedup')
plt.plot([1, 8], [1.0, 4.0], 'r--', linewidth=2, label='Min Requirement (8 threads >= 4.0x)')

plt.title("Strong Scaling Comparison\n(Fixed N=1024)")
plt.xlabel("Number of Threads")
plt.ylabel("Speedup")
plt.xticks(threads_list)
plt.grid(True, linestyle=':', alpha=0.7)
plt.legend()

# --------------------------
# 图 2：弱标并行效率对比
# --------------------------
plt.subplot(1, 2, 2)
# 遍历绘制三个策略的实线
for sched in SCHEDULES:
    s = style_map[sched]
    plt.plot(plot_data[sched]['weak_threads'], plot_data[sched]['weak_efficiency'],
             marker=s['marker'], color=s['color'], linewidth=2, label=s['label'])

# 理想线与及格线
plt.axhline(y=100.0, color='k', linestyle='--', alpha=0.6, label='Ideal Efficiency (100%)')
plt.axhline(y=70.0, color='r', linestyle='--', linewidth=2, label='Min Requirement (>=70%)')

plt.title("Weak Scaling Comparison\n(Task per thread fixed, Base N=256)")
plt.xlabel("Number of Threads")
plt.ylabel("Efficiency (%)")
plt.ylim(0, 110)
plt.xticks([1, 2, 4, 8])
plt.grid(True, linestyle=':', alpha=0.7)
plt.legend()

# 保存图片
img_filename = os.path.join(DATA_DIR, "scaling_comparison.png")
plt.savefig(img_filename, dpi=300, bbox_inches='tight')
plt.close()
print(f"🎉 综合性能折线图渲染完毕，已保存至 -> {img_filename}")