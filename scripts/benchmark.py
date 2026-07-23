import os
import subprocess
import csv
import matplotlib.pyplot as plt
import statistics

# --- 跨目录配置 ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, '../src'))

THREADS = [1, 2, 4, 8]
BENCHMARK_ITER = 2000
NUM_RUNS = 100  # 黄金准则：跑100次取均值抹平波动

# 三种调度策略对应的编译宏 (空字符串会让代码进入 #else 即 static 分支)
SCHEDULES = {
    "Static": "",
    "Dynamic": "-DSCHED_DYNAMIC",
    "Guided": "-DSCHED_GUIDED"
}


def compile_omp(flag):
    print(f"\n🔧 [跨目录编译] 正在注入调度策略宏指令 ...")
    cmd = f"make clean && make omp OMPFLAGS=\"-fopenmp {flag}\""
    subprocess.run(cmd, shell=True, cwd=SRC_DIR, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def run_omp_avg(N, threads, max_iters, runs):
    cmd = f"./jacobi_omp {N} {threads} {max_iters}"
    times = []
    print(f"    ▶ N={N:<4}, 核数={threads} | 正在跑 {runs} 次 ", end="", flush=True)

    for i in range(runs):
        res = subprocess.run(cmd, shell=True, cwd=SRC_DIR, capture_output=True, text=True)
        for line in res.stdout.split('\n'):
            if line.startswith("RESULT"):
                try:
                    t = float(line.split(',')[4].split('=')[1])
                    times.append(t)
                except:
                    pass
                break
        # 每跑 10 次打印一个点当进度条，免得干等焦虑
        if (i + 1) % 10 == 0:
            print(".", end="", flush=True)

    avg_time = statistics.mean(times)
    print(f" 平均耗时: {avg_time:.4f} 秒")
    return avg_time


print(f"🚀 === 启动极致压测流水线 (100 次大样本均值) ===")
print(f"⚠️  警告: 将执行 3x8x100 = 2400 次测试！")
print(f"请泡一杯咖啡，这大概需要 10 分钟左右的运算时间。")

results_speedup = {}
results_efficiency = {}
csv_data = []

for name, flag in SCHEDULES.items():
    print(f"\n=====================================")
    print(f" 🏆 开始压测: {name} 调度策略")
    print(f"=====================================")
    compile_omp(flag)

    print(f"  [1/2] 强标测试 (N=1024)")
    strong_times = []
    for t in THREADS:
        avg_t = run_omp_avg(1024, t, BENCHMARK_ITER, NUM_RUNS)
        strong_times.append(avg_t)

    base_strong = strong_times[0]
    speedups = [base_strong / st for st in strong_times]
    results_speedup[name] = speedups

    print(f"\n  [2/2] 弱标测试 (固定每核负载)")
    weak_configs = [(256, 1), (362, 2), (512, 4), (724, 8)]
    weak_times = []
    for N, t in weak_configs:
        avg_t = run_omp_avg(N, t, BENCHMARK_ITER, NUM_RUNS)
        weak_times.append(avg_t)

    base_weak = weak_times[0]
    efficiencies = [(base_weak / wt) * 100.0 for wt in weak_times]
    results_efficiency[name] = efficiencies

    for i in range(len(THREADS)):
        csv_data.append([
            name, THREADS[i],
            f"{strong_times[i]:.4f}", f"{speedups[i]:.2f}",
            weak_configs[i][0], f"{weak_times[i]:.4f}", f"{efficiencies[i]:.1f}%"
        ])

csv_file = os.path.join(SCRIPT_DIR, "omp_schedules_benchmark.csv")
with open(csv_file, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(
        ["Schedule", "Threads", "Strong_Time(s)", "Speedup", "Weak_N", "Weak_Time(s)", "Weak_Efficiency(%)"])
    writer.writerows(csv_data)
print(f"\n✅ 所有 100 次均值数据已写入 -> {csv_file}")

# ==========================================
# 📊 绘制顶级三剑客同台竞技折线图
# ==========================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
colors = {"Static": "#d62728", "Dynamic": "#2ca02c", "Guided": "#1f77b4"}
markers = {"Static": "o", "Dynamic": "s", "Guided": "^"}

# 强标图
for name in SCHEDULES.keys():
    ax1.plot(THREADS, results_speedup[name], color=colors[name], marker=markers[name], linewidth=2,
             label=f'{name} Speedup')
ax1.plot(THREADS, THREADS, 'gray', linestyle='--', label='Ideal Speedup')
ax1.set_xlabel("Number of Threads")
ax1.set_ylabel("Speedup")
ax1.set_title("Strong Scaling Comparison (100 runs avg)")
ax1.set_xticks(THREADS)
ax1.legend()
ax1.grid(True, linestyle=':')

# 弱标图
for name in SCHEDULES.keys():
    ax2.plot(THREADS, results_efficiency[name], color=colors[name], marker=markers[name], linewidth=2,
             label=f'{name} Efficiency')
ax2.axhline(100, color='gray', linestyle='--', label='Ideal Efficiency')
ax2.axhline(70, color='red', linestyle='--', linewidth=2, label='Target Red Line (70%)')
ax2.set_xlabel("Number of Threads")
ax2.set_ylabel("Efficiency (%)")
ax2.set_title("Weak Scaling Efficiency Comparison (100 runs avg)")
ax2.set_xticks(THREADS)
ax2.set_ylim(0, 115)
ax2.legend()
ax2.grid(True, linestyle=':')

plt.tight_layout()
plot_file = os.path.join(SCRIPT_DIR, "scaling_plot_all_schedules.png")
plt.savefig(plot_file, dpi=300)
print(f"✅ 终极三剑客对比折线图已保存至 -> {plot_file}")