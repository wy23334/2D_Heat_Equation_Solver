import os
import subprocess
import csv
import matplotlib.pyplot as plt

print("🔧 正在编译 OpenMP 代码...")
os.system("make clean && make omp")

THREADS = [1, 2, 4, 8]
BENCHMARK_ITER = 2000 # 固定迭代次数，测算绝对算力

def run_omp(N, threads, max_iters):
    cmd = f"./jacobi_omp {N} {threads} {max_iters}"
    print(f"   ▶ 正在跑: N={N:<4}, 核数={threads} ...", end=" ", flush=True)
    res = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    for line in res.stdout.split('\n'):
        if line.startswith("RESULT"):
            parts = line.split(',')
            t = float(parts[4].split('=')[1])
            temp = float(parts[5].split('=')[1])
            print(f"耗时 {t:.4f} 秒")
            return t, temp
    return 0.0, 0.0

print("\n🚀 === 0. 正确性验证 (Sanity Check) ===")
# 满配资源跑到彻底收敛，验证多核有没有算错
_, c_temp = run_omp(128, 4, 200000)
print(f"   [验证] 收敛中心温度: {c_temp:.4f} ℃ (预期 ~56.25 ℃)")
if abs(c_temp - 56.25) < 0.01:
    print("   ✅ 正确性验证通过！(与串行版结果完美一致)")
else:
    print("   ❌ 验证失败，存在计算偏差！")

print("\n🚀 === 1. 强标测试 (Strong Scaling, 固定巨型网格 N=1024) ===")
strong_times = []
for t in THREADS:
    t_val, _ = run_omp(1024, t, BENCHMARK_ITER)
    strong_times.append(t_val)

base_strong = strong_times[0]
speedups = [base_strong / t for t in strong_times]

print("\n🚀 === 2. 弱标测试 (Weak Scaling, 保证每核分到的数据量恒定) ===")
weak_configs = [(256, 1), (362, 2), (512, 4), (724, 8)]
weak_times = []
for N, t in weak_configs:
    t_val, _ = run_omp(N, t, BENCHMARK_ITER)
    weak_times.append(t_val)

base_weak = weak_times[0]
efficiencies = [(base_weak / t) * 100.0 for t in weak_times]

# 💾 导出 CSV
csv_file = "omp_performance.csv"
with open(csv_file, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["Threads", "Strong_Time(s)", "Speedup", "Weak_N", "Weak_Time(s)", "Weak_Efficiency(%)"])
    for i in range(len(THREADS)):
        writer.writerow([THREADS[i], f"{strong_times[i]:.4f}", f"{speedups[i]:.2f}",
                         weak_configs[i][0], f"{weak_times[i]:.4f}", f"{efficiencies[i]:.1f}%"])
print(f"\n✅ 性能数据表已保存至 -> {csv_file}")

# 📊 绘制符合报告要求的高端双子图
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# 图1：强标
ax1.plot(THREADS, speedups, 'ro-', linewidth=2, label='Actual Speedup')
ax1.plot(THREADS, THREADS, 'gray', linestyle='--', label='Ideal Speedup')
ax1.set_xlabel("Number of Threads")
ax1.set_ylabel("Speedup")
ax1.set_title("Strong Scaling (N=1024)")
ax1.set_xticks(THREADS)
ax1.legend()
ax1.grid(True, linestyle=':')

# 图2：弱标
ax2.plot(THREADS, efficiencies, 'bs-', linewidth=2, label='Actual Efficiency')
ax2.axhline(100, color='gray', linestyle='--', label='Ideal Efficiency')
ax2.axhline(70, color='red', linestyle='--', label='Target Red Line (70%)') # 验收红线
ax2.set_xlabel("Number of Threads")
ax2.set_ylabel("Efficiency (%)")
ax2.set_title("Weak Scaling Efficiency")
ax2.set_xticks(THREADS)
ax2.set_ylim(0, 115)
ax2.legend()
ax2.grid(True, linestyle=':')

plt.tight_layout()
plt.savefig("scaling_plot.png", dpi=300)
print("✅ 加速比与效率折线图已保存至 -> scaling_plot.png")