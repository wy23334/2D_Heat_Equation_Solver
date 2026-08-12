"""Phase 2: 对比串行与 OpenMP 版本的中心温度。"""

import subprocess
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
SERIAL_EXE = os.path.join(PROJECT_ROOT, "src", "jacobi_serial")
OMP_EXE = os.path.join(PROJECT_ROOT, "src", "jacobi_omp_static")


def get_center_temp(executable, N, threads=1, max_iters=200000):
    """运行指定求解器，并从标准输出解析中心温度。"""

    if "omp" in executable:
        # OpenMP 接口依次接收 N、线程数和最大迭代次数。
        cmd = [executable, str(N), str(threads), str(max_iters)]
    else:
        # 串行接口的第三个参数关闭热力图输出，避免覆盖 Phase 1 数据。
        cmd = [executable, str(N), str(max_iters), "0"]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"[ERROR] 程序退出码为 {result.returncode}: {result.stderr}")
        return None

    for line in result.stdout.split('\n'):
        # 串行程序使用中文标签输出中心温度。
        if "实际计算温度:" in line:
            return float(line.split(':')[1].replace('℃', '').strip())
        # OpenMP 程序使用统一 RESULT 记录格式。
        elif line.startswith("RESULT"):
            return float(line.split('Center=')[1])

    print(f"[ERROR] 未能从程序输出中解析中心温度：\n{result.stdout}")
    return None


print("开始执行串行与 OpenMP 正确性对比。\n")

# N=256 能在 200000 次上限内自然收敛；N=1024 在相同上限下处于
# 截断状态。选用已收敛问题可以同时验证数值一致性和停止判据。
test_N = 256

print(f"运行串行版本：N={test_N}")
serial_temp = get_center_temp(SERIAL_EXE, test_N)

print(f"正在运行并行版本 (N={test_N}, Threads=4)...")
omp_temp = get_center_temp(OMP_EXE, test_N, threads=4)

if serial_temp is not None and omp_temp is not None:
    print("\n正确性对比结果")
    print(f"串行版本中心温度: {serial_temp:.10f} ℃")
    print(f"并行版本中心温度: {omp_temp:.10f} ℃")

    diff = abs(serial_temp - omp_temp)
    print(f"两者绝对偏差 (Diff): {diff:.1e}")

    if diff < 1e-6:
        print("\n[PASS] 串行与 OpenMP 中心温度偏差小于 1e-6。")
    else:
        print("\n[FAIL] 中心温度偏差不小于 1e-6，需要检查数据竞争或归约逻辑。")
    print()
