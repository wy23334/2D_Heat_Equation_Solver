# /home/wy/Projects/2D_Heat_Equation_Solver/scripts/phase_2_verify_correctness.py
import subprocess
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
SERIAL_EXE = os.path.join(PROJECT_ROOT, "src", "jacobi_serial")
OMP_EXE = os.path.join(PROJECT_ROOT, "src", "jacobi_omp_static")


def get_center_temp(executable, N, threads=1, max_iters=200000):
    """运行对应的 C 程序并抓取中心温度"""

    if "omp" in executable:
        # 并行版本：支持完整传参 N, threads, max_iters
        cmd = [executable, str(N), str(threads), str(max_iters)]
    else:
        # 串行版：N, max_iters, write_heatmap=0，测试时不覆盖 Phase 1 输出。
        cmd = [executable, str(N), str(max_iters), "0"]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"❌ 运行失败 ({result.returncode}): {result.stderr}")
        return None

    for line in result.stdout.split('\n'):
        # 抓取串行代码的输出
        if "实际计算温度:" in line:
            return float(line.split(':')[1].replace('℃', '').strip())
        # 抓取并行代码的输出
        elif line.startswith("RESULT"):
            return float(line.split('Center=')[1])

    print(f"❌ 运行失败: \n{result.stdout}")
    return None


print("🔍 开始进行正确性极端验证 (Sanity Check)...\n")

# 【工程建议】：为了快速验证，这里改回 256。
# 如果用 1024，它们俩都会输出 21.280939（截断状态）；
# 如果用 256，它们俩都会输出 56.236824（自然收敛状态）。
test_N = 256

print(f"正在运行串行版本 (N={test_N})，请耐心等待...")
serial_temp = get_center_temp(SERIAL_EXE, test_N)

print(f"正在运行并行版本 (N={test_N}, Threads=4)...")
omp_temp = get_center_temp(OMP_EXE, test_N, threads=4)

if serial_temp is not None and omp_temp is not None:
    print("\n=============================================")
    print(f"串行版本中心温度: {serial_temp:.10f} ℃")
    print(f"并行版本中心温度: {omp_temp:.10f} ℃")

    diff = abs(serial_temp - omp_temp)
    print(f"两者绝对偏差 (Diff): {diff:.1e}")

    if diff < 1e-6:
        print("\n✅ 测试通过！并行版本的计算结果与串行完全一致，保证了数学正确性！")
    else:
        print("\n❌ 测试失败！发现数据竞争或计算异常，偏差大于 1e-6！")
    print("=============================================\n")
