# /home/wy/Projects/2D_Heat_Equation_Solver/scripts/verify_correctness.py
import subprocess

SERIAL_EXE = "../src/jacobi_serial"
OMP_EXE = "../src/jacobi_omp"


def get_center_temp(executable, N, threads=1, max_iters=200000):
    """运行对应的 C 程序并抓取中心温度"""

    if "omp" in executable:
        # 并行版本：支持完整传参 N, threads, max_iters
        cmd = [executable, str(N), str(threads), str(max_iters)]
    else:
        # 【核心修复】串行版本：只传 N！绝对不传多余参数，防止 C 代码解析失败回退到默认网格
        cmd = [executable, str(N)]

    result = subprocess.run(cmd, capture_output=True, text=True)

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
test_N = 1024

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