# [Req 1] ： 使用 perf stat -e cache-misses,cache-references 测量 cache miss 数量，保存为 cache_miss_report.csv
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


def create_test_files(source_file):
    """读取源文件，生成行优先（良好）和列优先（糟糕）两个版本的 C 代码"""
    if not os.path.exists(source_file):
        print(f"[!] 找不到源文件: {source_file}")
        sys.exit(1)

    with open(source_file, 'r', encoding='utf-8') as f:
        code = f.read()

    # 降低迭代次数以加速测试，并禁用文件写入以防干扰 IO
    code = re.sub(r'#define MAX_ITER \d+', '#define MAX_ITER 1000', code)
    code = re.sub(r'FILE \*fp = fopen\(.*?\);', 'FILE *fp = NULL;', code)

    # 1. 保存良好的版本 (行优先)
    good_file = os.path.join(SCRIPT_DIR, 'jacobi_good.c')
    with open(good_file, 'w', encoding='utf-8') as f:
        f.write(code)

    # 2. 生成糟糕的版本 (列优先)
    bad_code = code.replace(
        "for (int i = 1; i < N - 1; i++) {\n            for (int j = 1; j < N - 1; j++) {",
        "for (int j = 1; j < N - 1; j++) {\n            for (int i = 1; i < N - 1; i++) {"
    )

    bad_file = os.path.join(SCRIPT_DIR, 'jacobi_bad.c')
    with open(bad_file, 'w', encoding='utf-8') as f:
        f.write(bad_code)

    return good_file, bad_file


def compile_code(c_file, exe_name):
    """编译代码到脚本目录"""
    exe_path = os.path.join(SCRIPT_DIR, exe_name)
    print(f"[*] 正在编译 {c_file} -> {exe_path} ...")
    result = subprocess.run(["gcc", "-O2", c_file, "-o", exe_path], capture_output=True, text=True)
    if result.returncode != 0:
        print(f"[!] 编译失败:\n{result.stderr}")
        sys.exit(1)
    return exe_path


def run_perf(executable_path, N):
    """运行 perf 工具并提取硬件统计数据，仅返回 Refs 和 Misses 数量"""
    print(f"[*] 正在运行 {os.path.basename(executable_path)} (N={N}) 并收集原生 perf 数据...")

    cmd = ["perf", "stat", "-e", "cache-misses,cache-references", executable_path, str(N)]
    result = subprocess.run(cmd, stderr=subprocess.PIPE, stdout=subprocess.DEVNULL, text=True)

    output = result.stderr
    misses = 0
    refs = 0

    if "<not supported>" in output:
        print(f"\n[!] 严重警告: perf 报告硬件计数器不可用！原始输出:\n{output}")
        sys.exit(1)

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
        print(f"\n[!] 警告: 无法解析出数据，请检查正则表达式。原始输出:\n{output}")

    return refs, misses


def main():
    source_c = os.path.join(SRC_DIR, "jacobi_serial.c")
    N = 1024

    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)

    print("=== 阶段 1: 生成对比代码 ===")
    good_c, bad_c = create_test_files(source_c)

    print("\n=== 阶段 2: 编译代码 ===")
    good_exe = compile_code(good_c, "jacobi_good_exe")
    bad_exe = compile_code(bad_c, "jacobi_bad_exe")

    print("\n=== 阶段 3: 执行硬件性能监控 (perf) ===")
    good_refs, good_miss = run_perf(good_exe, N)
    bad_refs, bad_miss = run_perf(bad_exe, N)

    # ---------------- 终端可视化输出 ----------------
    terminal_report = (
            "=== 🎯 Cache Miss 数量对比表 (N=1024, MAX_ITER=1000) ===\n"
            + "-" * 75 + "\n"
            + f"{'访问模式 (Access Pattern)':<25} | {'Cache Refs (缓存引用)':<20} | {'Cache Misses (未命中数量)':<20}\n"
            + "-" * 75 + "\n"
            + f"{'行优先 (Row-Major, Good)':<25} | {good_refs:<20,} | {good_miss:<20,}\n"
            + f"{'列优先 (Col-Major, Bad)':<25} | {bad_refs:<20,} | {bad_miss:<20,}\n"
            + "-" * 75 + "\n"
    )
    print("\n" + terminal_report)

    # ---------------- 写入 CSV 文件 ----------------
    csv_filename = "cache_miss_report.csv"
    csv_report_path = os.path.join(DATA_DIR, csv_filename)

    with open(csv_report_path, "w", encoding="utf-8") as f:
        f.write("Access_Pattern,Cache_Refs,Cache_Misses\n")
        f.write(f"Row-Major_Good,{good_refs},{good_miss}\n")
        f.write(f"Col-Major_Bad,{bad_refs},{bad_miss}\n")

    print(f"[*] 报告已成功保存至: {csv_report_path}")

    # 清理临时文件
    for f in [good_c, bad_c, good_exe, bad_exe]:
        if os.path.exists(f):
            os.remove(f)
    print("[*] 测试完成，已清理临时文件。")


if __name__ == "__main__":
    main()