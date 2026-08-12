"""Phase 1: 读取温度场、验证中心温度并绘制热力图。"""

# [Phase 1 - Req 6] 读取 phase_1_heatmap.txt 绘制热力图，
# 并与 N=128 的中心温度参考值进行正确性对比。
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import zoom

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# 所有输入与输出均相对于项目 data 目录解析，避免依赖当前工作目录。
DATA_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "../data"))

# 创建结果目录；exist_ok 保证重复运行不会改变既有数据。
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

# Phase 1 串行求解器输出的温度场文件。
filename = os.path.join(DATA_DIR, "phase_1_heatmap.txt")

if not os.path.exists(filename):
    print(f"[ERROR] 找不到数据文件：{filename}")
    exit(1)

print(f"[INFO] 读取温度场文件：{filename}")

# 解析以“### N=”分隔的一个或多个温度矩阵。
matrices = {}
current_N = None
current_data = []

with open(filename, 'r') as f:
    for line in f:
        line = line.strip()
        if not line:
            continue

        # 分隔标记给出后续矩阵的网格规模。
        if line.startswith("### N="):
            # 遇到下一标记前，先保存已经解析完毕的矩阵。
            if current_N is not None:
                matrices[current_N] = np.array(current_data, dtype=float)

            # 初始化新矩阵的解析状态。
            current_N = int(line.split("=")[1])
            current_data = []
        else:
            # 普通数据行由空格分隔的浮点温度组成。
            row = [float(x) for x in line.split()]
            current_data.append(row)

# 文件结束时保存最后一个矩阵。
if current_N is not None and current_data:
    matrices[current_N] = np.array(current_data, dtype=float)

# 对每个网格执行中心温度验证并绘制热力图。
for N, data in matrices.items():
    print(f"\n处理网格数据：N={N}")

    # [Phase 1 - Req 6] 使用题目给出的 N=128 中心温度参考值。
    target_temp = 56.25

    if N % 2 != 0:
        center_temp = data[N // 2, N // 2]
    else:
        m = N // 2
        center_temp = (data[m - 1, m - 1] + data[m, m] + data[m - 1, m] + data[m, m - 1]) / 4.0

    error = abs(center_temp - target_temp) / target_temp * 100.0
    print(f"参考中心温度: {target_temp:.4f} ℃")
    print(f"计算中心温度: {center_temp:.4f} ℃")
    print(f"相对误差: {error:.4f}%")

    # 热力图直接使用求解器输出；bilinear 仅影响显示，不修改数值数据。
    plt.figure(figsize=(9, 7))
    im = plt.imshow(data, cmap='jet', interpolation='bilinear', origin='upper')
    plt.colorbar(im, label='Temperature (℃)')

    # 小网格使用双线性插值生成等温线。order=1 不产生高阶插值过冲，
    # 等温线只承担可视化作用，不参与中心温度误差计算。
    zoom_factor = max(1, 512 // N)
    levels_count = 12

    if zoom_factor > 1:
        smooth_data = zoom(data, zoom_factor, order=1)
        x = np.linspace(0, N - 1, smooth_data.shape[1])
        y = np.linspace(0, N - 1, smooth_data.shape[0])
        contours = plt.contour(x, y, smooth_data, levels=levels_count, colors='black', linewidths=1.0, alpha=0.7)
    else:
        contours = plt.contour(data, levels=levels_count, colors='black', linewidths=1.0, alpha=0.7)

    plt.clabel(contours, inline=True, fontsize=9, fmt='%.1f℃')

    plt.title(f"2D Heat Equation - (N={N})\nTop:50, Bottom:75, Left:100, Right:0")
    plt.xlabel("X (Columns)")
    plt.ylabel("Y (Rows)")

    # 使用 Phase 前缀保存结果，便于与其他阶段数据区分。
    img_name = os.path.join(DATA_DIR, f"phase_1_colormap_{N}.png")
    plt.savefig(img_name, dpi=300, bbox_inches='tight')

    # 显式释放画布，避免处理多个矩阵时累积图形资源。
    plt.close()
    print(f"[INFO] 热力图已保存：{img_name}")
