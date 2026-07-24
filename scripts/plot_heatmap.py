# /home/wy/Projects/2D_Heat_Equation_Solver/scripts/plot_heatmap.py
# phase 1 [Req 6]: 读取 heatmap.txt 绘制热力图（colormap），并与小规模手工计算结果对比
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import zoom  # 🌟 新增：导入 scipy 的放大插值模块

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# 现在只读取一个统一的 txt 文件
filename = os.path.join(SCRIPT_DIR, "heatmap.txt")

if not os.path.exists(filename):
    print(f"❌ 找不到数据文件：{filename}")
    exit(1)

print(f"📂 正在读取 {filename} 及其包含的多个矩阵...")

# ==========================================
# 1. 逐行解析带标记的文本文件
# ==========================================
matrices = {}
current_N = None
current_data = []

with open(filename, 'r') as f:
    for line in f:
        line = line.strip()
        if not line:
            continue

        # 遇到分割标记，准备接收新矩阵
        if line.startswith("### N="):
            # 如果 current_N 不为空，说明上一个矩阵的数据已经收集完毕，将其存入字典
            if current_N is not None:
                matrices[current_N] = np.array(current_data, dtype=float)

            # 提取新的 N 值，并清空临时列表准备收集新数据
            current_N = int(line.split("=")[1])
            current_data = []
        else:
            # 读取当前矩阵的每一行浮点数
            row = [float(x) for x in line.split()]
            current_data.append(row)

# 文件读取结束后，把最后一个收集到的矩阵也存入字典
if current_N is not None and current_data:
    matrices[current_N] = np.array(current_data, dtype=float)

# ==========================================
# 2. 遍历所有矩阵，依次验证并画图
# ==========================================
for N, data in matrices.items():
    print(f"\n======================================")
    print(f"   正在处理 N={N} 的数据...")
    print(f"======================================")

    # [Req 6]: 小规模手工计算结果对比验证
    target_temp = 56.25  # 基于拉普拉斯方程稳态对称性的理论中心温度

    if N % 2 != 0:
        center_temp = data[N // 2, N // 2]
    else:
        m = N // 2
        center_temp = (data[m - 1, m - 1] + data[m, m] + data[m - 1, m] + data[m, m - 1]) / 4.0

    error = abs(center_temp - target_temp) / target_temp * 100.0
    print(f"🔍 理论中心温度: {target_temp:.4f} ℃")
    print(f"🔍 实际中心温度: {center_temp:.4f} ℃")
    print(f"🔍 相对误差占比: {error:.4f}%")

    # --- 🎨 绘制热力图 (底层颜色块依然使用绝对真实的原数据) ---
    plt.figure(figsize=(9, 7))
    im = plt.imshow(data, cmap='jet', interpolation='bilinear', origin='upper')
    plt.colorbar(im, label='Temperature (℃)')

    # ==========================================
    # 🌟 修复与改进：恢复完美的 levels=12，并安全平滑
    # ==========================================
    zoom_factor = max(1, 512 // N)
    levels_count = 12  # 恢复你原本完美的 12 层划分！

    if zoom_factor > 1:
        # 【修复】：将 order=3 改为 order=1 (双线性插值)
        # 这样既能把小网格平滑，又绝对不会在极端的边界处产生过冲报错
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

    # 动态生成图片名称
    img_name = os.path.join(SCRIPT_DIR, f"colormap_{N}.png")
    plt.savefig(img_name, dpi=300, bbox_inches='tight')

    # 关闭当前画布
    plt.close()
    print(f"🎉 热力图与等温线渲染完毕，已保存至 -> {img_name}")