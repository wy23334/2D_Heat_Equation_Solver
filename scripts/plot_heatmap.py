# /home/wy/Projects/2D_Heat_Equation_Solver/scripts/plot_heatmap.py
import os
import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# ⚙️ 核心配置区 (直接在此修改参数)
# ==========================================
TARGET_N = 1024


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# 去读 txt
filename = os.path.join(SCRIPT_DIR, f"heatmap_{TARGET_N}.txt")
# 生成 png
img_name = os.path.join(SCRIPT_DIR, f"colormap_{TARGET_N}.png")

if not os.path.exists(filename):
    print(f"❌ 找不到数据文件：{filename}")
    exit(1)

print(f"📂 正在读取 {filename}...")
data = np.loadtxt(filename)
N = data.shape[0]

plt.figure(figsize=(9, 7))
im = plt.imshow(data, cmap='jet', interpolation='bilinear', origin='upper')
plt.colorbar(im, label='Temperature (℃)')

contours = plt.contour(data, levels=12, colors='black', linewidths=1.0, alpha=0.7)
plt.clabel(contours, inline=True, fontsize=9, fmt='%.1f℃')

plt.title(f"2D Heat Equation - (N={N})\nTop:50, Bottom:75, Left:100, Right:0")
plt.xlabel("X (Columns)")
plt.ylabel("Y (Rows)")

plt.savefig(img_name, dpi=300, bbox_inches='tight')
print(f"🎉 热力图与等温线渲染完毕，已保存至 -> {img_name}")