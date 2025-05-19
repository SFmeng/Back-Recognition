import numpy as np
import matplotlib.pyplot as plt

# ------------ 数据定义 ------------
methods = ['BDCSPN', 'Laplacian', 'MN', 'PN', 'RN', 'Simple', 'Ours']
shots = ['1-shot', '5-shot', '10-shot']
data = np.array([
    [46.06, 51.30, 55.79],  # BDCSPN
    [41.83, 48.69, 54.83],  # Laplacian
    [52.71, 62.66, 70.73],  # MN
    [52.85, 68.46, 72.01],  # PN
    [44.37, 54.59, 53.89],  # RN
    [45.96, 54.83, 57.55],  # Simple
    [56.49, 69.43, 73.39]  # Ours
])

# ------------ 顶会级配色方案 ------------
# colors = [
#     '#4E79A7',  # 科技蓝 (Nature主色)
#     '#59A14F',  # 莫兰迪绿 (Science对比色)
#     '#666666',  # 中性灰 (IEEE标准色)
#     '#B07AA1',  # 紫灰色 (Cell Reports配色)
#     '#FFBE7D',  # 暖沙橙 (PNAS强调色)
#     '#86BCB6',  # 青瓷色 (AAAS推荐色)
#     '#E15759'  # 学术红 (Nature主色，Ours强调)
# ]
# Science风格 [6](@ref)
# colors = ['#4455AA', '#55BBCC', '#BB5544', '#DDAA33', '#999933', '#882255', '#CC6677']
#
# Nature风格 [6](@ref)
colors = ['#4B6A9B', '#63B5CF', '#EBA17A', '#A9A9A9', '#4E805E', '#956984', '#C85252']



# ------------ 专业绘图设置 ------------
plt.rcParams.update({
    'font.family': 'Arial',  # 学术标准字体[8](@ref)
    'axes.spines.right': False,
    'axes.spines.top': False,
    'axes.labelweight': 'bold'
})

fig, ax = plt.subplots(figsize=(12, 7), dpi=300)
bar_width = 0.11
x = np.arange(len(shots))

# ------------ 简洁柱状图绘制 ------------
for i, (method, color) in enumerate(zip(methods, colors)):
    bars = ax.bar(
        x + i * bar_width,
        data[i],
        width=bar_width,
        color=color,
        edgecolor='white',
        linewidth=1.2,
        alpha=0.95,  # 微透明提升层次感
        label=method,
        zorder=3
    )

    # 统一数值标注位置[7](@ref)
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2,
                height + 0.5,  # 固定标注高度
                f'{height:.1f}%',
                ha='center',
                va='bottom',
                fontsize=10,
                color=color,  # 颜色与柱体一致
                weight='bold')

# ------------ 坐标轴优化 ------------
ax.set_xticks(x + 3 * bar_width)
ax.set_xticklabels(shots, fontsize=13, fontweight='semibold')
ax.set_ylabel('Accuracy (%)', fontsize=14, labelpad=12)
ax.set_ylim(35, 78)

# 单一Y轴网格系统[6](@ref)
ax.yaxis.grid(True, color='#EEEEEE', linewidth=1.2, linestyle='--')

# ------------ 精简图例系统 ------------
legend = ax.legend(
    ncol=3,
    fontsize=12,
    frameon=True,
    framealpha=0.9,
    loc='upper center',
    bbox_to_anchor=(0.5, 1.18),  # 紧凑布局
    facecolor='#FFFFFF',
    edgecolor='#666666',
    columnspacing=1.8
)

# 图例颜色同步
for text, color in zip(legend.get_texts(), colors):
    text.set_color('#333333')
    text.set_weight('semibold')

# ------------ 导出设置 ------------
plt.tight_layout()
plt.savefig('Performance_Comparison.png', dpi=600)  # 矢量图出版标准
plt.show()