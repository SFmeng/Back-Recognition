import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

# 参数设置
size = 256       # 正方形尺寸
sigma = 1.5     # 高斯模糊强度
kernel_size = 21 # 卷积核尺寸（需为奇数）

# 生成基础热图
x = np.linspace(-3, 3, size)
y = np.linspace(-3, 3, size)
X, Y = np.meshgrid(x, y)
Z = np.exp(-(X**2 + Y**2) / (2 * sigma**2))

# 应用高斯模糊[1,7](@ref)
blurred_Z = gaussian_filter(Z, sigma=sigma, mode='mirror')

# 可视化设置
plt.figure(figsize=(8,8))
heatmap = plt.imshow(blurred_Z, cmap='viridis', interpolation='bicubic')
plt.colorbar(heatmap, shrink=0.8)
plt.axis('off')

# 保存为正方形图像
plt.savefig('gaussian_heatmap.png', bbox_inches='tight', pad_inches=0)