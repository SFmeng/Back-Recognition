import numpy as np
import pywt
import matplotlib.pyplot as plt
import cv2
from scipy.signal import convolve2d

def wavelet_conv_visualization(img_path, wavelet='haar', level=2, kernel=np.array([[-1,-1,-1], [-1,8,-1], [-1,-1,-1]])):
    """
    小波分解 + 卷积处理可视化
    参数：
        img_path: 输入图像路径
        wavelet: 小波基类型，默认haar
        level: 分解层级，默认2级
        kernel: 卷积核，默认拉普拉斯边缘检测核
    """
    # 读取并预处理图像
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (256, 256))
    img = img.astype(np.float32) / 255.0

    # 小波分解
    coeffs = pywt.wavedec2(img, wavelet, level=level)
    LL, (LH, HL, HH) = coeffs[0], coeffs[1]

    # 定义卷积处理函数
    def process_coeff(coeff):
        # 卷积处理
        conv = convolve2d(coeff, kernel, mode='same', boundary='symm')
        # 对比度增强
        conv_norm = (conv - np.min(conv)) / (np.max(conv) - np.min(conv) + 1e-6)
        return np.clip(conv_norm, 0, 1)

    # 处理各子带
    LL_conv = process_coeff(LL)
    LH_conv = process_coeff(LH)
    HL_conv = process_coeff(HL)
    HH_conv = process_coeff(HH)

    # 可视化设置
    plt.figure(figsize=(16, 10))
    plt.suptitle(f"Wavelet Decomposition & Convolution Visualization (Level {level})", y=0.95, fontsize=14)

    # 绘制原始系数
    def plot_sub(coeff, pos, title):
        plt.subplot(4, 2, pos)
        plt.imshow(coeff, cmap='gray', vmin=0, vmax=1)
        plt.title(title, fontsize=10)
        plt.axis('off')

    # 原始子带
    plot_sub(LL, 1, 'Original LL Coefficients')
    plot_sub(LH, 3, 'Original LH Coefficients')
    plot_sub(HL, 5, 'Original HL Coefficients')
    plot_sub(HH, 7, 'Original HH Coefficients')

    # 卷积后子带
    plot_sub(LL_conv, 2, 'Convolved LL (Approximation)')
    plot_sub(LH_conv, 4, 'Convolved LH (Horizontal)')
    plot_sub(HL_conv, 6, 'Convolved HL (Vertical)')
    plot_sub(HH_conv, 8, 'Convolved HH (Diagonal)')

    plt.tight_layout()
    plt.savefig('wavelet_conv_LL_2_results.jpg', dpi=300, bbox_inches='tight')
    plt.show()

# 使用示例
kernel = np.array([[ 0, -1,  0],
                   [-1,  5, -1],
                   [ 0, -1,  0]])  # 锐化卷积核
wavelet_conv_visualization(
    r'walet_WT_LL.png',
    wavelet='db2',
    level=2,
    kernel=kernel
)