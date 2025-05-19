import numpy as np
import pywt
import matplotlib.pyplot as plt
import cv2


# 图像小波分解与灰度可视化
def wavelet_decomposition_grayscale(img_path, wavelet='haar', level=2):
    # 读取并预处理图像
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (256, 256))

    # 执行二级小波分解
    coeffs = pywt.wavedec2(img, wavelet, level=level)
    LL, (LH, HL, HH) = coeffs[0], coeffs[1]

    # 增强细节系数可视化对比度
    def enhance_contrast(coeff):
        coeff_abs = np.abs(coeff)
        return np.uint8(255 * (coeff_abs - np.min(coeff_abs)) /
                        (np.max(coeff_abs) - np.min(coeff_abs) + 1e-6))

    # 创建可视化图像
    LL_vis = enhance_contrast(LL)
    LH_vis = enhance_contrast(LH)
    HL_vis = enhance_contrast(HL)
    HH_vis = enhance_contrast(HH)

    # 绘制灰度对比图
    plt.figure(figsize=(12, 8))

    # 原始图像
    plt.subplot(2, 3, 1)
    plt.imshow(img, cmap='gray')
    plt.title('Original Image')

    # 近似系数
    plt.subplot(2, 3, 2)
    plt.imshow(LL_vis, cmap='gray')
    plt.title('LL (Approximation)')

    # 水平细节
    plt.subplot(2, 3, 4)
    plt.imshow(LH_vis, cmap='gray')  # 修改为灰度显示
    plt.title('LH (Horizontal Detail)')

    # 垂直细节
    plt.subplot(2, 3, 5)
    plt.imshow(HL_vis, cmap='gray')  # 修改为灰度显示
    plt.title('HL (Vertical Detail)')

    # 对角细节
    plt.subplot(2, 3, 6)
    plt.imshow(HH_vis, cmap='gray')  # 修改为灰度显示
    plt.title('HH (Diagonal Detail)')

    plt.tight_layout()
    plt.savefig('wavelet_LL_grayscale.jpg', dpi=300)
    plt.show()


# 使用示例
wavelet_decomposition_grayscale('walet_WT_LL.png')