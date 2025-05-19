import numpy as np
import pywt
import matplotlib.pyplot as plt
import cv2
from scipy.signal import convolve2d


def full_wavelet_pipeline(img_path, wavelet='haar', level=1):  # 修改level=1
    # 读取并预处理图像
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (256, 256))
    original = img.astype(np.float32) / 255.0

    # 小波分解（单级分解保证尺寸匹配）
    coeffs = pywt.wavedec2(original, wavelet, level=level)
    LL, (LH, HL, HH) = coeffs[0], coeffs[1]

    # 卷积核（尺寸调整为适合LL的尺寸）
    conv_kernel = np.array([[-1, -1, -1],
                            [-1, 8, -1],
                            [-1, -1, -1]])

    # 对近似系数进行卷积
    conv_result = convolve2d(LL, conv_kernel, mode='same', boundary='symm')

    # 重组系数进行逆变换
    processed_coeffs = [conv_result, (LH, HL, HH)]
    iwt_result = pywt.waverec2(processed_coeffs, wavelet)

    # 尺寸验证与调整
    if iwt_result.shape != original.shape:
        iwt_result = cv2.resize(iwt_result, original.shape[::-1])  # 尺寸调整

    # 最终输出处理
    final_output = 0.7 * original + 0.3 * iwt_result
    final_output = np.clip(final_output, 0, 1)

    # 可视化（保持原有代码不变）
    def visualize(img, title, pos, cmap='gray'):
        plt.subplot(2, 3, pos)
        plt.imshow(img, cmap=cmap)
        plt.title(f"{title}\n{img.shape}")
        plt.axis('off')

    plt.figure(figsize=(15, 8))
    visualize(original, "Input", 1)
    visualize(LL, "WT LL", 2)
    visualize(conv_result, "Conv", 3)
    visualize(iwt_result, "IWT", 4)
    visualize(final_output, "Output", 5)

    plt.tight_layout()
    plt.savefig('fixed_LL_2pipeline.jpg', dpi=300)
    plt.show()


# 使用示例
full_wavelet_pipeline('walet_WT_LL.png')