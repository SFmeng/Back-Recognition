import numpy as np
import matplotlib.pyplot as plt
import pywt
import cv2
from scipy.signal import convolve2d

def wavelet_image_pipeline(img_path, wavelet_name='db1', level=2, kernel=np.ones((3,3))/9):
    # 读取并预处理图像
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (256, 256))  # 统一尺寸
    img = img.astype(np.float32) / 255.0

    # 小波分解
    coeffs = pywt.wavedec2(img, wavelet_name, level=level)
    cA, cD = coeffs[0], coeffs[1:]  # 近似系数和细节系数

    # 对近似系数进行卷积处理
    conv_cA = convolve2d(cA, kernel, mode='same', boundary='symm')

    # 重组处理后的系数
    processed_coeffs = [conv_cA] + [tuple(d) for d in cD]

    # 逆小波变换
    reconstructed = pywt.waverec2(processed_coeffs, wavelet_name)
    reconstructed = np.clip(reconstructed, 0, 1)  # 限制数值范围

    # 可视化设置
    fig = plt.figure(figsize=(12, 8), dpi=150)
    gs = fig.add_gridspec(2, 4)

    # 绘制各阶段结果
    axes = [
        fig.add_subplot(gs[0, 0]),  # 输入图像
        fig.add_subplot(gs[0, 1]),  # 小波分解
        fig.add_subplot(gs[0, 2]),  # 卷积处理
        fig.add_subplot(gs[0, 3]),  # 逆变换
        fig.add_subplot(gs[1, :])    # 最终输出
    ]

    # 输入图像
    axes[0].imshow(img, cmap='gray')
    axes[0].set_title('Input Image')

    # 小波分解可视化
    wav_decomp, _ = pywt.coeffs_to_array(coeffs)
    axes[1].imshow(wav_decomp, cmap='gray')
    axes[1].set_title('Wavelet Decomposition')

    # 卷积处理可视化
    conv_vis, _ = pywt.coeffs_to_array(processed_coeffs)
    axes[2].imshow(conv_vis, cmap='gray')
    axes[2].set_title('Convolved Coefficients')

    # 逆变换结果
    axes[3].imshow(reconstructed, cmap='gray')
    axes[3].set_title('IWT Reconstruction')

    # 最终输出
    axes[4].imshow(np.abs(img - reconstructed), cmap='gray')
    axes[4].set_title('Final Output (Difference)')

    # 格式化布局
    plt.tight_layout()
    plt.savefig('wavelet_pipeline.png', bbox_inches='tight')
    plt.show()


# 使用示例（替换为您的图片路径）
wavelet_image_pipeline(r'C:\Study\myCode\py_code\AI\BackGround\data\CUB\images\1\IMG_20240401_221148.jpg', wavelet_name='db2', level=2)