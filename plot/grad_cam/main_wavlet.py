import os
import cv2
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
from collections import OrderedDict
from utils import GradCAM, show_cam_on_image
from easyfsl.modules import WaveletResNet12WithPretrainedFeatures


def main():
    # ================= 模型初始化 =================
    model = WaveletResNet12WithPretrainedFeatures(num_classes=640, wt_levels=2, wt_type='db1')

    # ================= 权重加载 =================
    checkpoint = torch.load("../../best_model.pth", map_location="cpu")

    # 提取backbone参数并适配键名
    backbone_state_dict = OrderedDict()
    for k, v in checkpoint["model_state"].items():
        if k.startswith("backbone."):
            new_key = k.replace("backbone.", "")
            backbone_state_dict[new_key] = v

    # 严格加载参数
    load_info = model.load_state_dict(backbone_state_dict, strict=True)
    print("参数加载状态:", load_info)

    # ================= 设备设置 =================
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    # ================= 目标层选择 =================
    target_layers = [model.fusion4.conv]

    # ================= 数据预处理 =================
    data_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # ================= 图像加载 =================
    img_path = r'C:\Study\myCode\py_code\AI\BackGround\data\CUB\images\66\5f7a49e16c226f9bcfab56474fddc89.jpg'
    img_pil = Image.open(img_path).convert('RGB')
    input_tensor = data_transform(img_pil).unsqueeze(0).to(device)
    img_np = np.array(img_pil)

    # ================= GradCAM计算 =================
    cam = GradCAM(model=model, target_layers=target_layers, use_cuda=torch.cuda.is_available())

    with torch.no_grad():
        output = model(input_tensor)
    # pred_category = torch.argmax(output).item()
    pred_category = 447


    grayscale_cam = cam(input_tensor=input_tensor, target_category=pred_category)
    grayscale_cam = cv2.resize(grayscale_cam[0], (img_np.shape[1], img_np.shape[0]))

    # ================= 可视化 =================
    visualization = show_cam_on_image(img_np.astype(np.float32) / 255., grayscale_cam, use_rgb=True)

    plt.figure(figsize=(10, 6))
    plt.subplot(121)
    plt.imshow(img_np)
    plt.title("Original Image")
    plt.axis('off')

    plt.subplot(122)
    plt.imshow(visualization)
    plt.title("Grad-CAM")
    plt.axis('off')

    plt.tight_layout()
    plt.savefig('grad_cam_result.jpg', dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    main()