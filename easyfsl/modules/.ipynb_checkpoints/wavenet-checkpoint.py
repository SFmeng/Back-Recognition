
import timm
import torch
from torch import nn
from easyfsl.modules.WTConv import WTConv2d


class FeatureGuidedFusion(nn.Module):
    def __init__(self, in_channels, guided_channels):
        super(FeatureGuidedFusion, self).__init__()
        self.conv = nn.Conv2d(in_channels + guided_channels, in_channels, kernel_size=1, stride=1, bias=False)
        self.bn = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, guided_feature):
        # 融合网络特征和预训练特征
        x = torch.cat([x, guided_feature], dim=1)  # 拼接特征
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None, wt_levels=1, wt_type='db1'):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.wtconv = WTConv2d(out_channels, out_channels, kernel_size=3, wt_levels=wt_levels, wt_type=wt_type)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.wtconv(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out


class WaveletResNetWithPretrainedFeatures(nn.Module):
    def __init__(self, block, layers, num_classes=1000, wt_levels=1, wt_type='db1'):
        super(WaveletResNetWithPretrainedFeatures, self).__init__()
        self.in_channels = 64

        # 加载预训练模型并提取特征
        self.pretrained_model = timm.create_model('resnet10t', pretrained=True)
        self.guided_features = {}

        # 初始化融合模块
        self.fusion1 = FeatureGuidedFusion(64, 64)
        self.fusion2 = FeatureGuidedFusion(128, 128)
        self.fusion3 = FeatureGuidedFusion(256, 256)
        self.fusion4 = FeatureGuidedFusion(512, 512)

        # 定义网络结构
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0], stride=1, wt_levels=wt_levels, wt_type=wt_type)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, wt_levels=wt_levels, wt_type=wt_type)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, wt_levels=wt_levels, wt_type=wt_type)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, wt_levels=wt_levels, wt_type=wt_type)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, block, out_channels, blocks, stride=1, wt_levels=1, wt_type='db1'):
        downsample = None
        if stride != 1 or self.in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        layers = [block(self.in_channels, out_channels, stride, downsample, wt_levels, wt_type)]
        self.in_channels = out_channels
        for _ in range(1, blocks):
            layers.append(block(out_channels, out_channels, wt_levels=wt_levels, wt_type=wt_type))
        return nn.Sequential(*layers)

    def extract_guided_features(self, x):
        # 从预训练模型中提取中间层特征
        for name, module in self.pretrained_model.named_children():
            x = module(x)
            if name in ['layer1', 'layer2', 'layer3', 'layer4']:
                self.guided_features[name] = x
        return self.guided_features

    def forward(self, x):
        # 提取预训练模型的特征
        guided_features = self.extract_guided_features(x.clone())

        # 网络前向传播并融合特征
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.fusion1(x, guided_features['layer1'])

        x = self.layer2(x)
        x = self.fusion2(x, guided_features['layer2'])

        x = self.layer3(x)
        x = self.fusion3(x, guided_features['layer3'])

        x = self.layer4(x)
        x = self.fusion4(x, guided_features['layer4'])

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x


def WaveletResNet12WithPretrainedFeatures(num_classes=1000, wt_levels=1, wt_type='db1'):
    return WaveletResNetWithPretrainedFeatures(BasicBlock, [1, 1, 1, 1], num_classes=num_classes, wt_levels=wt_levels, wt_type=wt_type)


if __name__ == "__main__":
    # 测试模型
    model = WaveletResNet12WithPretrainedFeatures(num_classes=10, wt_levels=2, wt_type='db1')
    input_tensor = torch.randn(1, 3, 224, 224)
    output = model(input_tensor)
    print("Output shape:", output.shape)
