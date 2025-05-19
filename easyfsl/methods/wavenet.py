import torch
from torch import nn

from easyfsl.modules.WTConv import WTConv2d


class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None, wt_levels=1, wt_type='db1'):
        super(BasicBlock, self).__init__()

        # 使用普通卷积处理通道变化
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        # 使用WTConv2d处理不变的通道
        self.wtconv = WTConv2d(out_channels, out_channels, kernel_size=3, wt_levels=wt_levels, wt_type=wt_type)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # 如果输入输出通道数或stride不匹配，使用downsample进行调整
        self.downsample = downsample

    def forward(self, x):
        identity = x

        # 普通卷积用于处理通道变化
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        # WTConv2d用于特征提取
        out = self.wtconv(out)
        out = self.bn2(out)

        # 如果需要调整通道或分辨率
        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class WaveletResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000, wt_levels=1, wt_type='db1'):
        super(WaveletResNet, self).__init__()
        self.in_channels = 64

        # 初始卷积层
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # 残差模块
        self.layer1 = self._make_layer(block, 64, layers[0], stride=1, wt_levels=wt_levels, wt_type=wt_type)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, wt_levels=wt_levels, wt_type=wt_type)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, wt_levels=wt_levels, wt_type=wt_type)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, wt_levels=wt_levels, wt_type=wt_type)

        # 分类头
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, block, out_channels, blocks, stride=1, wt_levels=1, wt_type='db1'):
        downsample = None
        if stride != 1 or self.in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample, wt_levels, wt_type))
        self.in_channels = out_channels
        for _ in range(1, blocks):
            layers.append(block(out_channels, out_channels, wt_levels=wt_levels, wt_type=wt_type))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x


def WaveletResNet12(num_classes=1000, wt_levels=1, wt_type='db1'):
    return WaveletResNet(BasicBlock, [1, 1, 1, 1], num_classes=num_classes, wt_levels=wt_levels, wt_type=wt_type)


if __name__ == "__main__":
    model = WaveletResNet12(num_classes=640, wt_levels=2, wt_type='db1')
    input_tensor = torch.randn(1, 3, 84, 84)
    output = model(input_tensor)
    print("Output shape:", output.shape)
