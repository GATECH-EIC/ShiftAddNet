# 2020.01.10-Replaced conv with adder
#            Huawei Technologies Co., Ltd. <foss@huawei.com>

import torch.nn as nn
# from models import adder
from adder import adder

__all__ = ['resnet34_shiftadd', 'resnet18_shiftadd']

def conv3x3(in_planes, out_planes, stride=1, quantize=False, weight_bits=8, sparsity=0):
    """3x3 convolution with padding"""
    shift = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)
    add = adder.Adder2D(out_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False, quantize=quantize, weight_bits=weight_bits, sparsity=sparsity)
    return nn.Sequential(shift, add)


def conv1x1(in_planes, out_planes, stride=1, quantize=False, weight_bits=8, sparsity=0):
    """1x1 convolution"""
    shift = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)
    add = adder.Adder2D(out_planes, out_planes, kernel_size=1, stride=1, bias=False, quantize=quantize, weight_bits=weight_bits, sparsity=sparsity)
    return nn.Sequential(shift, add)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, quantize=False, weight_bits=8, sparsity=0):
        super(BasicBlock, self).__init__()
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, planes, quantize=quantize, weight_bits=weight_bits, sparsity=sparsity)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, stride, quantize=quantize, weight_bits=weight_bits, sparsity=sparsity)
        self.bn2 = nn.BatchNorm2d(planes)
        # self.conv3 = conv1x1(planes, planes * self.expansion, quantize=quantize, weight_bits=weight_bits, sparsity=sparsity)
        # self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=False)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        # out = self.conv3(out)
        # out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, quantize=False, weight_bits=8, sparsity=0):
        super(Bottleneck, self).__init__()
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, planes, quantize=quantize, weight_bits=weight_bits, sparsity=sparsity)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, stride=stride, quantize=quantize, weight_bits=weight_bits, sparsity=sparsity)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion, threshold=threshold, sign_threshold=sign_threshold, distribution=distribution, quantize=quantize, weight_bits=weight_bits, sparsity=sparsity)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=False)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes, quantize=False, weight_bits=8, sparsity=0):
        super(ResNet, self).__init__()
        self.quantize = quantize
        self.weight_bits = weight_bits
        self.sparsity = sparsity
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=False)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Conv2d(512 * block.expansion, num_classes, 1)
        self.bn2 = nn.BatchNorm2d(num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion,
                            stride=stride, quantize=self.quantize, weight_bits=self.weight_bits, sparsity=self.sparsity),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes,
                            stride=stride, downsample=downsample, quantize=self.quantize, weight_bits=self.weight_bits, sparsity=self.sparsity))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes,
                            quantize=self.quantize, weight_bits=self.weight_bits, sparsity=self.sparsity))

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
        x = self.fc(x)
        x = self.bn2(x)

        return x.view(x.size(0), -1)

def resnet18_shiftadd(num_classes=1000, quantize=False, weight_bits=8, sparsity=0, **kwargs):
    model = ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes, quantize=quantize, weight_bits=weight_bits, sparsity=sparsity, **kwargs)
    return model


def resnet34_shiftadd(num_classes=1000, quantize=False, weight_bits=8, sparsity=0, **kwargs):
    model = ResNet(BasicBlock, [3, 4, 6, 3], num_classes=num_classes, quantize=quantize, weight_bits=weight_bits, sparsity=sparsity, **kwargs)
    return model


def resnet50_shiftadd(num_classes=1000, quantize=False, weight_bits=8, sparsity=0, **kwargs):
    model = ResNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes, quantize=quantize, weight_bits=weight_bits, sparsity=sparsity, **kwargs)
    return model


