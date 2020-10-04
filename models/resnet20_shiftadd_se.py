# from models import adder
from adder import adder
import torch.nn as nn
from se_shift import SEConv2d, SELinear
import torch.nn.functional as F

__all__ = ['resnet20_shiftadd_se', 'resnet20_shiftadd_se_vis']


def conv3x3(in_planes, out_planes, threshold, sign_threshold, distribution, stride=1, quantize=False, weight_bits=8, sparsity=0):
    " 3x3 convolution with padding "
    shift = SEConv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False, threshold=threshold, sign_threshold=sign_threshold, distribution=distribution)
    add = adder.Adder2D(out_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False, quantize=quantize, weight_bits=weight_bits, sparsity=sparsity)
    return nn.Sequential(shift, add)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, threshold, sign_threshold, distribution, stride=1, downsample=None, quantize=False, weight_bits=8, sparsity=0):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, threshold=threshold, sign_threshold=sign_threshold, distribution=distribution, stride=stride, quantize=quantize, weight_bits=weight_bits, sparsity=sparsity)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, threshold=threshold, sign_threshold=sign_threshold, distribution=distribution, quantize=quantize, weight_bits=weight_bits, sparsity=sparsity)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes, threshold, sign_threshold, distribution, quantize=False, weight_bits=8, sparsity=0):
        super(ResNet, self).__init__()
        self.inplanes = 16
        self.quantize = quantize
        self.threshold = threshold
        self.sign_threshold = sign_threshold
        self.distribution = distribution
        self.weight_bits = weight_bits
        self.sparsity = sparsity
        # self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv1 = SEConv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False, threshold=threshold, sign_threshold=sign_threshold)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 16, layers[0])
        self.layer2 = self._make_layer(block, 32, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 64, layers[2], stride=2)
        self.avgpool = nn.AvgPool2d(8, stride=1)
        # use conv as fc layer (addernet)
        # self.fc = nn.Conv2d(64 * block.expansion, num_classes, 1, bias=False)
        self.fc = SEConv2d(64 * block.expansion, num_classes, 1, bias=False, threshold=threshold, sign_threshold=sign_threshold)
        self.bn2 = nn.BatchNorm2d(num_classes)

        # for m in self.modules():
        #     if isinstance(m, nn.BatchNorm2d):
        #         m.weight.data.fill_(1)
        #         m.bias.data.zero_()
        for m in self.modules():
            # if isinstance(m, adder.Adder2D):
            #     nn.init.kaiming_normal_(m.adder, mode='fan_out', nonlinearity='relu')
            if isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                SEConv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False,
                        threshold=self.threshold, sign_threshold=self.sign_threshold, distribution=self.distribution), # shift
                adder.Adder2D(planes * block.expansion, planes * block.expansion, kernel_size=1, stride=1, bias=False,
                              quantize=self.quantize, weight_bits=self.weight_bits, sparsity=self.sparsity), # add
                nn.BatchNorm2d(planes * block.expansion)
            )

        layers = []
        layers.append(block(inplanes=self.inplanes, planes=planes, threshold=self.threshold,
                            sign_threshold=self.sign_threshold, distribution=self.distribution, stride=stride, downsample=downsample,
                            quantize=self.quantize, weight_bits=self.weight_bits, sparsity=self.sparsity))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(inplanes=self.inplanes, planes=planes, threshold=self.threshold, sign_threshold=self.sign_threshold,
                                distribution=self.distribution, quantize=self.quantize, weight_bits=self.weight_bits, sparsity=self.sparsity))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = self.fc(x)
        x = self.bn2(x)
        return x.view(x.size(0), -1)

class ResNet_vis(nn.Module):

    def __init__(self, block, layers, num_classes, threshold, sign_threshold, distribution, quantize=False, weight_bits=8, sparsity=0):
        super(ResNet_vis, self).__init__()
        self.inplanes = 16
        self.quantize = quantize
        self.threshold = threshold
        self.sign_threshold = sign_threshold
        self.distribution = distribution
        self.weight_bits = weight_bits
        self.sparsity = sparsity
        # self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv1 = SEConv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False, threshold=threshold, sign_threshold=sign_threshold)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 16, layers[0])
        self.layer2 = self._make_layer(block, 32, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 64, layers[2], stride=2)
        self.avgpool = nn.AvgPool2d(8, stride=1)
        # use conv as fc layer (addernet)
        # self.fc = nn.Conv2d(64 * block.expansion, num_classes, 1, bias=False)
        self.fc_1 = nn.Linear(64 * block.expansion, 2)
        self.fc_2 = nn.Linear(2, num_classes)

        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                SEConv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False,
                        threshold=self.threshold, sign_threshold=self.sign_threshold, distribution=self.distribution), # shift
                adder.Adder2D(planes * block.expansion, planes * block.expansion, kernel_size=1, stride=1, bias=False,
                              quantize=self.quantize, weight_bits=self.weight_bits, sparsity=self.sparsity), # add
                nn.BatchNorm2d(planes * block.expansion)
            )

        layers = []
        layers.append(block(inplanes=self.inplanes, planes=planes, threshold=self.threshold,
                            sign_threshold=self.sign_threshold, distribution=self.distribution, stride=stride, downsample=downsample,
                            quantize=self.quantize, weight_bits=self.weight_bits, sparsity=self.sparsity))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(inplanes=self.inplanes, planes=planes, threshold=self.threshold, sign_threshold=self.sign_threshold,
                                distribution=self.distribution, quantize=self.quantize, weight_bits=self.weight_bits, sparsity=self.sparsity))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        # x = self.avgpool(x)
        x = F.avg_pool2d(x, x.size()[3])
        x = x.view(x.size(0), -1)
        x = self.fc_1(x)
        x = self.fc_2(x)
        return x.view(x.size(0), -1)

def resnet20_shiftadd_se(threshold, sign_threshold, distribution, num_classes=10, **kwargs):
    return ResNet(BasicBlock, [3, 3, 3], num_classes=num_classes, threshold=threshold, sign_threshold=sign_threshold, distribution=distribution, **kwargs)

def resnet20_shiftadd_se_vis(threshold, sign_threshold, distribution, num_classes=10, **kwargs):
    return ResNet_vis(BasicBlock, [3, 3, 3], num_classes=num_classes, threshold=threshold, sign_threshold=sign_threshold, distribution=distribution, **kwargs)
