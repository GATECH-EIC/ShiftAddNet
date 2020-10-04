# 2020.01.10-Replaced conv with adder
#         Haoran & Xiaohan

# from models import adder
# from adder import adder
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from adder.quantize import quantize, quantize_grad, QuantMeasure, calculate_qparams

__all__ = ['resnet20_mult', 'LeNet_add_vis', 'resnet20_mult_vis']



class Mult2D(nn.Module):

    def __init__(self,
                 input_channel,
                 output_channel,
                 kernel_size,
                 stride = 1,
                 padding = 0,
                 bias = False,
                 quantize=False, weight_bits=8, sparsity=0, momentum=0.9):
        super(Mult2D, self).__init__()
        self.stride = stride
        self.padding = padding
        self.input_channel = input_channel
        self.output_channel = output_channel
        self.kernel_size = kernel_size
        self.quantize = quantize
        self.weight_bits = weight_bits
        self.sparsity = sparsity

        self.quantize_input_fw = QuantMeasure(shape_measure=(1, 1, 1, 1), flatten_dims=(1, -1), momentum=momentum)

        self.weight = torch.nn.Parameter(
            nn.init.normal_(torch.randn(
                output_channel,input_channel,kernel_size,kernel_size)))
        self.qweight = None
        self.bias = bias
        if bias:
            self.b = torch.nn.Parameter(
                nn.init.uniform_(torch.zeros(output_channel)))

        if self.sparsity != 0:
            self.s = torch.nn.Parameter(
                nn.init.uniform_(torch.randn(
                  output_channel,input_channel,kernel_size,kernel_size)))
            self.s.data.uniform_(0, 1)
            self.register_buffer('weight_mask', torch.Tensor(*self.weight.size()).float())
            self.set_mask()

    def forward(self, input):
        if self.sparsity != 0:
            # apply mask
            self.weight.data = self.weight.data * self.weight_mask.data

        if self.quantize == True:
            # quantization v2
            input_q = self.quantize_input_fw(input, self.weight_bits)
            weight_qparams = calculate_qparams(self.weight, num_bits=self.weight_bits, flatten_dims=(1, -1), reduce_dim=None)
            self.qweight = quantize(self.weight, qparams=weight_qparams)
            bias_fixed_point = None
            output = F.conv2d(input_q,
                               self.qweight,
                               None,
                               self.stride,
                               self.padding)
            output = quantize_grad(output, num_bits=self.weight_bits, flatten_dims=(1, -1))
        else:
            output = F.conv2d(input,
                               self.weight,
                               None,
                               self.stride,
                               self.padding)
        if self.bias:
            output += self.b.unsqueeze(0).unsqueeze(2).unsqueeze(3)

        return output

    def round_weight_each_step(self, weight, bits=16):
        # print('before quantize: ', input)
        # quantization v1
        # if bits == 1:
        #     return torch.sign(weight)
        # S = 2. ** (bits - 1)
        # if bits > 15 or bits == 1:
        #   delta = 0
        # else:
        #   delta = 1. / S
        # max_val = 1 - delta
        # min_val = delta - 1

        # weight_clamp = torch.clamp(weight, min_val, max_val)
        # qweight = torch.round(weight_clamp * S) / S
        # print('after quantize: ', input_round)

        # quantization v2
        weight_qparams = calculate_qparams(weight, num_bits=bits, flatten_dims=(1, -1), reduce_dim=None)
        qweight = quantize(weight, qparams=weight_qparams)
        weight_unique = torch.unique(qweight[0])
        print('add weight range:', weight_unique.size()[0]-1)
        return qweight

    def set_mask(self):
        # random fix zero
        self.weight_mask.data = (self.s > self.sparsity).float()
        assert self.weight_mask.requires_grad == False

def conv3x3(in_planes, out_planes, stride=1, quantize=False, weight_bits=8, sparsity=0):
    " 3x3 convolution with padding "
    return Mult2D(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False,
                         quantize=quantize, weight_bits=weight_bits, sparsity=sparsity)


class BasicBlock(nn.Module):
    expansion=1

    def __init__(self, inplanes, planes, stride=1, downsample=None, quantize=False, weight_bits=8, sparsity=0):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride=stride, quantize=quantize, weight_bits=weight_bits, sparsity=sparsity)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, quantize=quantize, weight_bits=weight_bits, sparsity=sparsity)
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

    def __init__(self, block, layers, num_classes=10, quantize=False, weight_bits=8, sparsity=0):
        super(ResNet, self).__init__()
        self.quantize = quantize
        self.weight_bits = weight_bits
        self.sparsity = sparsity
        self.inplanes = 16
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(8)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 8, layers[0])
        self.layer2 = self._make_layer(block, 16, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 16, layers[2], stride=2)
        self.avgpool = nn.AvgPool2d(8, stride=1)
        # use conv as fc layer (addernet)
        self.fc = nn.Conv2d(16 * block.expansion, num_classes, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(num_classes)


        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                Mult2D(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False,
                              quantize=self.quantize, weight_bits=self.weight_bits, sparsity=self.sparsity), # adder.Adder2D
                nn.BatchNorm2d(planes * block.expansion)
            )

        layers = []
        layers.append(block(inplanes = self.inplanes, planes = planes, stride = stride, downsample = downsample,
                            quantize=self.quantize, weight_bits=self.weight_bits, sparsity=self.sparsity))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(inplanes = self.inplanes, planes = planes, quantize=self.quantize, weight_bits=self.weight_bits, sparsity=self.sparsity))

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

    def __init__(self, block, layers, num_classes=10, quantize=False, weight_bits=8, sparsity=0):
        super(ResNet_vis, self).__init__()
        self.quantize = quantize
        self.weight_bits = weight_bits
        self.sparsity = sparsity
        self.inplanes = 16
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 16, layers[0])
        self.layer2 = self._make_layer(block, 32, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 64, layers[2], stride=2)
        self.avgpool = nn.AvgPool2d(8, stride=1)
        # use conv as fc layer (addernet)
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
                Mult2D(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False,
                              quantize=self.quantize, weight_bits=self.weight_bits, sparsity=self.sparsity), # adder.Adder2D
                nn.BatchNorm2d(planes * block.expansion)
            )

        layers = []
        layers.append(block(inplanes = self.inplanes, planes = planes, stride = stride, downsample = downsample,
                            quantize=self.quantize, weight_bits=self.weight_bits, sparsity=self.sparsity))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(inplanes = self.inplanes, planes = planes, quantize=self.quantize, weight_bits=self.weight_bits, sparsity=self.sparsity))

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

class LeNet_vis(nn.Module):

    def __init__(self, num_classes=10):
        super(LeNet_vis, self).__init__()
        self.conv1_1 = adder.Adder2D(1, 32, kernel_size=5, padding=2)
        self.prelu1_1 = nn.PReLU()
        self.conv1_2 = adder.Adder2D(32, 32, kernel_size=5, padding=2)
        self.prelu1_2 = nn.PReLU()
        # self.conv2_1 = adder.Adder2D(32, 64, kernel_size=5, padding=2)
        # self.prelu2_1 = nn.PReLU()
        # self.conv2_2 = adder.Adder2D(64, 64, kernel_size=5, padding=2)
        # self.prelu2_2 = nn.PReLU()
        # self.conv3_1 = adder.Adder2D(64, 128, kernel_size=5, padding=2)
        # self.prelu3_1 = nn.PReLU()
        # self.conv3_2 = adder.Adder2D(128, 128, kernel_size=5, padding=2)
        # self.prelu3_2 = nn.PReLU()
        self.preluip1 = nn.PReLU()
        self.ip1 = nn.Linear(32*3*3, 2)
        self.ip2 = nn.Linear(2, 10, bias=False)

    def forward(self, x):
        x = self.prelu1_1(self.conv1_1(x))
        x = self.prelu1_2(self.conv1_2(x))
        x = F.max_pool2d(x,2)
        # x = self.prelu2_1(self.conv2_1(x))
        # x = self.prelu2_2(self.conv2_2(x))
        x = F.max_pool2d(x,2)
        # x = self.prelu3_1(self.conv3_1(x))
        # x = self.prelu3_2(self.conv3_2(x))
        x = F.max_pool2d(x,2)
        x = x.view(-1, 32*3*3)
        ip1 = self.preluip1(self.ip1(x))
        ip2 = self.ip2(ip1)
        # return ip1, F.log_softmax(ip2, dim=1)
        return ip1, ip2


def resnet20_mult(num_classes=10, quantize=False, weight_bits=8, sparsity=0, **kwargs):
    return ResNet(BasicBlock, [3, 3, 3], num_classes=num_classes, quantize=quantize,
                    weight_bits=weight_bits, sparsity=sparsity)

def resnet20_mult_vis(num_classes=10, quantize=False, weight_bits=8, sparsity=0, **kwargs):
    return ResNet_vis(BasicBlock, [3, 3, 3], num_classes=num_classes, quantize=quantize,
                    weight_bits=weight_bits, sparsity=sparsity)


def LeNet_add_vis(num_classes=10, **kwargs):
    return LeNet_vis(num_classes=num_classes)
