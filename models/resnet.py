'''ResNet in PyTorch.

For Pre-activation ResNet, see 'preact_resnet.py'.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from adder.quantize import quantize, quantize_grad, QuantMeasure, calculate_qparams

__all__ = ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152', 'resnet20', 'resnet32', 'resnet44', 'resnet56', 'resnet110', 'resnet1202']

class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


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
            print('?')
            # apply mask
            self.weight.data = self.weight.data * self.weight_mask.data

        if self.quantize == True:
            # quantization v2
            print('?')
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
            print('?')
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
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, option='A', quantize=False, weight_bits=8, sparsity=0):
        super(BasicBlock, self).__init__()
        # self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.conv1 = conv3x3(in_planes, planes, stride=stride, quantize=quantize, weight_bits=weight_bits, sparsity=sparsity)
        self.bn1 = nn.BatchNorm2d(planes)
        # self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv2 = conv3x3(planes, planes, stride=1, quantize=quantize, weight_bits=weight_bits, sparsity=sparsity)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            """
            For CIFAR10 ResNet paper uses option A.
            """
            if option == 'A':
                self.shortcut = LambdaLayer(lambda x:
                                            F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes//4, planes//4), "constant", 0))
            elif option == 'B':
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(self.expansion*planes)
                )
            else:
                print("ResNet option should be either 'A' or 'B'. Option passed was: ", option)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, option='A', quantize=False, weight_bits=8, sparsity=0):
        super(ResNet, self).__init__()
        """
        For CIFAR10 ResNet paper uses option A.
        """
        self.quantize = quantize
        self.weight_bits = weight_bits
        self.sparsity = sparsity
        self.option = option
        if self.option == 'A':
            self.in_planes = 16

            self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(16)
            self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
            self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
            self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
            self.linear = nn.Linear(64, num_classes)
        elif self.option == 'B':
            self.in_planes = 64

            self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(64)
            self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
            self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
            self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
            self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
            self.linear = nn.Linear(512*block.expansion, num_classes)
        else:
            print("ResNet option should be either 'A' or 'B'. Option passed was: ", self.option)

        # init (new add)
        for m in self.modules():
            if isinstance(m, Mult2D):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, quantize=self.quantize, weight_bits=self.weight_bits, sparsity=self.sparsity))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        if self.option == 'A':
            out = F.avg_pool2d(out, out.size()[3])
        else: # if self.option == 'B':
            out = self.layer4(out)
            out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

# The following models use the architecture mentioned in
# the original ResNet paper on how the architecture should
# be for CIFAR10
# def resnet20(num_classes=10):
#     return ResNet(BasicBlock, [3, 3, 3], option='A', num_classes=num_classes)

def resnet20(num_classes=10, quantize=False, weight_bits=8, sparsity=0, **kwargs):
    return ResNet(BasicBlock, [3, 3, 3], num_classes=num_classes, quantize=quantize,
                    weight_bits=weight_bits, sparsity=sparsity)

def resnet32():
    return ResNet(BasicBlock, [5, 5, 5], option='A')

def resnet44():
    return ResNet(BasicBlock, [7, 7, 7], option='A')

def resnet56():
    return ResNet(BasicBlock, [9, 9, 9], option='A')

def resnet110():
    return ResNet(BasicBlock, [18, 18, 18], option='A')

def resnet1202():
    return ResNet(BasicBlock, [200, 200, 200], option='A')

# The following models use the Imagenet architecture
# which is not consistent with the original ResNet paper's
# description on how the models should look like for CIFAR10
def resnet18():
    return ResNet(BasicBlock, [2,2,2,2], option='B')

def resnet34():
    return ResNet(BasicBlock, [3,4,6,3], option='B')

def resnet50():
    return ResNet(Bottleneck, [3,4,6,3], option='B')

def resnet101():
    return ResNet(Bottleneck, [3,4,23,3], option='B')

def resnet152():
    return ResNet(Bottleneck, [3,8,36,3], option='B')

def test():
    net = resnet18()
    y = net(torch.randn(1,3,32,32))
    print(y.size())

# test()