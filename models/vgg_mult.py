'''
Modified from https://github.com/pytorch/vision.git
Copy from https://github.com/Jerry-2017/DoubleBlindImage/blob/master/code/gaussiansmooth/vgg.py
'''
import math
from adder import adder
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.autograd import Function
from adder.quantize import quantize, quantize_grad, QuantMeasure, calculate_qparams

__all__ = ['Conv6',
    'VGG', 'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn',
    'vgg19_bn', 'vgg19', 'vgg11_nd', 'vgg11_nd_s', 'vgg13_nd', 'vgg13_nd_s', 'vgg16_nd', 'vgg16_nd_s', 'vgg19_nd', 'vgg19_nd_s',
    'vgg11_nd_ss', 'vgg13_nd_ss', 'vgg16_nd_ss', 'vgg19_small_mult',
]

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


class VGG(nn.Module):
    '''
    VGG model
    '''
    def __init__(self, features, num_classes=10, dropout=True, small=False, supersmall=False):
        super(VGG, self).__init__()
        self.features = features
        cls_layers = []
        if dropout or supersmall:
            cls_layers.append(nn.Dropout())
        if not (small or supersmall):
            cls_layers.append(nn.Linear(512, 512))
            cls_layers.append(nn.ReLU())
            if dropout:
                cls_layers.append(nn.Dropout())
        if not supersmall:
            cls_layers.append(nn.Linear(512, 512))
            cls_layers.append(nn.ReLU())
        cls_layers.append(nn.Linear(512, num_classes))

        self.classifier = nn.Sequential(*cls_layers)
         # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                m.bias.data.zero_()

        for m in self.modules():
            if isinstance(m, Mult2D):
                n = m.kernel_size * m.kernel_size * m.output_channel
                # m.weight.data.normal_(0, math.sqrt(2. / n))
                nn.init.kaiming_normal(m.weight, mode='fan_out')
                # m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


    def forward(self, x):
        x = self.features(x)
        # for conv in self.features:
        #     x = conv(x)
        #     print(x.shape)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


def make_layers(cfg, quantize=False, weight_bits=8, sparsity=0, quantize_v='sbm', batch_norm=True):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            # conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            conv2d = conv3x3(in_channels, v, quantize=False, weight_bits=8, sparsity=0,)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=False)]
            else:
                layers += [conv2d, nn.ReLU(inplace=False)]
            in_channels = v
    return nn.Sequential(*layers)


cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M',
          512, 512, 512, 512, 'M'],
    '6': [64, 'M', 64, 'M', 128, 'M', 128, 'M', 256, 'M', 512],
}

def Conv6(num_classes=10, **kwargs):
    return VGG(make_layers(cfg['6'], **kwargs), num_classes=num_classes)

def vgg11():
    """VGG 11-layer model (configuration "A")"""
    return VGG(make_layers(cfg['A']))

def vgg11_nd():
    """VGG 11-layer model (configuration "A")"""
    return VGG(make_layers(cfg['A']), dropout=False)

def vgg11_nd_s():
    """VGG 11-layer model (configuration "A")"""
    return VGG(make_layers(cfg['A']), dropout=False, small=True)

def vgg11_nd_ss():
    """VGG 11-layer model (configuration "A")"""
    return VGG(make_layers(cfg['A']), dropout=False, small=True, supersmall=True)


def vgg11_bn():
    """VGG 11-layer model (configuration "A") with batch normalization"""
    return VGG(make_layers(cfg['A'], batch_norm=True))


def vgg13():
    """VGG 13-layer model (configuration "B")"""
    return VGG(make_layers(cfg['B']))

def vgg13_nd():
    """VGG 13-layer model (configuration "B")"""
    return VGG(make_layers(cfg['B']), dropout=False)

def vgg13_nd_s():
    """VGG 13-layer model (configuration "B")"""
    return VGG(make_layers(cfg['B']), dropout=False, small=True)

def vgg13_nd_ss():
    """VGG 13-layer model (configuration "B")"""
    return VGG(make_layers(cfg['B']), dropout=False, small=True, supersmall=True)


def vgg13_bn():
    """VGG 13-layer model (configuration "B") with batch normalization"""
    return VGG(make_layers(cfg['B'], batch_norm=True))


def vgg16():
    """VGG 16-layer model (configuration "D")"""
    return VGG(make_layers(cfg['D']))

def vgg16_nd():
    """VGG 16-layer model (configuration "D")"""
    return VGG(make_layers(cfg['D']), dropout=False)

def vgg16_nd_s():
    """VGG 16-layer model (configuration "D")"""
    return VGG(make_layers(cfg['D']), dropout=False, small=True)

def vgg16_nd_ss():
    """VGG 16-layer model (configuration "D")"""
    return VGG(make_layers(cfg['D']), dropout=False, small=True, supersmall=True)


def vgg16_bn():
    """VGG 16-layer model (configuration "D") with batch normalization"""
    return VGG(make_layers(cfg['D'], batch_norm=True))


def vgg19():
    """VGG 19-layer model (configuration "E")"""
    return VGG(make_layers(cfg['E']))

def vgg19_nd():
    """VGG 19-layer model (configuration "E")"""
    return VGG(make_layers(cfg['E']), dropout=False)

def vgg19_small_mult(num_classes=10, **kwargs):
    """VGG 19-layer model (configuration "E")"""
    return VGG(make_layers(cfg['B']), num_classes=num_classes, dropout=False, small=True)

def vgg19_nd_ss():
    """VGG 19-layer model (configuration "E")"""
    return VGG(make_layers(cfg['E']), dropout=False, small=True, supersmall=True)



def vgg19_bn():
    """VGG 19-layer model (configuration 'E') with batch normalization"""
    return VGG(make_layers(cfg['E'], batch_norm=True))