'''
 # @ Author: Xiaohan Chen
 # @ Email: chernxh@tamu.edu
 # @ Create Time: 2019-07-11 02:06:24
 # @ Modified by: Xiaohan Chen
 # @ Modified time: 2019-08-17 22:10
 # @ Description:
 '''

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

from torch.nn.modules.utils import _pair
from se_shift.utils_quantize import quantize, quantize_grad, conv2d_biprec, QuantMeasure
from se_shift.utils_quantize import sparsify_and_nearestpow2
from se_shift.alg import VEC_2_SHAPE
from torch.autograd import Function
import numpy as np
from deepshift import ste

def round_act_to_fixed(input, bits=16):
    if bits == 1:
        return torch.sign(input)
    S = 2. ** (bits - 1)

    input_round = torch.round(input * S) / S

    return input_round

class RoundActFixedPoint(Function):
    @staticmethod
    def forward(ctx, input, bits):
        return round_act_to_fixed(input, bits)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None

def round_act_fixed_point(input, bits):
    return RoundActFixedPoint.apply(input, bits)


def dynamic_range_for_sign(sign, threshold):
    # print(sign, threshold)
    sign[sign < -threshold] = -1
    sign[sign > threshold] = 1
    sign[(-threshold <= sign) & (sign <= threshold)] = 0
    return sign

class RoundFunction(Function):
    @staticmethod
    def forward(ctx, input, threshold):
        return dynamic_range_for_sign(input, threshold)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None

def round(input, threshold):
    return RoundFunction.apply(input, threshold)

class SEConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=False, size_splits=64,
                 threshold=5e-3, sign_threshold=0.5, distribution='uniform'):
        super(SEConv2d, self).__init__()
        if in_channels % groups != 0:
            raise ValueError('in_channels must be divisible by groups')
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)
        self.groups = groups
        self.size_splits = size_splits
        self.sign_threshold = sign_threshold
        self.distribution = distribution
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.weight = torch.nn.Parameter(nn.init.normal_(torch.randn(
            self.out_channels, self.in_channels, kernel_size[0], kernel_size[1])))
        self.p = torch.nn.Parameter(nn.init.uniform_(torch.randn(
            self.out_channels, self.in_channels, kernel_size[0], kernel_size[1])))
        self.s = torch.nn.Parameter(nn.init.uniform_(torch.randn(
            self.out_channels, self.in_channels, kernel_size[0], kernel_size[1])))
        self.register_buffer('mask', torch.Tensor(*self.weight.size()).float())
        self.threshold = threshold
        for i in range(-10, 1):
            if 2**i >= threshold:
                self.min_p = -i
                break
        self.shift_range = (-1 * self.min_p, 0)

        self.reset_parameters()

    def reset_dweight_counter(self):
        self.dweight_counter = self.weight.new_zeros(self.weight.size()).float()

    def reset_parameters(self):
        # n = self.in_channels
        # init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        # init.kaiming_normal_(self.weight, mode='fan_out', nonlinearity='relu')
        #
        if self.distribution == 'kaiming_normal':
            init.kaiming_normal_(self.weight, mode='fan_out', nonlinearity='relu')
            self.set_mask() # quantize
            self.s.data.uniform_(-1, 1)
            sign = ste.sign(round(self.s, self.sign_threshold))
            self.weight.data *= abs(sign)
        else:
            if self.distribution == 'uniform':
                self.p.data.uniform_(-self.min_p - 0.5, -1 + 0.5)
            elif self.distribution == 'normal':
                self.p.data.normal_(-self.min_p / 2, 1)
            self.p.data = ste.clamp(self.p.data, *self.shift_range)
            self.p.data = ste.round(self.p.data, 'deterministic')
            self.s.data.uniform_(-1, 1)
            sign = ste.sign(round(self.s, self.sign_threshold))
            # self.weight.data = torch.sign(self.weight) * (2 ** self.p.data)
            self.weight.data = sign * (2 ** self.p.data)

        if self.bias is not None:
            print('use bias')
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def extra_repr(self):
        s = ('{in_channels}, {out_channels}, kernel_size={kernel_size}'
             ', stride={stride}')
        if self.padding != (0,) * len(self.padding):
            s += ', padding={padding}'
        if self.dilation != (1,) * len(self.dilation):
            s += ', dilation={dilation}'
        # if self.output_padding != (0,) * len(self.output_padding):
        #     s += ', output_padding={output_padding}'
        if self.groups != 1:
            s += ', groups={groups}'
        if self.bias is None:
            s += ', bias=False'
        return s.format(**self.__dict__)

    def set_mask(self):
        self.weight.data = self.sparsify_and_quantize_weight(mask=False)
        self.mask.data = (self.weight != 0.0).float()
        assert self.mask.requires_grad == False

    def sparsify_and_quantize_weight(self, mask=True):
        qweight = sparsify_and_nearestpow2(self.weight, self.threshold)
        if mask:
            qweight = qweight * self.mask
        return qweight

    def get_weight(self, mask=True):
        qweight = self.sparsify_and_quantize_weight(mask=mask)
        return qweight

    def forward(self, input):
        # Get the weight
        # weight = self.get_weight(mask=True)
        weight = self.weight
        input = round_act_fixed_point(input, bits=16)
        # intput = ste.round_fixed_point(input)
        # weight = self.get_weight(mask=False)

        output = F.conv2d(input, weight, self.bias, self.stride,
                          self.padding, self.dilation, self.groups)

        return output
