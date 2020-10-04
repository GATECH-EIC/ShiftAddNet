from adder import adder
import torch
import torch.nn as nn
import torch.nn.functional as F
from adder.quantize import quantize, quantize_grad, QuantMeasure, calculate_qparams

__all__ = ['CNN_mult']

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
                output_channel,input_channel,kernel_size[0],kernel_size[1])))
        self.qweight = None
        self.bias = bias
        if bias:
            self.b = torch.nn.Parameter(
                nn.init.uniform_(torch.zeros(output_channel)))

        if self.sparsity != 0:
            self.s = torch.nn.Parameter(
                nn.init.uniform_(torch.randn(
                  output_channel,input_channel,kernel_size[0],kernel_size[1])))
            self.s.data.uniform_(0, 1)
            self.register_buffer('weight_mask', torch.Tensor(*self.weight.size()).float())
            self.set_mask()

    def forward(self, input):
        if self.sparsity != 0:
            # apply mask
            self.weight.data = self.weight.data * self.weight_mask.data

        if self.quantize == True:
            # print('quantize to {} bits'.format(self.weight_bits))
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

def conv_mult(in_planes, out_planes, kernel_size=(3,3), stride=1, padding=0, quantize=False, weight_bits=8, sparsity=0):
    " 3x3 convolution with padding "
    return Mult2D(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, bias=False,
                         quantize=quantize, weight_bits=weight_bits, sparsity=sparsity)


class CNN(nn.Module):
    def __init__(self, num_classes, quantize=False, weight_bits=8, sparsity=0):
        super(CNN, self).__init__()
        self.quantize = quantize
        self.weight_bits = weight_bits
        self.sparsity = sparsity

        self.conv1 = conv_mult(1, 5, kernel_size=(12, 5), quantize=self.quantize, weight_bits=self.weight_bits, sparsity=self.sparsity)
        self.bn1 = nn.BatchNorm2d(5)
        self.conv2 = conv_mult(5, 10, kernel_size=(5, 1), quantize=self.quantize, weight_bits=self.weight_bits, sparsity=self.sparsity)
        self.bn2 = nn.BatchNorm2d(10)
        # self.conv3 = nn.Conv2d(36, 24, kernel_size=(12, 1))
        self.pool1 = nn.MaxPool2d((4,4))
        self.pool2 = nn.MaxPool2d((2,2))
        self.fc1 = conv_mult(590, num_classes, kernel_size=(1,1), quantize=self.quantize, weight_bits=self.weight_bits, sparsity=self.sparsity)
        self.fc2 = nn.BatchNorm2d(num_classes)

    def forward(self, inputs):
        x = self.pool1(F.relu(self.bn1(self.conv1(inputs))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        # x = self.pool(F.relu(self.conv3(x)))
        # x = torch.flatten(x, start_dim=1)
        x = x.view(x.size(0), -1)
        x = torch.unsqueeze(x, dim=2)
        x = torch.unsqueeze(x, dim=3)

        x = self.fc1(x)
        x = self.fc2(x)
        # return F.softmax(x)
        return x.view(x.size(0), -1)

def CNN_mult(num_classes=10, quantize=False, weight_bits=8, sparsity=0, quantize_v='sbm', **kwargs):
    return CNN(num_classes, quantize=quantize, weight_bits=weight_bits, sparsity=sparsity)