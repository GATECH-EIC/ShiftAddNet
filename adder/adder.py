'''
Refer to AdderNet code.
Efficient CUDA implementation for AdderNet training.
'''
import torch
import torch.nn as nn
# import adder_cuda
import numpy as np
from torch.autograd import Function
from .quantize import quantize, quantize_grad, QuantMeasure, calculate_qparams
import deepshift.ste as ste

from torch.utils.cpp_extension import load
adder_cuda = load(
  'adder_cuda', ['adder/adder_cuda.cpp', 'adder/adder_cuda_kernel.cu'], verbose=True)


def get_conv2d_output_shape(input, weight, stride, padding):
    n_filters, d_filter, h_filter, w_filter = weight.size()
    n_x, d_x, h_x, w_x = input.size()

    h_out = (h_x - h_filter + 2 * padding) // stride + 1
    w_out = (w_x - w_filter + 2 * padding) // stride + 1

    return (n_x, n_filters, h_out, w_out)

## quantization v1
def round_weight_to_fixed(input, bits=16):
    # print('before quantize: ', input)
    if bits == 1:
        return torch.sign(input)
    S = 2. ** (bits - 1)
    if bits > 15 or bits == 1:
      delta = 0
    else:
      delta = 1. / S
    max_val = 1 - delta
    min_val = delta - 1

    input_clamp = torch.clamp(input, min_val, max_val)
    input_round = torch.round(input_clamp * S) / S
    # print('after quantize: ', input_round)
    return input_round

def round_act_to_fixed(input, bits=16):
    if bits == 1:
        return torch.sign(input)
    S = 2. ** (bits - 1)

    input_round = torch.round(input * S) / S

    return input_round

# def shift(x):
#     #TODO: edge case, when x contains 0
#     return 2.**torch.round(torch.log2(x))

# def S(bits):
#     return 2.**(bits-1)

# def C(x, bits):
#     if bits > 15 or bits == 1:
#         delta = 0
#     else:
#         delta = 1. / S(bits)
#     upper = 1  - delta
#     lower = -1 + delta
#     # upper = x.abs().max()
#     # lower = - upper
#     return torch.clamp(x, lower, upper)

# def Q(x, bits):
#     assert bits != -1
#     if bits==1:
#         return torch.sign(x)
#     # if bits > 15:
#     #     return x
#     return torch.round(x*S(bits))/S(bits)

# def SR(x):
#     r = torch.cuda.FloatTensor(*x.size()).uniform_()
#     return torch.floor(x+r)

# def QE(x, bits=32):
#     max_entry = x.abs().max()
#     if max_entry == 0:
#         return x
#     assert max_entry != 0, "QE blow"
#     x /= shift(max_entry)
#     return Q(C(x, bits), bits)

# def QG(x, bits_G=32):
#     max_entry = x.abs().max()
#     assert max_entry != 0, "QG blow"
#     x /= shift(max_entry)
#     norm = SR(x)
#     return norm / S(bits_G)


bitsU = 16
def scale(x):
    scale = torch.max(torch.abs(x))
    result = 2.**torch.round(torch.log2(scale))
    return result

def delta(bits):
    result = (2.**(1-bits))
    return result

def clip(x, bits):
    if bits >= 32:
        step = 0
    else:
        step = delta(bits)
    ceil  = 1 - step
    floor = step - 1
    result = torch.clamp(x, floor, ceil)
    return result

def quant(x, bits):
    if bits >= 32:
        result = x
    else:
        result = torch.round(x/delta(bits))*delta(bits)
    return result

def qw(x, bitsW):
    bits = bitsW
    if bits >= 32:
        result = x
    else:
        result = quant(x,bits) # remove clip for adding layer
    return result

def qa(x, bitsA):
    bits = bitsA
    if bits >= 32:
        result = x
    else:
        result = quant(x,bits)
    return result

class RoundWeightFixedPoint(Function):
    @staticmethod
    def forward(ctx, input, bits):
        return qw(input, bits)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None

class RoundActFixedPoint(Function):
    @staticmethod
    def forward(ctx, input, bits):
        return qa(input, bits)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None

def round_weight_fixed_point(input, bits):
    return RoundWeightFixedPoint.apply(input, bits)

def round_act_fixed_point(input, bits):
    return RoundActFixedPoint.apply(input, bits)


def qe(x, bitsE):
    bits = bitsE
    if bits >= 32:
        result = x
    else:
        dscale = scale(x)
        result = dscale*clip(quant(x/dscale,bits),bits)
    return result

def qg(x, bitsG):
    bits = bitsG
    if bits >= 32:
        result = x
    else:
        # dscale = scale(x)
        # x = x / dscale
        # factor = 128
        # bitsR = 32
        # norm = quant(factor * x, bitsR)
        #
        # norm_sign = torch.sign(norm)
        # norm_abs = torch.abs(norm)
        # norm_int = torch.floor(norm_abs)
        # norm_float = norm_abs - norm_int
        # rand_float = torch.FloatTensor(*x.size()).uniform_()
        # norm = norm_sign * ( norm_int + 0.5 * (torch.sign(norm_float - rand_float) + 1) )
        # norm = torch.clamp(norm,-factor+1,factor-1)
        # result = quant(norm*delta(bits)/128,15)

        dscale = scale(x)
        x = x / dscale
        factor = 128
        bitsR = 32
        norm = quant(factor * x, bitsR)

        norm_sign = torch.sign(norm)
        norm_abs = torch.abs(norm)
        norm_int = torch.floor(norm_abs)
        norm_float = norm_abs - norm_int
        rand_float = torch.FloatTensor(*x.size()).uniform_().cuda()
        norm = norm_sign * ( norm_int + 0.5 * (torch.sign(norm_float - rand_float) + 1) )
        norm = torch.clamp(norm,-factor+1,factor-1)
        result = norm/128

    return result


########

class Adder2DFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx,
                input,
                weight,
                kernel_size,
                stride,
                padding,
                eta, quantize,
                weight_bits,
                quantize_v):
        ctx.save_for_backward(input, weight)
        ctx.kernel_size = kernel_size
        ctx.stride = stride
        ctx.padding = padding
        ctx.eta = eta
        ctx.quantize = quantize
        ctx.weight_bits = weight_bits
        ctx.quantize_v = quantize_v

        output = input.new_zeros(
            get_conv2d_output_shape(input, weight, stride, padding))
        adder_cuda.forward(input,
                           weight,
                           output,
                           kernel_size, kernel_size,
                           stride, stride,
                           padding, padding)

        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, weight = ctx.saved_tensors
        grad_input = grad_weight = None
        eta, kernel_size, stride, padding = (
            ctx.eta, ctx.kernel_size, ctx.stride, ctx.padding
        )

        # quantize grad_output v1
        if ctx.quantize == True and ctx.quantize_v == 'wageubn':
            grad_output = qe(grad_output, ctx.weight_bits)

        # input
        if ctx.needs_input_grad[0]:
            grad_input = torch.zeros_like(input)
            adder_cuda.backward_input(grad_output,
                                      input,
                                      weight,
                                      grad_input,
                                      kernel_size, kernel_size,
                                      stride, stride,
                                      padding, padding)

        # weight
        if ctx.needs_input_grad[1]:
            grad_weight = torch.zeros_like(weight)
            adder_cuda.backward_weight(grad_output,
                                       input,
                                       weight,
                                       grad_weight,
                                       kernel_size, kernel_size,
                                       stride, stride,
                                       padding, padding)
            grad_weight = eta * np.sqrt(grad_weight.numel()) / torch.norm(grad_weight) * grad_weight

            if ctx.quantize == True and ctx.quantize_v == 'wageubn':
                grad_weight = qg(grad_weight, ctx.weight_bits)

        return grad_input, grad_weight, None, None, None, None, None, None, None


class Adder2D(nn.Module):

    def __init__(self,
                 input_channel,
                 output_channel,
                 kernel_size,
                 stride = 1,
                 padding = 0,
                 bias = False,
                 eta = 0.2,
                 quantize=False, weight_bits=8, sparsity=0, momentum=0.9, quantize_v='sbm'):
        super(Adder2D, self).__init__()
        self.stride = stride
        self.padding = padding
        self.input_channel = input_channel
        self.output_channel = output_channel
        self.kernel_size = kernel_size
        self.eta = eta
        self.quantize = quantize
        self.weight_bits = weight_bits
        self.sparsity = sparsity
        self.quantize_v = quantize_v
        # print(quantize_v)

        if self.quantize:
            self.quantize_input_fw = QuantMeasure(shape_measure=(1, 1, 1, 1), flatten_dims=(1, -1), momentum=momentum)

        self.adder = torch.nn.Parameter(
            nn.init.normal_(torch.randn(
                output_channel,input_channel,kernel_size,kernel_size)))
        self.qadder = None
        self.bias = bias
        if bias:
            self.b = torch.nn.Parameter(
                nn.init.uniform_(torch.zeros(output_channel)))

        if self.sparsity != 0:
            self.s = torch.nn.Parameter(
                nn.init.uniform_(torch.randn(
                  output_channel,input_channel,kernel_size,kernel_size)))
            self.s.data.uniform_(0, 1)
            self.register_buffer('adder_mask', torch.Tensor(*self.adder.size()).float())
            self.set_mask()

        if self.quantize is True:
            print(self.quantize)
            print('quantize adder layer to {} bits.'.format(self.weight_bits))

    def forward(self, input):
        if self.sparsity != 0:
            # apply mask
            self.adder.data = self.adder.data * self.adder_mask.data

        if self.quantize is True:
            # shift_range = (-1 * (2 ** (self.weight_bits - 1) - 1), 0)
            # self.adder.data = ste.clampabs(self.adder.data, 2**shift_range[0], 2**shift_range[1])
            # weight_q = ste.round_power_of_2(self.adder, 'deterministic')

            # quantization v1
            if self.quantize_v == 'wageubn':
                self.qadder = round_weight_fixed_point(self.adder, self.weight_bits)
                input_q = round_act_fixed_point(input, self.weight_bits)

            # quantization v2
            if self.quantize_v == 'sbm':
                input_q = self.quantize_input_fw(input, self.weight_bits)
                weight_qparams = calculate_qparams(self.adder, num_bits=self.weight_bits, flatten_dims=(1, -1), reduce_dim=None)
                self.qadder = quantize(self.adder, qparams=weight_qparams)
            bias_fixed_point = None
            output = Adder2DFunction.apply(input_q,
                                           self.qadder,
                                           self.kernel_size,
                                           self.stride,
                                           self.padding,
                                           self.eta,
                                           self.quantize,
                                           self.weight_bits,
                                           self.quantize_v)
            if self.quantize_v == 'sbm':
                output = quantize_grad(output, num_bits=self.weight_bits, flatten_dims=(1, -1))
        else:
            output = Adder2DFunction.apply(input,
                                           self.adder,
                                           self.kernel_size,
                                           self.stride,
                                           self.padding,
                                           self.eta,
                                           self.quantize,
                                           self.weight_bits,
                                           self.quantize_v)
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
        self.adder_mask.data = (self.s > self.sparsity).float()
        assert self.adder_mask.requires_grad == False