import torch as tt
from torch import nn
from torch.utils.cpp_extension import load
import torch.nn.functional as F

# adder_cuda = load(
#     'adder_cuda', ['adder_cuda.cpp', 'adder_cuda_kernel.cu'], verbose=True)

from adder.adder import Adder2D
from adder.adder_slow import adder2d, adder2d_function

# help(adder_cuda)``1        1`

def check_forward():
    batch_size = 1
    in_channels = 64
    out_channels = 64
    # in_channels = 1
    # out_channels = 1
    in_size = 256
    # in_size = 3
    kernel_size = 3
    padding = 1
    stride = 1
    out_size = (in_size + 2 * padding - kernel_size) // stride + 1
    print(out_size)

    input  = tt.randn(batch_size, in_channels, in_size, in_size).cuda()
    weight = tt.randn(out_channels, in_channels, kernel_size, kernel_size).cuda()
    bias   = tt.randn(out_channels).cuda()
    # output = tt.randn(batch_size, out_channels, out_size, out_size).cuda()

    adder_ref = adder2d(in_channels,
                        out_channels,
                        kernel_size,
                        stride,
                        padding,
                        bias = True).cuda()
    adder_ref.adder.data.copy_(weight)
    adder_ref.b.data.copy_(bias)

    adder = Adder2D(in_channels,
                    out_channels,
                    kernel_size,
                    stride,
                    padding,
                    bias = True,
                    eta = 0.2).cuda()
    adder.adder.data.copy_(weight)
    adder.b.data.copy_(bias)

    # adder_cuda.forward(input,
    #                    weight,
    #                    # bias,
    #                    output,
    #                    kernel_size, kernel_size,
    #                    stride, stride,
    #                    padding, padding)
    adder(input)
    adder_ref(input)
    input.clone()
    weight.clone()
    # output.clone()

    # print(input)
    # print(weight)
    # print("our output: ", output)
    # out_ref = adder2d_function(input, weight, stride, padding)
    # print("addernet ref: ", out_ref)
    # print("by hand no bias: ", -(input - weight).abs().sum())
    # print(F.conv2d(input, weight, bias, padding=padding))
    # out_ref = F.conv2d(input, weight, bias, padding=padding)

    import time

    time_b = time.time()
    # adder_cuda.forward(input,
    #                   weight,
    #                   # bias,
    #                   output,
    #                   kernel_size, kernel_size,
    #                   stride, stride,
    #                   padding, padding)
    output = adder(input)
    time_e = time.time()
    print("hand_written_conv: {:.4f}us".format((time_e - time_b) * 1e6))

    time_b = time.time()
    # out_ref = F.conv2d(input, weight, bias, padding=padding)
    # out_ref = adder2d_function(input, weight, stride, padding)
    out_ref = adder_ref(input)
    time_e = time.time()
    print("builtin_conv: {:.4f}us".format((time_e - time_b) * 1e6))

    print("max error: {:.3e}".format(float((out_ref - output).abs().max())))

    time_b = time.time()
    # adder_cuda.forward(input,
    #                   weight,
    #                   # bias,
    #                   output,
    #                   kernel_size, kernel_size,
    #                   stride, stride,
    #                   padding, padding)
    output = adder(input)
    time_e = time.time()
    print("hand_written_conv: {:.4f}us".format((time_e - time_b) * 1e6))

    time_b = time.time()
    # out_ref = F.conv2d(input, weight, bias, padding=padding)
    # out_ref = adder2d_function(input, weight, stride, padding)
    out_ref = adder_ref(input)
    time_e = time.time()
    print("builtin_conv: {:.4f}us".format((time_e - time_b) * 1e6))

    print("max error: {:.3e}".format(float((out_ref - output).abs().max())))


def check_grad_in():
    batch_size = 1
    in_channels = 64
    out_channels = 64
    in_size = 128
    kernel_size = 3
    padding = 1
    stride = 1
    # batch_size = 1
    # in_channels = 1
    # out_channels = 1
    # in_size = 2
    # kernel_size = 2
    # padding = 0
    # stride = 1
    out_size = (in_size + 2 * padding - kernel_size) // stride + 1
    print(out_size)

    input = tt.randn(batch_size, in_channels, in_size, in_size).cuda()
    grad_input = tt.randn(batch_size, in_channels, in_size, in_size).cuda()
    input.requires_grad = True
    weight = tt.randn(out_channels, in_channels, kernel_size, kernel_size).cuda()
    weight.requires_grad = True
    bias = tt.randn(out_channels).cuda()
    grad_output = tt.randn(batch_size, out_channels, out_size, out_size).cuda()

    adder_ref = adder2d(in_channels,
                        out_channels,
                        kernel_size,
                        stride,
                        padding,
                        bias = True).cuda()
    adder_ref.adder.data.copy_(weight)
    adder_ref.b.data.copy_(bias)

    adder = Adder2D(in_channels,
                    out_channels,
                    kernel_size,
                    stride,
                    padding,
                    bias = True,
                    eta = 0.2).cuda()
    adder.adder.data.copy_(weight)
    adder.b.data.copy_(bias)

    # outref = F.conv2d(input, weight, bias, padding=padding)
    # out_ref = adder2d_function(input, weight, stride, padding)
    out_ref = adder_ref(input)
    out_ref.backward(grad_output)

    grad_clone = input.grad.clone()
    input.grad.zero_()

    # adder_cuda.backward_input(grad_output,
    #                           input,
    #                           weight,
    #                           grad_input,
    #                           kernel_size, kernel_size,
    #                           stride, stride,
    #                           padding, padding)
    output = adder(input)
    output.backward(grad_output)
    grad_input = input.grad.clone()

    # print("input")
    # print(input)
    # print("weight ref")
    # print(adder_ref.adder)
    # print("weight our")
    # print(adder.adder)
    # print("output ref")
    # print(out_ref)
    # print("output our")
    # print(output)
    # print("grad output")
    # print(grad_output)
    # print("grad_in ref")
    # print(grad_clone)
    # print("grad_in our")
    # print(grad_input)

    print(((grad_clone - grad_input)).abs().max())


def check_grad_weight():
    batch_size = 1
    in_channels = 6
    out_channels = 6
    in_size = 128
    kernel_size = 3
    padding = 1
    stride = 1
    # batch_size = 1
    # in_channels = 1
    # out_channels = 1
    # in_size = 1
    # kernel_size = 1
    # padding = 0
    # stride = 1
    out_size = (in_size + 2 * padding - kernel_size) // stride + 1
    print(out_size)

    input = tt.randn(batch_size, in_channels, in_size, in_size).cuda()
    input.requires_grad = True
    weight = tt.randn(out_channels, in_channels, kernel_size, kernel_size).cuda()
    weight.requires_grad = True
    bias = tt.randn(out_channels).cuda()
    grad_output = tt.randn(batch_size, out_channels, out_size, out_size).cuda()

    adder_ref = adder2d(in_channels,
                        out_channels,
                        kernel_size,
                        stride,
                        padding,
                        bias = True).cuda()
    adder_ref.adder.data.copy_(weight)
    adder_ref.b.data.copy_(bias)

    adder = Adder2D(in_channels,
                    out_channels,
                    kernel_size,
                    stride,
                    padding,
                    bias = True,
                    eta = 0.2).cuda()
    adder.adder.data.copy_(weight)
    adder.b.data.copy_(bias)

    # outref = F.conv2d(input, weight, bias, padding=padding)
    # out_ref = adder2d_function(input, weight, stride, padding)
    out_ref = adder_ref(input)
    out_ref.backward(grad_output, retain_graph=True)
    grad_clone = adder_ref.adder.grad.clone()
    adder_ref.adder.grad.zero_()

    # grad_weight = weight.clone()
    # adder_cuda.backward_weight(grad_output,
    #                            input,
    #                            weight,
    #                            grad_weight,
    #                            kernel_size, kernel_size,
    #                            stride, stride,
    #                            padding, padding)
    output = adder(input)
    output.backward(grad_output, retain_graph=True)
    grad_weight = adder.adder.grad.clone()
    adder.adder.grad.zero_()

    # print("input")
    # print(input)
    # print("weight")
    # print(weight)
    # print("output ref")
    # print(out_ref)
    # # print("output our", output)
    # print("grad output")
    # print(grad_output)
    # print("grad_weight ref")
    # print(grad_clone)
    # print("grad_weight our")
    # print(grad_weight)

    eps = 1e-6
    print(((grad_clone - grad_weight) / (grad_clone.abs() + eps)).abs().max())

    import time
    time_b = time.time()
    # adder_cuda.backward_weight(grad_output,
    #                            input,
    #                            weight,
    #                            grad_weight,
    #                            kernel_size, kernel_size,
    #                            stride, stride,
    #                            padding, padding)
    output.backward(grad_output, retain_graph=True)
    time_e = time.time()
    print("hand_written_conv: {:.4f}us".format((time_e - time_b) * 1e6))

    # outref = F.conv2d(input, weight, bias, padding=padding)
    # out_ref = adder2d_function(input, weight, stride, padding)
    time_b = time.time()
    out_ref.backward(grad_output, retain_graph=True)
    time_e = time.time()
    print("builtin_conv: {:.4f}us".format((time_e - time_b) * 1e6))

    time_b = time.time()
    # adder_cuda.backward_weight(grad_output,
    #                            input,
    #                            weight,
    #                            grad_weight,
    #                            kernel_size, kernel_size,
    #                            stride, stride,
    #                            padding, padding)
    output.backward(grad_output, retain_graph=True)
    time_e = time.time()
    print("hand_written_conv: {:.4f}us".format((time_e - time_b) * 1e6))

    # outref = F.conv2d(input, weight, bias, padding=padding)
    # out_ref = adder2d_function(input, weight, stride, padding)
    time_b = time.time()
    out_ref.backward(grad_output, retain_graph=True)
    time_e = time.time()
    print("builtin_conv: {:.4f}us".format((time_e - time_b) * 1e6))

    time_b = time.time()
    # adder_cuda.backward_weight(grad_output,
    #                            input,
    #                            weight,
    #                            grad_weight,
    #                            kernel_size, kernel_size,
    #                            stride, stride,
    #                            padding, padding)
    output.backward(grad_output, retain_graph=True)
    time_e = time.time()
    print("hand_written_conv: {:.4f}us".format((time_e - time_b) * 1e6))

    # outref = F.conv2d(input, weight, bias, padding=padding)
    # out_ref = adder2d_function(input, weight, stride, padding)
    time_b = time.time()
    out_ref.backward(grad_output, retain_graph=True)
    time_e = time.time()
    print("builtin_conv: {:.4f}us".format((time_e - time_b) * 1e6))


def check_naive_clone():
    batch_size = 1
    in_channels = 1
    out_channels = 1
    in_size = 3
    kernel_size = 1
    padding = 0
    stride = 1
    out_size = (in_size + 2 * padding - kernel_size) // stride + 1
    print(out_size)

    input = tt.randn(batch_size, in_channels, in_size, in_size).cuda()
    weight = tt.randn(out_channels, in_channels, kernel_size, kernel_size).cuda()
    # bias = tt.randn(out_channels).cuda()
    output = tt.randn(batch_size, out_channels, out_size, out_size).cuda()

    result = adder_cuda.forward(input,
                               weight,
                               # bias,
                               output,
                               kernel_size, kernel_size,
                               stride, stride,
                               padding, padding)
    print(result)
    input.clone()
    weight.clone()
    # bias.clone()
    output.clone()

    # F.conv2d(input, weight, bias, padding=padding)

    # input.clone()


if __name__ == '__main__':
    check_forward()
    check_grad_in()
    check_grad_weight()
    # check_naive_clone()
