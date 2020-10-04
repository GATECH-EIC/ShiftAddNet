from adder import adder
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['CNN_add']

def conv_add(in_planes, out_planes, kernel_size=(3, 3), stride=1, padding=0, quantize=False, weight_bits=8, sparsity=0):
    " 3x3 convolution with padding "
    return adder.Adder2D(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, bias=False,
                         quantize=quantize, weight_bits=weight_bits, sparsity=sparsity)


class CNN(nn.Module):
    def __init__(self, num_classes, quantize=False, weight_bits=8, sparsity=0):
        super(CNN, self).__init__()
        self.quantize = quantize
        self.weight_bits = weight_bits
        self.sparsity = sparsity

        self.conv1 = conv_add(1, 5, kernel_size=(12, 5), quantize=self.quantize, weight_bits=self.weight_bits, sparsity=self.sparsity)
        self.bn1 = nn.BatchNorm2d(5)
        self.conv2 = conv_add(5, 10, kernel_size=(5, 1), quantize=self.quantize, weight_bits=self.weight_bits, sparsity=self.sparsity)
        self.bn2 = nn.BatchNorm2d(10)
        # self.conv3 = nn.Conv2d(36, 24, kernel_size=(12, 1))
        self.pool1 = nn.MaxPool2d((4,4))
        self.pool2 = nn.MaxPool2d((2,2))
        self.fc1 = conv_add(590, num_classes, kernel_size=(1,1), quantize=self.quantize, weight_bits=self.weight_bits, sparsity=self.sparsity)
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

def CNN_add(num_classes=10, quantize=False, weight_bits=8, sparsity=0, quantize_v='sbm', **kwargs):
    return CNN(num_classes, quantize=quantize, weight_bits=weight_bits, sparsity=sparsity)