'''
 # @ Author: Xiaohan Chen
 # @ Email: chernxh@tamu.edu
 # @ Create Time: 2019-07-11 01:35:15
 # @ Modified by: Xiaohan Chen
 # @ Modified time: 2019-07-11 01:36:19
 # @ Description:
 '''

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

from se_shift.utils_quantize import sparsify_and_nearestpow2
from se_shift.alg import VEC_2_SHAPE


class SELinear(nn.Module):
    """SmartExchange Linear Layer.

    Arguments:
        nn {[type]} -- [description]
    """
    def __init__(self, in_features, out_features, bias=True, threshold=5e-3):
        super(SELinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_splits, self.size_splits = VEC_2_SHAPE[self.in_features]
        self.size_B = 3

        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)

        self.threshold = threshold
        self.C = nn.Parameter(torch.Tensor(self.out_features * self.num_splits,
                                           self.size_splits, self.size_B)).float()
        self.B = nn.Parameter(torch.Tensor(self.C.size()[0],
                                           self.size_B, self.size_B)).float()
        self.register_buffer('mask', torch.Tensor(*self.C.size()).float())
        self.register_buffer('weight', torch.Tensor(out_features, in_features).float())
        self.set_mask()

        self.reset_parameters()

    def reset_dC_counter(self):
        self.dC_counter = self.C.new_zeros(self.C.size()).float()

    def reset_parameters(self):
        init.kaiming_uniform_(self.C, a=math.sqrt(5))
        with torch.no_grad():
            self.B.normal_(0, 1.0 / math.sqrt(3.0))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def set_mask(self):
        self.C.data = self.sparsify_and_quantize_C(mask=False)
        self.mask.data = (self.C != 0.0).float()
        assert self.mask.requires_grad == False

    def sparsify_and_quantize_C(self, mask=True):
        qC = sparsify_and_nearestpow2(self.C, self.threshold)
        if mask:
            qC = qC * self.mask
        return qC

    def get_weight(self, mask=True):
        qC = self.sparsify_and_quantize_C(mask=mask)
        # if mask:
        #     qC = qC * self.mask
        weight = torch.bmm(qC, self.B)
        # Reshape `self.BC` to (out_features, in_features)
        return weight.reshape(self.out_features, -1)[:,0:self.in_features]

    def forward(self, input):
        weight = self.get_weight(mask=True)
        return F.linear(input, weight, self.bias)

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )
