'''
refer to AdderNet code
supply `new_cdist` for trianing purpose

'''
import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Function
import math

def new_cdist(p, eta):
    class cdist(torch.autograd.Function):
        @staticmethod
        def forward(ctx, W, X):
            ctx.save_for_backward(W, X)
            out = -torch.cdist(W, X, p)
            return out

        @staticmethod
        def backward(ctx, grad_output):
            W, X = ctx.saved_tensors
            grad_W = grad_X = None
            if ctx.needs_input_grad[0]:
                # print('grad dim:', grad_output.shape)
                # print('X dim:', X.shape)
                # print('W dim:', W.shape)
                # back propogation
                X_unsqueeze = torch.unsqueeze(X, 0).expand(W.shape[0], X.shape[0], X.shape[1])
                W_unsqueeze = torch.unsqueeze(W, 1).expand(W.shape[0], X.shape[0], W.shape[1])
                grad_unsqueeze = torch.unsqueeze(grad_output, 2).expand(grad_output.shape[0], grad_output.shape[1], W.shape[1])
                grad_W = ((X_unsqueeze - W_unsqueeze) * grad_unsqueeze).sum(1)
                grad_W = eta * np.sqrt(grad_W.numel()) / torch.norm(grad_W) * grad_W
            if ctx.needs_input_grad[1]:
                grad_X = (torch.nn.functional.hardtanh((W_unsqueeze - X_unsqueeze), min_val=-1., max_val=1.) * grad_unsqueeze).sum(0)
                # print("hardtanh input", W_unsqueeze - X_unsqueeze)
                # print("hardtanh", torch.nn.functional.hardtanh((W_unsqueeze - X_unsqueeze), min_val=-1., max_val=1.))
            return grad_W, grad_X
    return cdist().apply

eta = 0.2
cdist = new_cdist(1, eta)

def adder2d_function(X, W, stride=1, padding=0):
    n_filters, d_filter, h_filter, w_filter = W.size()
    n_x, d_x, h_x, w_x = X.size()

    h_out = (h_x - h_filter + 2 * padding) / stride + 1
    w_out = (w_x - w_filter + 2 * padding) / stride + 1

    h_out, w_out = int(h_out), int(w_out)
    # print('W dim:', W.shape)
    # print('X dim:', X.shape)
    X_col = torch.nn.functional.unfold(X.view(1, -1, h_x, w_x), h_filter, dilation=1, padding=padding, stride=stride).view(n_x, -1, h_out*w_out)
    # print('X_fold dim:', X_col.shape)
    X_col = X_col.permute(1,2,0).contiguous().view(X_col.size(1),-1)
    W_col = W.view(n_filters, -1)

    # print('W dim:', W_col.shape)
    # print('X dim:', X_col.shape)
    # exit()
    # out = -torch.cdist(W_col,X_col.transpose(0,1),1)
    out = cdist(W_col, X_col.transpose(0,1))
    # print('out dim:', out.shape)

    out = out.view(n_filters, h_out, w_out, n_x)
    out = out.permute(3, 0, 1, 2).contiguous()

    return out


class adder2d(nn.Module):

    def __init__(self,input_channel,output_channel,kernel_size, stride=1, padding=0, bias = False):
        super(adder2d, self).__init__()
        self.stride = stride
        self.padding = padding
        self.input_channel = input_channel
        self.output_channel = output_channel
        self.kernel_size = kernel_size
        self.adder = torch.nn.Parameter(nn.init.normal_(torch.randn(output_channel,input_channel,kernel_size,kernel_size)))
        self.bias = bias
        if bias:
            self.b = torch.nn.Parameter(nn.init.uniform_(torch.zeros(output_channel)))

    def forward(self, x):
        output = adder2d_function(x,self.adder, self.stride, self.padding)
        if self.bias:
            output += self.b.unsqueeze(0).unsqueeze(2).unsqueeze(3)

        return output
