'''
Modified from https://github.com/pytorch/vision.git
Copy from https://github.com/Jerry-2017/DoubleBlindImage/blob/master/code/gaussiansmooth/vgg.py
'''
import math
from adder import adder
# from conv import adder
import torch.nn as nn
import torch.nn.init as init

__all__ = [
    'VGG', 'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn',
    'vgg19_bn', 'vgg19', 'vgg11_nd', 'vgg11_nd_s', 'vgg13_nd', 'vgg13_nd_s', 'vgg16_nd', 'vgg16_nd_s', 'vgg19_nd', 'vgg19_nd_s',
    'vgg11_nd_ss', 'vgg13_nd_ss', 'vgg16_nd_ss', 'vgg19_small_mult_cuda',
]

def conv3x3(in_planes, out_planes, stride=1):
    " 3x3 convolution with padding "
    return adder.Adder2D(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)



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


    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


def make_layers(cfg, batch_norm=True):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            # conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            conv2d = conv3x3(in_channels, v)
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
}


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

def vgg19_small_mult_cuda(num_classes=10, **kwargs):
    """VGG 19-layer model (configuration "E")"""
    return VGG(make_layers(cfg['E']), num_classes=num_classes, dropout=False, small=True)

def vgg19_nd_ss():
    """VGG 19-layer model (configuration "E")"""
    return VGG(make_layers(cfg['E']), dropout=False, small=True, supersmall=True)



def vgg19_bn():
    """VGG 19-layer model (configuration 'E') with batch normalization"""
    return VGG(make_layers(cfg['E'], batch_norm=True))