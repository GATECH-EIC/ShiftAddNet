import argparse
import os, time
import torch
import shutil
import numpy as np
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import  models
import torch.backends.cudnn as cudnn

import deepshift
from deepshift.convert import convert_to_shift, round_shift_weights, count_layer_type
from models import adder as adder_slow
from adder import adder as adder_fast

import collections
from collections import OrderedDict

# Training settings
parser = argparse.ArgumentParser(description='PyTorch Pruning')
parser.add_argument('--dataset', type=str, default='cifar10', help='training dataset')
parser.add_argument('--data_path', type=str, default=None, help='path to dataset')
parser.add_argument('--batch_size', type=int, default=256, metavar='N', help='batch size for training')
parser.add_argument('--test_batch_size', type=int, default=256, metavar='N', help='batch size for testing')
parser.add_argument('--epochs', type=int, default=160, metavar='N', help='number of epochs to train')
parser.add_argument('--start_epoch', type=int, default=0, metavar='N', help='restart point')
parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to latest checkpoint')
parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed')
parser.add_argument('--save', default='./logs', type=str, metavar='PATH', help='path to save prune model')
parser.add_argument('--arch', default='resnet20', type=str, help='architecture to use')
parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')
# multi-gpus
parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
# shift hyper-parameters
parser.add_argument('--shift_depth', type=int, default=0, help='how many layers to convert to shift')
parser.add_argument('--shift_type', type=str, choices=['Q', 'PS'], help='shift type for representing weights')
parser.add_argument('--rounding', default='deterministic', choices=['deterministic', 'stochastic'])
parser.add_argument('--weight_bits', type=int, default=5, help='number of bits to represent the shift weights')
parser.add_argument('--use-kernel', type=lambda x:bool(distutils.util.strtobool(x)), default=False, help='whether using custom shift kernel')
# pruning ratio
parser.add_argument('--percent', default=0.6, type=float, help='percentage of weight to prune')
parser.add_argument('--prune_method', default='magnitude', choices=['random', 'magnitude'])
parser.add_argument('--prune_layer', default='all', choices=['shift', 'add', 'all'])

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

if not os.path.exists(args.save):
    os.makedirs(args.save)

cudnn.benchmark = True

gpu = args.gpu_ids
gpu_ids = args.gpu_ids.split(',')
args.gpu_ids = []
for gpu_id in gpu_ids:
    id = int(gpu_id)
    args.gpu_ids.append(id)
print(args.gpu_ids)
if len(args.gpu_ids) > 0:
   torch.cuda.set_device(args.gpu_ids[0])

kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
if args.dataset == 'cifar10':
    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('./data.cifar10', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.Pad(4),
                           transforms.RandomCrop(32),
                           transforms.RandomHorizontalFlip(),
                           transforms.ToTensor(),
                           transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                       ])),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('./data.cifar10', train=False, transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                       ])),
        batch_size=args.test_batch_size, shuffle=True, **kwargs)
elif args.dataset == 'cifar100':
    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR100('./data.cifar100', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.Pad(4),
                           transforms.RandomCrop(32),
                           transforms.RandomHorizontalFlip(),
                           transforms.ToTensor(),
                           transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                       ])),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR100('./data.cifar100', train=False, transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                       ])),
        batch_size=args.test_batch_size, shuffle=True, **kwargs)
elif args.dataset == 'mnist':
    trainset = datasets.MNIST('../MNIST', download=True, train=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))]
        )
    )
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    testset = datasets.MNIST('../MNIST', download=True, train=False,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))]
        )
    )
    test_loader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size, shuffle=True, num_workers=4)
else:
    # Data loading code
    traindir = os.path.join(args.data, 'train')
    valdir = os.path.join(args.data, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=16, pin_memory=True)

    test_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.test_batch_size, shuffle=False,
        num_workers=16, pin_memory=True)

if args.dataset == 'imagenet':
    model = models.__dict__[args.arch](num_classes=1000)
elif args.dataset == 'cifar10':
    model = models.__dict__[args.arch](num_classes=10)
elif args.dataset == 'cifar100':
    model = models.__dict__[args.arch](num_classes=100)
elif args.dataset == 'mnist':
    model = models.__dict__[args.arch](num_classes=10)
else:
    raise NotImplementedError('No such dataset!')

if 'shift' in args.arch: # no pretrain
    model, _ = convert_to_shift(model, args.shift_depth, args.shift_type, convert_weights=False, use_kernel=args.use_kernel, rounding=args.rounding, weight_bits=args.weight_bits)

if args.cuda:
    model.cuda()
if len(args.gpu_ids) > 1:
    model = torch.nn.DataParallel(model, device_ids=args.gpu_ids)

# save name
# name model sub-directory "shift_all" if all layers are converted to shift layers
conv2d_layers_count = count_layer_type(model, nn.Conv2d) #+ count_layer_type(model, unoptimized.UnoptimizedConv2d)
linear_layers_count = count_layer_type(model, nn.Linear) #+ count_layer_type(model, unoptimized.UnoptimizedLinear)
print(conv2d_layers_count)

if (args.shift_depth > 0):
    if (args.shift_type == 'Q'):
        shift_label = "shift_q"
    else:
        shift_label = "shift_ps"
else:
    shift_label = "shift"

# if (conv2d_layers_count==0 and linear_layers_count==0):
if conv2d_layers_count == 0:
    shift_label += "_all"
else:
    shift_label += "_%s" % (args.shift_depth)

if (args.shift_depth > 0):
    shift_label += "_wb_%s" % (args.weight_bits)

args.save = os.path.join(args.save, shift_label)
args.save = os.path.join(args.save, 'prune_'+str(args.prune_method)+'_'+str(args.prune_layer)+'_'+str(args.percent))
if not os.path.exists(args.save):
    os.makedirs(args.save)

def save_checkpoint(state, is_best, epoch, filepath):
    filename = os.path.join(filepath, 'pruned.pth.tar')
    torch.save(state, filename)

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def test(model):
    model.eval()
    test_loss = 0
    test_acc = 0
    for data, target in test_loader:
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
        test_loss += F.cross_entropy(output, target, size_average=False).item() # sum up batch loss
        prec1, prec5 = accuracy(output.data, target.data, topk=(1, 5))
        test_acc += prec1.item()

    test_loss /= len(test_loader.dataset)
    return test_loss, np.round(test_acc / len(test_loader), 2)

def change_name(state_dict):
    new_state_dict = OrderedDict()
    for key, value in state_dict.items():
        if 'conv' in key and '.1.weight' in key:
            new_key = key.replace('weight', 'adder')
        elif 'downsample' in key and '.1.weight' in key:
            new_key = key.replace('weight', 'adder')
        else:
            new_key = key
        new_state_dict[new_key] = value
    return new_state_dict

if args.resume:
    if os.path.isfile(args.resume):
        print("=> loading checkpoint '{}'".format(args.resume))
        checkpoint = torch.load(args.resume)
        # args.start_epoch = checkpoint['epoch']
        best_prec1 = checkpoint['best_prec1']
        if 'add' in args.arch:
            checkpoint['state_dict'] = change_name(checkpoint['state_dict'])
        model.load_state_dict(checkpoint['state_dict'])
        print("=> loaded checkpoint '{}' (epoch {}) Prec1: {:f}"
              .format(args.resume, checkpoint['epoch'], best_prec1))
    else:
        print("=> no checkpoint found at '{}'".format(args.resume))
else:
    save_checkpoint({'state_dict': model.state_dict()}, False, epoch='init', filepath=args.save)

# round weights to ensure that the results are due to powers of 2 weights
model = round_shift_weights(model)
print('\nEvaluation only')
test_loss0, test_acc0 = test(model)
print('Before pruning: Test Loss:  %.8f, Test Acc:  %.2f' % (test_loss0, test_acc0))


def create_mask(shape, rate):
    mask = torch.cuda.FloatTensor(shape).uniform_() > rate
    return mask + 0

# -------------------------------------------------------------
if 'shift' in args.arch and args.prune_layer != 'add':
    print('prune for shift layer:')
    if args.shift_type == 'Q':
        shift_module = deepshift.modules_q.Conv2dShiftQ
    elif args.shift_type == 'PS':
        shift_module = deepshift.modules.Conv2dShift
    else:
        raise NotImplementedError
    # pruning
    if args.shift_type == 'Q':
        total = 0
        for m in model.modules():
            if isinstance(m, shift_module):
                total += m.weight.data.numel()
        shift_weights = torch.zeros(total)
        index = 0
        for m in model.modules():
            if isinstance(m, shift_module):
                size = m.weight.data.numel()
                shift_weights[index:(index+size)] = m.weight.data.view(-1).abs().clone()
                index += size

        y, i = torch.sort(shift_weights)
        thre_index = int(total * args.percent)
        thre = y[thre_index] - 1e-7
        pruned = 0
        print('Pruning threshold: {}'.format(thre))
        zero_flag = False
        # ----------------------------------------------------------------
        if args.prune_method == 'magnitude':
            for k, m in enumerate(model.modules()):
                if isinstance(m, shift_module):
                    shift_copy = m.weight.data.abs().clone()
                    # prune at boundary (weight == thre)
                    _mask = torch.eq(shift_copy, thre+1e-7).float().cuda()
                    _mask = _mask * torch.cuda.FloatTensor(shift_copy.shape).uniform_(-args.percent, 1-args.percent)
                    shift_copy += _mask
                    # ---------------------------------
                    mask = shift_copy.gt(thre).float().cuda()
                    pruned = pruned + mask.numel() - torch.sum(mask)
                    m.weight.data = m.weight.data.mul_(mask) + 1 - mask # no shift
                    if int(torch.sum(mask)) == 0:
                        zero_flag = True
                    print('layer index: {:d} \t total params: {:d} \t remaining params: {:d}'.
                        format(k, mask.numel(), int(torch.sum(mask))))
        elif args.prune_method == 'random':
            for k, m in enumerate(model.modules()):
                if isinstance(m, shift_module):
                    shift_copy = m.weight.data.abs().clone()
                    mask = create_mask(shift_copy.shape, args.percent)
                    pruned = pruned + mask.numel() - torch.sum(mask)
                    m.weight.data = m.weight.data.mul_(mask) + 1 - mask # no shift
                    if int(torch.sum(mask)) == 0:
                        zero_flag = True
                    print('layer index: {:d} \t total params: {:d} \t remaining params: {:d}'.
                        format(k, mask.numel(), int(torch.sum(mask))))
        else:
            raise NotImplementedError
        # ----------------------------------------------------------------
    elif args.shift_type == 'PS':
        total = 0
        for m in model.modules():
            if isinstance(m, shift_module):
                total += m.shift.data.numel()
        shift_weights = torch.zeros(total)
        index = 0
        for m in model.modules():
            if isinstance(m, shift_module):
                size = m.shift.data.numel()
                shift_weights[index:(index+size)] = m.shift.data.view(-1).abs().clone()
                index += size

        y, i = torch.sort(shift_weights)
        thre_index = int(total * args.percent)
        thre = y[thre_index] - 1e-7
        pruned = 0
        print('Pruning threshold: {}'.format(thre))
        zero_flag = False
        # ----------------------------------------------------------------
        if args.prune_method == 'magnitude':
            for k, m in enumerate(model.modules()):
                if isinstance(m, shift_module):
                    shift_copy = m.shift.data.abs().clone()
                    mask = shift_copy.gt(thre).float().cuda()
                    pruned = pruned + mask.numel() - torch.sum(mask)
                    m.shift.data.mul_(mask)
                    m.sign.data.mul_(mask)
                    if int(torch.sum(mask)) == 0:
                        zero_flag = True
                    print('layer index: {:d} \t total params: {:d} \t remaining params: {:d}'.
                        format(k, mask.numel(), int(torch.sum(mask))))
        elif args.prune_method == 'random':
            for k, m in enumerate(model.modules()):
                if isinstance(m, shift_module):
                    shift_copy = m.shift.data.abs().clone()
                    mask = create_mask(shift_copy.shape, args.percent)
                    pruned = pruned + mask.numel() - torch.sum(mask)
                    m.shift.data.mul_(mask)
                    m.sign.data.mul_(mask)
                    if int(torch.sum(mask)) == 0:
                        zero_flag = True
                    print('layer index: {:d} \t total params: {:d} \t remaining params: {:d}'.
                        format(k, mask.numel(), int(torch.sum(mask))))
        else:
            raise NotImplementedError
        # ----------------------------------------------------------------
    print('Total conv params: {}, Pruned conv params: {}, Pruned ratio: {}'.format(total, pruned, float(pruned)/total))
if 'add' in args.arch and args.prune_layer != 'shift':
    print('prune for adder layer:')
    adder_module = adder_slow.adder2d
    adder_module = adder_fast.Adder2D
    total = 0
    for m in model.modules():
        if isinstance(m, adder_module):
            total += m.adder.data.numel()
    adder_weights = torch.zeros(total)
    index = 0
    for m in model.modules():
        if isinstance(m, adder_module):
            size = m.adder.data.numel()
            adder_weights[index:(index+size)] = m.adder.data.view(-1).abs().clone()
            index += size

    y, i = torch.sort(adder_weights)
    thre_index = int(total * args.percent)
    thre = y[thre_index]
    pruned = 0
    print('Pruning threshold: {}'.format(thre))
    zero_flag = False
    # ----------------------------------------------------------------
    if args.prune_method == 'magnitude':
        for k, m in enumerate(model.modules()):
            if isinstance(m, adder_module):
                adder_copy = m.adder.data.abs().clone()
                mask = adder_copy.gt(thre).float().cuda()
                pruned = pruned + mask.numel() - torch.sum(mask)
                m.adder.data.mul_(mask)
                if int(torch.sum(mask)) == 0:
                    zero_flag = True
                print('layer index: {:d} \t total params: {:d} \t remaining params: {:d}'.
                    format(k, mask.numel(), int(torch.sum(mask))))
    elif args.prune_method == 'random':
        for k, m in enumerate(model.modules()):
            if isinstance(m, shift_module):
                shift_copy = m.adder.data.abs().clone()
                mask = create_mask(shift_copy.shape, args.percent)
                pruned = pruned + mask.numel() - torch.sum(mask)
                m.adder.data.mul_(mask)
                if int(torch.sum(mask)) == 0:
                    zero_flag = True
                print('layer index: {:d} \t total params: {:d} \t remaining params: {:d}'.
                    format(k, mask.numel(), int(torch.sum(mask))))
    else:
        raise NotImplementedError
    # ----------------------------------------------------------------
    print('Total conv params: {}, Pruned conv params: {}, Pruned ratio: {}'.format(total, pruned, float(pruned)/total))
# -------------------------------------------------------------

print('\nTesting')
test_loss1, test_acc1 = test(model)
print('After Pruning: Test Loss:  %.8f, Test Acc:  %.2f' % (test_loss1, test_acc1))
save_checkpoint({
        'epoch': 0,
        'state_dict': model.state_dict(),
        'acc': test_acc1,
        'best_acc': 0.,
    }, False, epoch=0, filepath=args.save)

with open(os.path.join(args.save, 'prune.txt'), 'w') as f:
    f.write('Before pruning: Test Loss:  %.8f, Test Acc:  %.2f\n' % (test_loss0, test_acc0))
    f.write('Total conv params: {}, Pruned conv params: {}, Pruned ratio: {}\n'.format(total, pruned, float(pruned)/total))
    f.write('After Pruning: Test Loss:  %.8f, Test Acc:  %.2f\n' % (test_loss1, test_acc1))

    if zero_flag:
        f.write("There exists a layer with 0 parameters left.")