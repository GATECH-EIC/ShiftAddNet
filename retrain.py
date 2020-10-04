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
import models
import optim
import torch.backends.cudnn as cudnn
from torch_poly_lr_decay import PolynomialLRDecay

from deepshift.convert import convert_to_shift, round_shift_weights, count_layer_type

import matplotlib.pyplot as plt
from adder import adder as adder_fast
import deepshift

import collections
from collections import OrderedDict

# Training settings
parser = argparse.ArgumentParser(description='PyTorch AdderNet Trainning')
parser.add_argument('--dataset', type=str, default='cifar10', help='training dataset')
parser.add_argument('--data_path', type=str, default=None, help='path to dataset')
parser.add_argument('--batch_size', type=int, default=256, metavar='N', help='batch size for training')
parser.add_argument('--test_batch_size', type=int, default=256, metavar='N', help='batch size for testing')
parser.add_argument('--epochs', type=int, default=160, metavar='N', help='number of epochs to train')
parser.add_argument('--start_epoch', type=int, default=0, metavar='N', help='restart point')
parser.add_argument('--schedule', type=int, nargs='+', default=[80, 120], help='learning rate schedule')
parser.add_argument('--lr', type=float, default=0.1, metavar='LR', help='learning rate')
parser.add_argument('--lr-sign', default=None, type=float, help='separate initial learning rate for sign params')
parser.add_argument('--lr_decay', default='stepwise', type=str, choices=['stepwise', 'poly'])
parser.add_argument('--optimizer', type=str, default='sgd', help='used optimizer')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M', help='SGD momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float, metavar='W', help='weight decay')
parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to latest checkpoint')
parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed')
parser.add_argument('--save', default='./logs', type=str, metavar='PATH', help='path to save prune model')
parser.add_argument('--save_suffix', default='retrain_', type=str, help='identify which retrain')
parser.add_argument('--arch', default='resnet20', type=str, help='architecture to use')
parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')
parser.add_argument('--log_interval', type=int, default=100, metavar='N', help='how many batches to wait before logging training status')
# multi-gpus
parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
# shift hyper-parameters
parser.add_argument('--shift_depth', type=int, default=0, help='how many layers to convert to shift')
parser.add_argument('--shift_type', type=str, choices=['Q', 'PS'], help='shift type for representing weights')
parser.add_argument('--rounding', default='deterministic', choices=['deterministic', 'stochastic'])
parser.add_argument('--weight_bits', type=int, default=5, help='number of bits to represent the shift weights')
parser.add_argument('--use-kernel', type=lambda x:bool(distutils.util.strtobool(x)), default=False, help='whether using custom shift kernel')
# add hyper-parameters
parser.add_argument('--add_quant', type=bool, default=False, help='whether to quantize adder layer')
parser.add_argument('--add_bits', type=int, default=8, help='number of bits to represent the adder filters')
# visualization
parser.add_argument('--visualize', action="store_true", default=False, help='if use visualization')
# reinit
parser.add_argument('--reinit', type=str, default=None, help='whether reinit or finetune')

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
# if len(args.gpu_ids) > 0:
#    torch.cuda.set_device(args.gpu_ids[0])

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
    num_cls = 1000
elif args.dataset == 'cifar10':
    num_cls = 10
elif args.dataset == 'cifar100':
    num_cls = 100
elif args.dataset == 'mnist':
    num_cls = 10
else:
    raise NotImplementedError('No such dataset!')

model = models.__dict__[args.arch](num_classes=num_cls, quantize=args.add_quant, weight_bits=args.add_bits)
if args.reinit is not None:
    model_ref = models.__dict__[args.arch](num_classes=num_cls, quantize=args.add_quant, weight_bits=args.add_bits)

if 'shift' in args.arch: # no pretrain
    model, _ = convert_to_shift(model, args.shift_depth, args.shift_type, convert_weights=False, use_kernel=args.use_kernel, rounding=args.rounding, weight_bits=args.weight_bits)
    if args.reinit is not None:
        model_ref, _ = convert_to_shift(model_ref, args.shift_depth, args.shift_type, convert_weights=False, use_kernel=args.use_kernel, rounding=args.rounding, weight_bits=args.weight_bits)


# create optimizer
model_other_params = []
model_sign_params = []
model_shift_params = []

for name, param in model.named_parameters():
    if(name.endswith(".sign")):
        model_sign_params.append(param)
    elif(name.endswith(".shift")):
        model_shift_params.append(param)
    else:
        model_other_params.append(param)

params_dict = [
    {"params": model_other_params},
    {"params": model_sign_params, 'lr': args.lr_sign if args.lr_sign is not None else args.lr, 'weight_decay': 0},
    {"params": model_shift_params, 'lr': args.lr, 'weight_decay': 0}
    ]

# optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
optimizer = None
if (args.optimizer.lower() == "sgd"):
    optimizer = torch.optim.SGD(params_dict, args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
elif (args.optimizer.lower() == "adadelta"):
    optimizer = torch.optim.Adadelta(params_dict, args.lr, weight_decay=args.weight_decay)
elif (args.optimizer.lower() == "adagrad"):
    optimizer = torch.optim.Adagrad(params_dict, args.lr, weight_decay=args.weight_decay)
elif (args.optimizer.lower() == "adam"):
    optimizer = torch.optim.Adam(params_dict, args.lr, weight_decay=args.weight_decay)
elif (args.optimizer.lower() == "rmsprop"):
    optimizer = torch.optim.RMSprop(params_dict, args.lr, weight_decay=args.weight_decay)
elif (args.optimizer.lower() == "radam"):
    optimizer = optim.RAdam(params_dict, args.lr, weight_decay=args.weight_decay)
elif (args.optimizer.lower() == "ranger"):
    optimizer = optim.Ranger(params_dict, args.lr, weight_decay=args.weight_decay)
else:
    raise ValueError("Optimizer type: ", args.optimizer, " is not supported or known")

scheduler_poly_lr_decay = PolynomialLRDecay(optimizer, max_decay_steps=args.epochs, end_learning_rate=0.0001, power=0.9)

def save_checkpoint(state, is_best, epoch, filepath):
    if epoch == 'init':
        filepath = os.path.join(filepath, 'init.pth.tar')
        torch.save(state, filepath)
    else:
        # filename = os.path.join(filepath, 'ckpt'+str(epoch)+'.pth.tar')
        # torch.save(state, filename)
        filename = os.path.join(filepath, 'ckpt.pth.tar')
        torch.save(state, filename)
        if is_best:
            shutil.copyfile(filename, os.path.join(filepath, 'model_best.pth.tar'))

if args.resume:
    if os.path.isfile(args.resume):
        print("=> loading checkpoint '{}'".format(args.resume))
        checkpoint = torch.load(args.resume, map_location='cpu')
        # args.start_epoch = checkpoint['epoch']
        # best_prec1 = checkpoint['best_prec1']
        model.load_state_dict(checkpoint['state_dict'])
        # optimizer.load_state_dict(checkpoint['optimizer'])
        print("=> loaded checkpoint '{}' (epoch {})"
              .format(args.resume, checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(args.resume))
else:
    save_checkpoint({'state_dict': model.state_dict()}, False, epoch='init', filepath=args.save)

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


if args.cuda:
    model.cuda()
    if args.reinit is not None:
        model_ref.cuda()
if len(args.gpu_ids) > 1:
    model = torch.nn.DataParallel(model, device_ids=args.gpu_ids)

if args.reinit is not None:
    print('use reinit!')
    # checkpoint = torch.load(args.reinit)
    # checkpoint['state_dict'] = change_name(checkpoint['state_dict'])
    # model_ref.load_state_dict(checkpoint['state_dict'])

    for m, m_ref in zip(model.modules(), model_ref.modules()):
        if isinstance(m, deepshift.modules_q.Conv2dShiftQ): # Q shift
            weight_copy = m.weight.data.abs().clone()
            # mask = weight_copy.gt(0).float().cuda()
            mask = torch.ne(weight_copy, 1.).float().cuda()
            m_ref.weight.data = m_ref.weight.data.mul_(mask) + 1 - mask # no shift
        elif isinstance(m, deepshift.modules.Conv2dShift): # PS shift
            weight_copy = m.shift.data.abs().clone()
            mask = weight_copy.gt(0).float().cuda()
            m_ref.shift.data.mul_(mask)
            m_ref.sign.data.mul_(mask)
        elif isinstance(m, adder_fast.Adder2D):
            adder_copy = m.adder.data.abs().clone()
            mask = adder_copy.gt(0).float().cuda()
            m_ref.adder.data.mul_(mask)
    model = model_ref


# print model summary
model_summary = None
try:
    model_summary, model_params_info = torchsummary.summary_string(model, input_size=(3,32,32))
    print(model_summary)
    print("WARNING: The summary function reports duplicate parameters for multi-GPU case")
except:
    print("WARNING: Unable to obtain summary of model")

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
if args.visualize:
    args.save = os.path.join('./visualization', args.save[2:])
args.save = os.path.join(args.save, args.save_suffix)
print(args.save)
if not os.path.exists(args.save):
    os.makedirs(args.save)

history_score = np.zeros((args.epochs, 3))

def visualize(feat, labels, epoch):
    plt.ion()
    c = ['#ff0000', '#ffff00', '#00ff00', '#00ffff', '#0000ff',
         '#ff00ff', '#990000', '#999900', '#009900', '#009999']
    plt.clf()
    for i in range(10):
        plt.plot(feat[labels == i, 0], feat[labels == i, 1], '.', c=c[i])
    plt.legend(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'], loc = 'upper right')
    # plt.xlim(xmin=-8,xmax=8)
    # plt.ylim(ymin=-8,ymax=8)
    # plt.text(-7.8,7.3,"epoch=%d" % epoch)
    plt.title("epoch=%d" % epoch)
    vis_dir = os.path.join(args.save, 'visualization')
    if not os.path.exists(vis_dir):
        os.makedirs(vis_dir)
    plt.savefig(vis_dir+'/epoch=%d.jpg' % epoch)
    plt.draw()
    plt.pause(0.001)

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

def train(epoch):
    model.train()
    global history_score
    avg_loss = 0.
    train_acc = 0.
    end_time = time.time()
    feat_loader = []
    idx_loader = []
    for batch_idx, (data, target) in enumerate(train_loader):
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        if args.visualize:
            ### store for visualization ###
            feat, output = model(data)
            feat_loader.append(feat)
            idx_loader.append(target)
            ###############################
        else:
            output = model(data)
        loss = F.cross_entropy(output, target)
        avg_loss += loss.item()
        prec1, prec5 = accuracy(output.data, target.data, topk=(1, 5))
        train_acc += prec1.item()
        loss.backward()
        #-----------------------------------------
        for k, m in enumerate(model.modules()):
            # print(k, m)
            if isinstance(m, deepshift.modules_q.Conv2dShiftQ): # Q shift
                weight_copy = m.weight.data.abs().clone()
                mask = torch.ne(weight_copy, 1.).float().cuda()
                # print(mask)
                m.weight.grad.data.mul_(mask)
            elif isinstance(m, deepshift.modules.Conv2dShift): # PS shift
                weight_copy = m.shift.data.abs().clone()
                mask = weight_copy.gt(0).float().cuda()
                m.shift.grad.data.mul_(mask)
                m.sign.grad.data.mul_(mask)
            elif isinstance(m, adder_fast.Adder2D):
                adder_copy = m.adder.data.abs().clone()
                mask = adder_copy.gt(0).float().cuda()
                m.adder.grad.data.mul_(mask)
        #-----------------------------------------
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.1f}%)]\tLoss: {:.6f}'.format(epoch, batch_idx * len(data), len(train_loader.dataset), 100. * batch_idx / len(train_loader), loss.item()))
    history_score[epoch][0] = avg_loss / len(train_loader)
    history_score[epoch][1] = np.round(train_acc / len(train_loader), 2)
    if args.visualize:
        feat = torch.cat(feat_loader, 0)
        labels = torch.cat(idx_loader, 0)
        visualize(feat.data.cpu().numpy(), labels.data.cpu().numpy(), epoch)

def test():
    model.eval()
    test_loss = 0
    test_acc = 0
    for data, target in test_loader:
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        if args.visualize:
            _, output = model(data)
        else:
            output = model(data)
        test_loss += F.cross_entropy(output, target, size_average=False).item() # sum up batch loss
        prec1, prec5 = accuracy(output.data, target.data, topk=(1, 5))
        test_acc += prec1.item()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, test_acc, len(test_loader), test_acc / len(test_loader)))
    return np.round(test_acc / len(test_loader), 2)


best_prec1 = 0.
for epoch in range(args.start_epoch, args.epochs):
    if args.lr_decay == 'stepwise':
        # step-wise LR schedule
        if epoch in args.schedule:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.1
    elif args.lr_decay == 'poly':
        # Poly LR schedule
        scheduler_poly_lr_decay.step(epoch)
    else:
        raise NotImplementedError
    train(epoch)
    prec1 = test()
    history_score[epoch][2] = prec1
    np.savetxt(os.path.join(args.save, 'record.txt'), history_score, fmt='%10.5f', delimiter=',')
    is_best = prec1 > best_prec1
    best_prec1 = max(prec1, best_prec1)
    model_rounded = round_shift_weights(model, clone=True)
    save_checkpoint({
        'epoch': epoch + 1,
        'state_dict': model_rounded.state_dict(),
        'best_prec1': best_prec1,
        'optimizer': optimizer.state_dict(),
    }, is_best, epoch, filepath=args.save)

print("Best accuracy: " + str(best_prec1))
history_score[-1][0] = best_prec1
np.savetxt(os.path.join(args.save, 'record.txt'), history_score, fmt='%10.5f', delimiter=',')