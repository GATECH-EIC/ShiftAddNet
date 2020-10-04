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
from cyclicLR import CyclicCosAnnealingLR

from deepshift.convert import convert_to_shift, round_shift_weights, count_layer_type

import matplotlib.pyplot as plt
import time

from adder.adder import Adder2D
import deepshift
from visualize import visualizer
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4,5,6,7'

# Training settings
parser = argparse.ArgumentParser(description='PyTorch AdderNet Trainning')
parser.add_argument('--data', type=str, default='/data3/imagenet-data/raw-data/', help='path to imagenet')
parser.add_argument('--dataset', type=str, default='cifar10', help='training dataset')
parser.add_argument('--data_path', type=str, default=None, help='path to dataset')
parser.add_argument('--batch_size', type=int, default=256, metavar='N', help='batch size for training')
parser.add_argument('--test_batch_size', type=int, default=256, metavar='N', help='batch size for testing')
parser.add_argument('--epochs', type=int, default=160, metavar='N', help='number of epochs to train')
parser.add_argument('--start_epoch', type=int, default=0, metavar='N', help='restart point')
parser.add_argument('--schedule', type=int, nargs='+', default=[80, 120], help='learning rate schedule')
parser.add_argument('--lr', type=float, default=0.1, metavar='LR', help='learning rate')
parser.add_argument('--lr-sign', default=None, type=float, help='separate initial learning rate for sign params')
parser.add_argument('--lr_decay', default='stepwise', type=str, choices=['stepwise', 'cosine', 'cyclic_cosine'])
parser.add_argument('--optimizer', type=str, default='sgd', help='used optimizer')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M', help='SGD momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float, metavar='W', help='weight decay')
parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to latest checkpoint')
parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed')
parser.add_argument('--save', default='./logs', type=str, metavar='PATH', help='path to save prune model')
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
parser.add_argument('--sign_threshold_ps', type=float, default=None, help='can be controled')
parser.add_argument('--use-kernel', type=lambda x:bool(distutils.util.strtobool(x)), default=False, help='whether using custom shift kernel')
# add hyper-parameters
parser.add_argument('--add_quant', type=bool, default=False, help='whether to quantize adder layer')
parser.add_argument('--add_bits', type=int, default=8, help='number of bits to represent the adder filters')
parser.add_argument('--quantize_v', type=str, default='sbm', help='quantize version')
# shift hyper-parameters
parser.add_argument('--shift_quant_bits', type=int, default=0, help='quantization training for shift layer')
# visualization
parser.add_argument('--visualize', action="store_true", default=False, help='if use visualization')
# distributed parallel
parser.add_argument("--local_rank", type=int, default=0)
parser.add_argument("--port", type=str, default="15000")
parser.add_argument('--distributed', action='store_true', help='whether to use distributed training')
# tropical visualization
parser.add_argument('--vis_tropical',default=False, type=bool, help='visualize last layer of network')
parser.add_argument('--visualize_dir',default="", type=str, help='visualize last layer of network')
parser.add_argument('--eval_only', action='store_true', help='whether only evaluation')

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

if args.distributed:
    os.environ['MASTER_PORT'] = args.port
    torch.distributed.init_process_group(backend="nccl")

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
    num_classes = 1000
    model = models.__dict__[args.arch](num_classes=1000, quantize=args.add_quant, weight_bits=args.add_bits)
elif args.dataset == 'cifar10':
    num_classes = 10
    model = models.__dict__[args.arch](num_classes=10, quantize=args.add_quant, weight_bits=args.add_bits, quantize_v=args.quantize_v)
elif args.dataset == 'cifar100':
    num_classes = 100
    model = models.__dict__[args.arch](num_classes=100, quantize=args.add_quant, weight_bits=args.add_bits, quantize_v=args.quantize_v)
elif args.dataset == 'mnist':
    model = models.__dict__[args.arch](num_classes=10, quantize=args.add_quant, weight_bits=args.add_bits)
else:
    raise NotImplementedError('No such dataset!')

# print(model)
# M = []
# N = []
# K = []
# S = []
# C = []
# size = 32
# for m in model.modules():
#     if isinstance(m, nn.Conv2d):
#         M.append(m.weight.shape[0])
#         N.append(m.weight.shape[1])
#         K.append(m.weight.shape[2])
#         S.append(m.stride[0])
#         C.append(int(size))
#         if S[-1] == 2:
#             size /= 2
# print('M', M)
# print('N', N)
# print('K', K)
# print('S', S)
# print('C', C)
# print(len(M))
# for i in range(len(M)):
#     print('const int M{} = {}, N{} = {}, K{} = {}, S{} = {}, C{} = {};'.format(
#             i, M[i], i, N[i], i, K[i], i, S[i], i, C[i]))
#     print('const int H{} = C{} - S{} + K{};'.format(i, i, i, i))
# exit()
if 'shift' in args.arch: # no pretrain
    model, _ = convert_to_shift(model, args.shift_depth, args.shift_type, convert_weights=False, use_kernel=args.use_kernel, rounding=args.rounding,
        weight_bits=args.weight_bits, sign_threshold_ps=args.sign_threshold_ps, quant_bits=args.shift_quant_bits)

if args.cuda:
    model.cuda()
if len(args.gpu_ids) > 1:
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=args.gpu_ids)
    else:
        model = torch.nn.DataParallel(model, device_ids=args.gpu_ids)

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

schedule_cosine_lr_decay = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, eta_min=0, last_epoch=-1)
scheduler_cyclic_cosine_lr_decay = CyclicCosAnnealingLR(optimizer, milestones=[40,60,80,100,140,180,200,240,280,300,340,400], decay_milestones=[100, 200, 300, 400], eta_min=0)

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

def load_add_state_dict(state_dict):
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if 'weight' in k and not 'bn' in k and not 'fc' in k:
            if k == 'conv1.weight' or 'downsample.1' in k:
                new_state_dict[k] = v
                continue
            k = k[:-6] + 'adder'
        # print(k)
        new_state_dict[k] = v
    return new_state_dict

def load_shiftadd_state_dict(state_dict):
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if 'weight' in k and not 'bn' in k and not 'fc' in k:
            if k == 'conv1.weight' or 'downsample.2' in k:
                new_state_dict[k] = v
                continue
            k = k[:-6] + 'adder'
        # print(k)
        new_state_dict[k] = v
    return new_state_dict


if args.resume:
    if os.path.isfile(args.resume):
        print("=> loading checkpoint '{}'".format(args.resume))
        checkpoint = torch.load(args.resume, map_location='cpu')
        args.start_epoch = checkpoint['epoch']
        best_prec1 = checkpoint['best_prec1']
        try:
            try:
                model.load_state_dict(checkpoint['state_dict'])
            except:
                model.load_state_dict(load_add_state_dict(checkpoint['state_dict']))
        except:
            model.load_state_dict(load_shiftadd_state_dict(checkpoint['state_dict']))
        if not args.eval_only:
            optimizer.load_state_dict(checkpoint['optimizer'])
        print("=> loaded checkpoint '{}' (epoch {}) Prec1: {:f}"
              .format(args.resume, checkpoint['epoch'], best_prec1))
    else:
        print("=> no checkpoint found at '{}'".format(args.resume))
else:
    save_checkpoint({'state_dict': model.state_dict()}, False, epoch='init', filepath=args.save)

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

if args.add_quant:
    shift_label += '_add-{}'.format(args.add_bits)

if args.sign_threshold_ps:
    shift_label += '_ps_thre-{}'.format(args.sign_threshold_ps)

args.save = os.path.join(args.save, shift_label)
if args.visualize:
    args.save = os.path.join('./visualization', args.save[2:])
if not os.path.exists(args.save):
    os.makedirs(args.save)

history_score = np.zeros((args.epochs, 4))

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

if 'shift' in args.arch:
    if args.shift_type == 'Q':
        shift_module = deepshift.modules_q.Conv2dShiftQ
    elif args.shift_type == 'PS':
        shift_module = deepshift.modules.Conv2dShift
    else:
        raise NotImplementedError

def get_shift_range(model):
    if 'shift' in args.arch:
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
            weight_unique = torch.unique(shift_weights)
            print(weight_unique)
            print('shift_range:', weight_unique.size()[0]-1)
        elif args.shift_type == 'PS':
            total = 0
            for m in model.modules():
                if isinstance(m, shift_module):
                    total += m.sign.data.numel()
            sign_weights = torch.zeros(total)
            shift_weights = torch.zeros(total)
            index = 0
            for m in model.modules():
                if isinstance(m, shift_module):
                    size = m.sign.data.numel()
                    sign_weights[index:(index+size)] = m.sign.data.view(-1).abs().clone()
                    shift_weights[index:(index+size)] = m.shift.data.view(-1).abs().clone()
                    index += size

            y, i = torch.sort(shift_weights)
            shift_unique = torch.unique(shift_weights)
            print(shift_unique)
            weight_dist = []
            for value in shift_unique:
                weight_dist.append(np.round((torch.sum(shift_weights == value) / float(total)) * 100, 2))
            print(weight_dist)
            print('shift_range:', shift_unique.size()[0]-1)
            print('pruning ratio:', np.round((torch.sum(sign_weights == 0) / float(total)) * 100, 2), '%')

from deepshift import utils
def train(epoch):
    model.train()
    global history_score
    avg_loss = 0.
    train_acc = 0.
    end_time = time.time()
    feat_loader = []
    idx_loader = []
    start_time = time.time()
    # batch_time = time.time()
    for batch_idx, (data, target) in enumerate(train_loader):

        # print('total time for one batch: {}'.format(time.time()-batch_time))
        # batch_time = time.time()

        # for name, m in model.named_modules():
        #     # print(name)
        #     # if isinstance(m, deepshift.modules.Conv2dShift):
        #     # if name == 'layer1.1.conv1':
        #     if name == 'layer2.2.conv1.0':
        #         pre = utils.round(m.shift.data, 'deterministic')
        #         pre_s = m.sign.data
                # print(m.shift.grad.data)

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



        # for name, m in model.named_modules():
        #     if isinstance(m, deepshift.modules.Conv2dShift):
        #         m.shift.grad = None
        #         m.sign.grad = None
        #     # if name == 'layer1.1.conv1':
        #     # if name == 'layer2.2.conv1.0':
        #     #     now = utils.round(m.shift.data, 'deterministic')
        #     #     now_s = m.sign.data
        #     #     print(torch.sum(abs(now - pre)))
        #         # print(torch.sum(abs(now_s - pre_s)))
        #     if isinstance(m, Adder2D):
        #         print(name, torch.norm(m.adder.grad))
        # exit()

        optimizer.step()

        # if args.add_quant:
        #     for name, m in model.named_modules():
        #         if isinstance(m, Adder2D):
        #             m.adder.data = m.round_weight_each_step(m.adder.data, m.weight_bits)
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.1f}%)]\tLoss: {:.6f}'.format(epoch, batch_idx * len(data), len(train_loader.dataset), 100. * batch_idx / len(train_loader), loss.item()))
    history_score[epoch][0] = avg_loss / len(train_loader)
    history_score[epoch][1] = np.round(train_acc / len(train_loader), 2)

    print('total time for one epoch: {}'.format(time.time()-start_time))

    if args.visualize:
        feat = torch.cat(feat_loader, 0)
        labels = torch.cat(idx_loader, 0)
        visualize(feat.data.cpu().numpy(), labels.data.cpu().numpy(), epoch)

def test():
    model.eval()
    test_loss = 0
    test_acc = 0
    test_acc_5 = 0
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
        test_acc_5 += prec5.item()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Prec1: {}/{} ({:.2f}%), Prec5: ({:.2f}%)\n'.format(
        test_loss, test_acc, len(test_loader), test_acc / len(test_loader), test_acc_5 / len(test_loader)))
    return np.round(test_acc / len(test_loader), 2), np.round(test_acc_5 / len(test_loader), 2)


best_prec1 = 0.
best_prec5 = 0.
vis = visualizer(args.epochs, num_classes)
for epoch in range(args.start_epoch, args.epochs):

    if args.eval_only:
        with torch.no_grad():
            prec1, prec5 = test()
            print('Prec1: {}; Prec5: {}'.format(prec1, prec5))
        exit()


    if args.lr_decay == 'stepwise':
        # step-wise LR schedule
        if epoch in args.schedule:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.1
    elif args.lr_decay == 'cosine':
        schedule_cosine_lr_decay.step(epoch)
    elif args.lr_decay == 'cyclic_cosine':
        scheduler_cyclic_cosine_lr_decay.step(epoch)
    else:
        raise NotImplementedError

    train(epoch)
    if args.vis_tropical:
        vis.visualize(list(model.children())[-1].weight.data.cpu(), epoch, dir=args.visualize_dir)
    with torch.no_grad():
        prec1, prec5 = test()
    history_score[epoch][2] = prec1
    history_score[epoch][3] = prec5
    np.savetxt(os.path.join(args.save, 'record.txt'), history_score, fmt='%10.5f', delimiter=',')
    is_best = prec1 > best_prec1
    best_prec1 = max(prec1, best_prec1)
    best_prec5 = max(prec5, best_prec5)
    model_rounded = round_shift_weights(model, clone=False)
    get_shift_range(model_rounded)
    save_checkpoint({
        'epoch': epoch + 1,
        'state_dict': model_rounded.state_dict(),
        'best_prec1': best_prec1,
        'optimizer': optimizer.state_dict(),
    }, is_best, epoch, filepath=args.save)
    # break

print("Best accuracy: " + str(best_prec1))
history_score[-1][0] = best_prec1
history_score[-1][1] = best_prec5
np.savetxt(os.path.join(args.save, 'record.txt'), history_score, fmt='%10.5f', delimiter=',')