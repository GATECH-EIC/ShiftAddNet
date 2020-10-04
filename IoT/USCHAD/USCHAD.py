import sys, os
import time
import numpy as np
import random
import shutil
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics.classification import accuracy_score, recall_score, f1_score
import scipy.stats as st

import argparse
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.autograd import Variable
import torch.utils.data as data

import models
from adder import adder
from deepshift.convert import convert_to_shift, round_shift_weights, count_layer_type

class CNN(nn.Module):
    def __init__(self, num_classes):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 5, kernel_size=(12, 5))
        self.bn1 = nn.BatchNorm2d(5)
        self.conv2 = nn.Conv2d(5, 10, kernel_size=(5, 1))
        self.bn2 = nn.BatchNorm2d(10)
        # self.conv3 = nn.Conv2d(36, 24, kernel_size=(12, 1))
        self.pool1 = nn.MaxPool2d((4,4))
        self.pool2 = nn.MaxPool2d((2,2))
        # self.fc1 = nn.Linear(8120, 120)
        # self.fc2 = nn.Linear(120, num_classes)
        self.fc1 = nn.Conv2d(590, num_classes, 1, bias=False)
        self.fc2 = nn.BatchNorm2d(num_classes)

    def forward(self, inputs):
        print(inputs.shape)
        x = self.pool1(F.relu(self.bn1(self.conv1(inputs))))
        print(x.shape)
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        print(x.shape)
        # x = self.pool(F.relu(self.conv3(x)))
        # x = torch.flatten(x, start_dim=1)
        x = x.view(x.size(0), -1)
        x = torch.unsqueeze(x, dim=2)
        x = torch.unsqueeze(x, dim=3)

        x = self.fc1(x)
        x = self.fc2(x)
        # return F.softmax(x)
        return x.view(x.size(0), -1)


class MHEALTH(data.Dataset):
    def __init__(self, input_data, input_data_labels):

        self.data = torch.from_numpy(input_data).float()
        temp_arr = input_data_labels
        final_arr = []
        final_arr = np.argmax(temp_arr, axis=1)
        # final_arr = temp_arr
        # Reshaping array to get normalized feature vector
        # for x in temp_arr:
        #     number_of_features = 12

        #     temp_a = [0. for x in range(number_of_features + 1)]
        #     temp_a[int(x)-1] = 1.
        #     final_arr.append(temp_a)

        final_arr = np.asarray(final_arr)
        self.target = torch.from_numpy(final_arr).float()  # Labels for input
        self.target = self.target.reshape(-1, 1)

        self.n_samples = self.data.shape[0]  # Length of input

        print(self.data.shape)
        # print(self.target)

    def __len__(self):  # Length of the dataset.
        return self.n_samples

    def __getitem__(self, index):  # Function that returns one point and one label.
        return torch.Tensor(self.data[index]), torch.Tensor(self.target[index])

# Training settings
parser = argparse.ArgumentParser(description='PyTorch AdderNet Trainning')
parser.add_argument('--data', type=str, default='/data3/imagenet-data/raw-data/', help='path to imagenet')
parser.add_argument('--dataset', type=str, default='USCHAD', help='training dataset')
parser.add_argument('--arch', default='CNN', type=str, help='architecture to use')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M', help='SGD momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float, metavar='W', help='weight decay')
parser.add_argument('--schedule', type=int, nargs='+', default=[80, 120], help='learning rate schedule')
parser.add_argument('--lr', type=float, default=0.1, metavar='LR', help='learning rate')
parser.add_argument('--lr-sign', default=None, type=float, help='separate initial learning rate for sign params')
parser.add_argument('--lr_decay', default='stepwise', type=str, choices=['stepwise', 'poly', 'cosine', 'cyclic_cosine'])
parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')
parser.add_argument('--log_interval', type=int, default=10, metavar='N', help='how many batches to wait before logging training status')
parser.add_argument('--save', default='./rebuttal/MHEALTH', type=str, metavar='PATH', help='path to save prune model')
parser.add_argument('--epochs', type=int, default=160, metavar='N', help='number of epochs to train')

# shift hyper-parameters
parser.add_argument('--shift_depth', type=int, default=100, help='how many layers to convert to shift')
parser.add_argument('--shift_type', type=str, default='PS', choices=['Q', 'PS'], help='shift type for representing weights')
parser.add_argument('--rounding', default='deterministic', choices=['deterministic', 'stochastic'])
parser.add_argument('--weight_bits', type=int, default=5, help='number of bits to represent the shift weights')
parser.add_argument('--sign_threshold_ps', type=float, default=0.5, help='can be controled')
parser.add_argument('--use_kernel', type=lambda x:bool(distutils.util.strtobool(x)), default=False, help='whether using custom shift kernel')
parser.add_argument('--shift_quant_bits', type=int, default=32, help='quantization training for shift layer')
# add hyper-parameters
parser.add_argument('--add_quant', type=bool, default=False, help='whether to quantize adder layer')
parser.add_argument('--add_bits', type=int, default=8, help='number of bits to represent the adder filters')
parser.add_argument('--quantize_v', type=str, default='sbm', help='quantize version')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

if not os.path.exists(args.save):
    os.makedirs(args.save)
else:
    print('delete origin folder')
    shutil.rmtree(args.save)
    os.makedirs(args.save)

# data
np.random.seed(12227)
if args.dataset == 'MHEALTH':
    data_input_file = '../WearableSensorData/data/LOSO/MHEALTH.npz'
elif args.dataset == 'USCHAD':
    data_input_file = '../WearableSensorData/data/LOSO/USCHAD.npz'

def check_neighboring(sis, elem1, elem2):
    if(len(sis)==1):
        return False
    for i in range(0, len(sis)-1):
        if sis[i] == elem1 and sis[i+1] == elem2:
            return True
        if sis[i] == elem2 and sis[i + 1] == elem1:
            return True
    return False

def activity_image(raw_signals):
    seq =  np.arange(0, raw_signals.shape[3], 1)
    sis = []
    n_sis = 1
    i = 0
    j = i+1
    sis.append(i)
    while i!=j:
        if j==len(seq):
            j=0
        elif check_neighboring(sis, i, j) == False:
            sis.append(j)
            i = j
            j = j+1
        else:
            j = j + 1

    output = []
    for sample in raw_signals:
        signal_image = sample[0]
        signal_image = signal_image[:, sis]
        signal_image = np.transpose(signal_image)

        fshift = np.fft.fftshift(signal_image)
        fshift = np.transpose(fshift)
        # import cv2
        # cv2.imshow('tete', magnitude_spectrum)
        # cv2.waitKey(1000)

        output.append([fshift])

    output = np.array(output)
    return output


tmp = np.load(data_input_file)

X = tmp['X']
y = tmp['y']
folds = tmp['folds']

n_class = y.shape[1]
X = activity_image(X)
_, _, img_rows, img_cols = X.shape
avg_acc = []
avg_recall = []
avg_f1 = []

# model
# model = CNN(num_classes=n_class)
# if 'add' in args.arch:
#     model = CNN_add(num_classes=n_class)
def get_model():
    # if not 'se' in args.arch:
    #     model = models.__dict__[args.arch](num_classes=n_class, quantize=args.add_quant, weight_bits=args.add_bits, quantize_v=args.quantize_v)
    # else:
    #     threshold = 5e-3
    #     sign_threshold = 0.5
    #     distribution = 'kaiming_normal'
    #     model = models.__dict__[args.arch](threshold=threshold, sign_threshold=sign_threshold, distribution=distribution,
    #         num_classes=n_class, quantize=args.add_quant, weight_bits=args.add_bits, quantize_v=args.quantize_v)
    # if 'shift' in args.arch: # no pretrain
    #     model, _ = convert_to_shift(model, args.shift_depth, args.shift_type, convert_weights=False, use_kernel=args.use_kernel, rounding=args.rounding,
    #         weight_bits=args.weight_bits, sign_threshold_ps=args.sign_threshold_ps, quant_bits=args.shift_quant_bits)

    model = CNN(num_classes=n_class)

    if args.cuda:
        model.cuda()

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

    optimizer = torch.optim.Adam(model.parameters(), args.lr, weight_decay=args.weight_decay)
    return model, optimizer

# train
epochs = args.epochs
batch_size = 100
# optimizer = torch.optim.SGD(params_dict, args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
# optimizer = torch.optim.Adadelta(params_dict, args.lr, weight_decay=args.weight_decay)

history_score = np.zeros((epochs, 4))

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


def train(epoch, train_loader):
    model.train()
    global history_score
    avg_loss = 0.
    train_acc = 0.
    start_time = time.time()
    # batch_time = time.time()
    for batch_idx, (data, target) in enumerate(train_loader):
        if args.cuda:
            data, target = data.cuda(), target.cuda()
            target = torch.squeeze(target)
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)
        # print(output[0])
        # print(target, torch.argmax(output, axis=1))
        loss = F.cross_entropy(output, target.long())
        avg_loss += loss.item()
        prec1, prec5 = accuracy(output.data, target.data, topk=(1, 5))
        train_acc += prec1.item()
        loss.backward()
        optimizer.step()

        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.1f}%)]\tLoss: {:.6f}'.format(epoch, batch_idx * len(data), len(train_loader.dataset), 100. * batch_idx / len(train_loader), loss.item()))
    history_score[epoch][0] = avg_loss / len(train_loader)
    history_score[epoch][1] = np.round(train_acc / len(train_loader), 2)

    # print('total time for one epoch: {}'.format(time.time()-start_time))

def test(test_loader):
    model.eval()
    test_loss = 0
    test_acc = 0
    test_acc_5 = 0
    for data, target in test_loader:
        if args.cuda:
            data, target = data.cuda(), target.cuda()
            target = torch.squeeze(target)
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
        test_loss += F.cross_entropy(output, target.long(), size_average=False).item() # sum up batch loss
        prec1, prec5 = accuracy(output.data, target.data, topk=(1, 5))
        test_acc += prec1.item()
        test_acc_5 += prec5.item()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Prec1: {}/{} ({:.2f}%), Prec5: ({:.2f}%)\n'.format(
        test_loss, test_acc, len(test_loader), test_acc / len(test_loader), test_acc_5 / len(test_loader)))
    return np.round(test_acc / len(test_loader), 2), np.round(test_acc_5 / len(test_loader), 2)

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


for i in range(0, len(folds)):
# for i in range(0, 3):

    model, optimizer = get_model()

    train_idx = folds[i][0]
    test_idx = folds[i][1]

    X_train = X[train_idx]
    y_train = y[train_idx]
    X_test = X[test_idx]
    y_test = y[test_idx]

    train_dataset = MHEALTH(X_train, y_train)
    test_dataset = MHEALTH(X_test, y_test)

    kwargs = {'num_workers': 1, 'pin_memory': True, 'drop_last': True}
    train_loader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, **kwargs)
    print(len(train_loader))
    test_loader = data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True, **kwargs)


    best_prec1 = 0.
    best_prec5 = 0.
    for epoch in range(epochs):

        # step-wise LR schedule
        if epoch in args.schedule:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.1

        train(epoch, train_loader)
        with torch.no_grad():
            prec1, prec5 = test(test_loader)

        history_score[epoch][2] = prec1
        history_score[epoch][3] = prec5

        np.savetxt(os.path.join(args.save, 'record.txt'), history_score, fmt='%10.5f', delimiter=',')
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        best_prec5 = max(prec5, best_prec5)


        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
            'best_prec5': best_prec5,
            'optimizer': optimizer.state_dict(),
        }, is_best, epoch, filepath=args.save)

    print("Best accuracy: " + str(best_prec1))
    history_score[-1][0] = best_prec1
    history_score[-1][1] = best_prec5
    np.savetxt(os.path.join(args.save, 'record.txt'), history_score, fmt='%10.5f', delimiter=',')

    avg_acc.append(best_prec1)
    del model

print(avg_acc)
print('Mean Accuracy[{:.4f}]'.format(np.mean(avg_acc)))
history_score[-1][0] = np.mean(avg_acc)
np.savetxt(os.path.join(args.save, 'record.txt'), history_score, fmt='%10.5f', delimiter=',')