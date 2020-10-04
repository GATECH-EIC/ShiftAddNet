import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import math

import os

outputdir = "./visualize/"

class visualizer:
    def __init__(self, num_epochs=160, num_classes=10):
        self.num_epochs = num_epochs
        self.historical_angles = np.zeros((num_epochs, num_classes * 2))

    def getAngles(self, data, epoch, dir):
        up = np.roll(data, 1, axis=0)
        down = np.roll(data, -1, axis=0)
        down_vec = data - up
        up_vec = data - down
        dot_p = np.sum(down_vec * up_vec, axis=1)
        down_vec_abs = np.sqrt(np.sum(down_vec * down_vec, axis=1))
        up_vec_abs = np.sqrt(np.sum(up_vec * up_vec, axis=1))
        result = dot_p / (down_vec_abs * up_vec_abs)
        result = np.pi - np.arccos(result)
        total_length = np.sum(down_vec_abs)
        dists = list(down_vec_abs / total_length)
        indices = -down_vec[:,1]*up_vec[:,0] + down_vec[:,0]*up_vec[:,1] > 0
        result[indices] = -result[indices]
        angles = list(result)
        to_return = dists + angles
        first = dists[0]
        dists = dists[1:] + [first]
        cum_x = 0
        x_p = [cum_x]
        for d in dists:
            cum_x += d
            x_p.append(cum_x)
            x_p.append(cum_x)
        x_p = x_p[:-1]
        cum_ang = angles[0]
        y_p = [cum_ang, cum_ang]
        for a in angles[1:]:
            cum_ang += a
            y_p.append(cum_ang)
            y_p.append(cum_ang)
        plt.scatter(x_p,y_p)
        plt.plot(x_p,y_p)
        plt.title(str(epoch) + "_turning", fontsize=16)
        plt.savefig(dir + str(epoch) + "_turning.png")
        plt.clf()
        return to_return

    def visualize(self, data, epoch, dir):
        my_folder = outputdir + dir + "/"
        if not os.path.exists(my_folder):
            os.makedirs(my_folder)
        if type(data).__module__ != np.__name__:
            data = data.numpy()
        # data[:,[0, 1]] = data[:,[1, 0]]
        angles = self.getAngles(data, epoch, my_folder)
        self.historical_angles[epoch] = angles
        np.savetxt(my_folder+"record.txt", self.historical_angles, fmt = '%10.5f', delimiter=',')
        xp = list(data[:, 0])
        yp = list(data[:, 1])
        xp.append(xp[0])
        yp.append(yp[0])
        colors = cm.rainbow(np.linspace(0,1,len(yp)-1))
        # print(xp)
        plt.scatter(xp[:-1], yp[:-1], color=colors)
        plt.plot(xp, yp)
        plt.title(epoch, fontsize=16)
        plt.savefig(my_folder + str(epoch) + ".png")
        plt.clf()


def calcDist(l1, d1, l2, d2):
    d1 = [0] + d1
    d2 = [0] + d2
    for i in range(1, len(d1)):
        d1[i] += d1[i-1]
        d2[i] += d2[i-1]
        # l1[i] += l1[i-1]
        # l2[i] += l2[i-1]
    # print(d1)
    # print(d2)
    i1 = 0
    i2 = 0
    lo = []
    do = []
    while i1 != len(l1) and i2 != len(l2):
        lr = l2[i2]
        dr = d1[i1] - d2[i2]
        if l1[i1] > l2[i2]:
            l1[i1] = l1[i1] - l2[i2]
            i2 += 1
        else:
            lr = l1[i1]
            l2[i2] = l2[i2] - l1[i1]
            if l2[i2] == 0:
                i2 += 1
            i1 += 1
        lo.append(lr)
        do.append(dr)
    return lo, do

def visualizeDist(l1,d1,l2,d2):
    dists, angles = calcDist(l1,d1,l2,d2)
    print(dists)
    print(angles)
    vals = list(map(lambda x: x**2, angles))
    cum = 0
    for i in range(len(dists)):
        cum += dists[i] * vals[i]
    returnV = math.sqrt(cum)
    print("Value: ", returnV)
    cum_x = 0
    x_p = [cum_x]
    for d in dists:
        cum_x += d
        x_p.append(cum_x)
        x_p.append(cum_x)
    x_p = x_p[:-1]
    y_p = []
    for a in angles:
        y_p.append(a)
        y_p.append(a)
    # print(x_p)
    # print(y_p)
    plt.scatter(x_p, y_p)
    plt.plot(x_p, y_p)
    plt.title("turning", fontsize=16)
    plt.show()

def test1():
    resnet_18_l = [0.10849,   0.09012,   0.14690,   0.08424,   0.07548,   0.06378,   0.08530,   0.12059,   0.14780,   0.07730]
    resnet_18_d = [2.85074,   2.70006,  -1.56091,  -1.86364,   2.63755,   1.75712,   2.79340,  -1.95066,   2.44057,   2.76214]
    resnet_34_l = [0.11218,   0.09523,   0.15414,   0.07863,   0.06508,   0.06129,   0.08223,   0.12109,   0.14936,   0.08077]
    resnet_34_d = [2.85143,   2.70527,  -1.61512,  -1.87177,   2.62351,   1.82452,   2.79955,  -1.92280,   2.41455,   2.75724]
    vgg16_70_l = [0.10689,   0.08774,   0.07646,   0.07925,   0.10622,   0.08335,   0.10252,   0.12393,   0.16130,   0.07233]
    vgg16_90_l = [0.11512,   0.09672,   0.08301,   0.07211,   0.10002,   0.07790,   0.09780,   0.12225,   0.16320,   0.07188]
    vgg16_70_d = [-2.79004,  -2.07985,   0.30123,  -2.12770,   2.74480,   1.61702,   2.80769,  -2.29280,  -1.78716,  -2.67637]
    vgg16_90_d = [2.87221,   2.11969,  -0.53863,   2.17683,  -2.76256,  -1.45068,  -2.88341,   2.23499,   1.77969,   2.73507]
    vgg16_4_bit_l = [0.12741,   0.14714,   0.18163,   0.05971,   0.02967,   0.03985,   0.04329,   0.08074,   0.17858,   0.11197]
    vgg16_4_bit_d = [-2.95573,  -2.63826,   2.17131,  -2.56512,   2.94125,  -2.41291,   2.64663,   1.83478,  -2.45982,  -2.84533]
    vgg16_2_bit_l = [0.17338,   0.14427,   0.11152,   0.05832,   0.03114,   0.06802,   0.06782,   0.08833,   0.13972,   0.11748]
    vgg16_2_bit_d = [-3.08017,   2.05823,  -3.10166,   2.25084,  -2.82317,  -3.09568,  -2.42401,  -1.79215,   3.04093,  2.68363]
    d1 = [-2.80310,  -2.72550,   1.50190,   1.95942,  -2.57412,  -1.81261,  -2.72949,   1.92498,  -2.59050,  -2.71735]
    l1 = [0.10931, 0.08832, 0.13613, 0.08585, 0.08863, 0.07024, 0.09070, 0.12064, 0.13292, 0.07726]
    d2 = list(map(lambda x: x, vgg16_4_bit_d))
    l2 = vgg16_4_bit_l
    visualizeDist(l1, d1, l2, d2)

def test2():
    rectangle = np.asarray([[0,0],
                       [0,2],
                       [1,2],
                       [1,0]])
    triangle = np.asarray([[0,0],[2,0], [0,2]])
    data = rectangle
    vis = visualizer(2,4)
    vis.visualize(data,0,"test")


if __name__ == '__main__':
    test2()