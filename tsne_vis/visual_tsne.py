import numpy as np
import gzip
# import cPickle
import pickle as cPickle
import argparse

#Import scikitlearn for machine learning functionalities
import sklearn
from sklearn.manifold import TSNE

parser = argparse.ArgumentParser(description='PyTorch TSNE Plot')
parser.add_argument('--save_dir', type=str, default='resnet20_add_FIX8', help='path to save directory')
parser.add_argument('--scratch', action='store_true', help='whether generate output_2d from scratch')
parser.add_argument('--dim_3d', action='store_true', help='whether to show 3D perspective')
args = parser.parse_args()

save_dir = args.save_dir

font_board = 2

output = np.load(save_dir + '/output.npy').astype(np.float64)
# data = np.load(save_dir + '/data.npy')
target = np.load(save_dir + '/target.npy')
# print('data shape: ', data.shape)
print('target shape: ', target.shape)
print('output shape: ', output.shape)

if not args.dim_3d:
    if args.scratch:
        output_2d = TSNE(perplexity=30).fit_transform(output)
        np.save(save_dir + '/output_2d.npy', output_2d) #, allow_pickle=False)
    else:
        output_2d = np.load(save_dir + '/output_2d.npy')

    target = target.reshape(target.shape[0])

    import matplotlib.pyplot as plt
    import matplotlib.patheffects as pe

    fig = plt.figure(figsize=(8,6))
#     ax = plt.subplot(aspect='equal')
    ax = fig.add_subplot(1,1,1)
    sc = ax.scatter(output_2d[:, 0], output_2d[:, 1], lw=0, s=10, c=target)

    # Add the labels for each digit.
    txts = []
    for i in range(10):
        # Position of each label.
        xtext, ytext = np.median(output_2d[target == i, :], axis=0)
        txt = ax.text(xtext, ytext, str(i), fontsize=24)
        txt.set_path_effects([pe.Stroke(linewidth=5, foreground="w"), pe.Normal()])
        txts.append(txt)

    ax.spines['bottom'].set_linewidth(font_board)
    ax.spines['bottom'].set_color('black')
    ax.spines['left'].set_linewidth(font_board)
    ax.spines['left'].set_color('black')


    ax.spines['top'].set_linewidth(font_board)
    ax.spines['top'].set_color('black')
    ax.spines['right'].set_linewidth(font_board)
    ax.spines['right'].set_color('black')

    ax.set_xticks([])
    ax.set_yticks([])


    plt.savefig(save_dir + '/{}_output_2d.svg'.format(save_dir), bbox_inches='tight')
else:
    # 3D
    if args.scratch:
        output_3d = TSNE(perplexity=30, n_components=3).fit_transform(output)
        np.save(save_dir + '/output_3d.npy', output_3d) #, allow_pickle=False)
    else:
        output_3d = np.load(save_dir + '/output_3d.npy')


    target = target.reshape(target.shape[0])

    output_3d_1 = output_3d[target==4, :]
    output_3d_2 = output_3d[target==7, :]
    output_3d = np.vstack((output_3d_1, output_3d_2))
    target_1 = target[target==4]
    target_2 = target[target==7]
    target = np.vstack((np.expand_dims(target_1, axis=1), np.expand_dims(target_2, axis=1)))
    target = target.reshape(target.shape[0])
    print(output_3d.shape)
    print(target.shape)

    import matplotlib.pyplot as plt
    import matplotlib.patheffects as pe
    from mpl_toolkits.mplot3d import Axes3D

    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(111, projection='3d')
    color = ['#440154', '#482878', '#3E4989', '#31688E', '#26828E', '#1F9E89', '#35B779', '#6ECE58', '#B5DE2B', '#FDE725']
    def get_color(target):
        _color = []
        for i in range(target.shape[0]):
            _color.append(color[target[i]])
        return np.array(_color)
    sc = ax.scatter(output_3d[:, 0], output_3d[:, 1], output_3d[:, 2], lw=0, s=10, c=get_color(target))


    # ax.spines['bottom'].set_linewidth(font_board)
    # ax.spines['bottom'].set_color('black')
    # ax.spines['left'].set_linewidth(font_board)
    # ax.spines['left'].set_color('black')


    # ax.spines['top'].set_linewidth(font_board)
    # ax.spines['top'].set_color('black')
    # ax.spines['right'].set_linewidth(font_board)
    # ax.spines['right'].set_color('black')

    ax.set_xticklabels([])
    ax.set_yticks([])
    ax.set_zticklabels([])
    # ax.set_zticks([])
    # ax.grid()

    # Add the labels for each digit.
    txts = []
    for i in range(10):
        # Position of each label.
        if output_3d[target == i, :].shape[0] > 10:
            print(i)
            xtext, ytext, ztext = np.median(output_3d[target == i, :], axis=0)
            txt = ax.text(xtext, ytext, ztext, str(i), fontsize=24)
            txt.set_path_effects([pe.Stroke(linewidth=5, foreground="w"), pe.Normal()])
            txts.append(txt)
#     ax.legend()

    # plt.show()

    plt.savefig(save_dir + '/{}_output_3d_4_7.svg'.format(save_dir), bbox_inches='tight')