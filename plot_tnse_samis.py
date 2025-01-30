from __future__ import print_function

import argparse
import os.path as osp
try:
    import imageio
except ImportError:
    raise ImportError("Please install imageio by: pip3 install imageio")
import matplotlib
import torch.utils
import torch.utils
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as clr

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms

import os
import sys
import time
import torch
import torch.nn as nn
import torch.nn.init as init

from utils import *
from mqnet_utils import str_to_bool
from src.metrics import Proxy

import glob



# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"
parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--total_epochs', default=100, type=int, help='epoch size')
parser.add_argument('--selected_epoch', default=None, type=int, help='epoch size')
parser.add_argument('--mode', type=str, default='sequential', choices=['sequential', 'single'])
parser.add_argument('--bs', default=512, type=int, help='batch size')
parser.add_argument("--device", default='0',
                    help="Set to 0 or 1 to enable CUDA training, cpu otherwise")
parser.add_argument('--save-dir', type=str, default='images')
parser.add_argument('--experiment_name', default='standard_resnet18_over_cifar10_with_sgd_bn_sched_cosine')
parser.add_argument("--seed", type=int, default=1,
                    help="random seed (default: 1)")
parser.add_argument('--models_folder', type=str, default='ckpts')
parser.add_argument("--resume", default=False, type=str_to_bool, help="whether parallel or not")
parser.add_argument('--split', type=str, default='train')
args = parser.parse_args()


device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)




def get_mean_and_std(dataset):
    '''Compute the mean and std value of dataset.'''
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=2)
    mean = torch.zeros(3)
    std = torch.zeros(3)
    print('==> Computing mean and std..')
    for inputs, targets in dataloader:
        for i in range(3):
            mean[i] += inputs[:,i,:,:].mean()
            std[i] += inputs[:,i,:,:].std()
    mean.div_(len(dataset))
    std.div_(len(dataset))
    return mean, std

def init_params(net):
    '''Init layer parameters.'''
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            init.kaiming_normal(m.weight, mode='fan_out')
            if m.bias:
                init.constant(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            init.constant(m.weight, 1)
            init.constant(m.bias, 0)
        elif isinstance(m, nn.Linear):
            init.normal(m.weight, std=1e-3)
            if m.bias:
                init.constant(m.bias, 0)


# _, term_width = os.popen('stty size', 'r').read().split()
# term_width = int(term_width)

TOTAL_BAR_LENGTH = 65.
last_time = time.time()
begin_time = last_time
def progress_bar(current, total, msg=None):
    global last_time, begin_time
    if current == 0:
        begin_time = time.time()  # Reset for new bar.

    cur_len = int(TOTAL_BAR_LENGTH*current/total)
    rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1

    sys.stdout.write(' [')
    for i in range(cur_len):
        sys.stdout.write('=')
    sys.stdout.write('>')
    for i in range(rest_len):
        sys.stdout.write('.')
    sys.stdout.write(']')

    cur_time = time.time()
    step_time = cur_time - last_time
    last_time = cur_time
    tot_time = cur_time - begin_time

    L = []
    L.append('  Step: %s' % format_time(step_time))
    L.append(' | Tot: %s' % format_time(tot_time))
    if msg:
        L.append(' | ' + msg)

    msg = ''.join(L)
    sys.stdout.write(msg)
    # for i in range(term_width-int(TOTAL_BAR_LENGTH)-len(msg)-3):
    #     sys.stdout.write(' ')

    # Go back to the center of the bar.
    # for i in range(term_width-int(TOTAL_BAR_LENGTH/2)+2):
    #     sys.stdout.write('\b')
    sys.stdout.write(' %d/%d ' % (current+1, total))

    if current < total-1:
        sys.stdout.write('\r')
    else:
        sys.stdout.write('\n')
    sys.stdout.flush()

def format_time(seconds):
    days = int(seconds / 3600/24)
    seconds = seconds - days*3600*24
    hours = int(seconds / 3600)
    seconds = seconds - hours*3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes*60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds*1000)

    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's'
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms'
        i += 1
    if f == '':
        f = '0ms'
    return f

def write_record(file_path,str):
    if not os.path.exists(file_path):
        # os.makedirs(file_path)
        os.system(r"touch {}".format(file_path))
    f = open(file_path, 'a')
    f.write(str)
    f.close()

def count_parameters(model,all=True):
    # If all= Flase, we only return the trainable parameters; tested
    return sum(p.numel() for p in model.parameters() if p.requires_grad or all)

def adjust_learning_rate(optimizer, epoch, lr):
    """Sets the learning rate to the initial LR decayed by 10 every 80 epochs"""
    lr = lr * (0.1 ** (epoch // 70))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def main():
    # Data
    print('==> Preparing data..')
    
    normalize = transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                    std=[x / 255.0 for x in [63.0, 62.1, 66.7]])
    test_transform = transforms.Compose([transforms.ToTensor(), normalize])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=test_transform)
    train_indx = np.array(range(len(trainset.targets)))
    trainset.data = trainset.data[train_indx, :, :, :]
    trainset.targets = np.array(trainset.targets)[train_indx].tolist()
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.bs, shuffle=False, num_workers=1)

    if args.resume:
        found_plots = glob.glob(osp.join(args.save_dir, args.split, '*.png'))
        epochs = [int(f.split('_')[-1].split('.')[0]) for f in found_plots]
        start_epoch = max(epochs) + 1
        print('Last epoch found is {}! Resuming from {}...'.format(max(epochs), start_epoch))
    else:
        start_epoch = 1


    for epoch in range(start_epoch, args.total_epochs+1):
        print('EPOCH: {}'.format(epoch))
        cuda_available = torch.cuda.is_available()
        if cuda_available and args.device != 'cpu':
            dev_str = 'cuda:{}'.format(args.device)
            device = torch.device(dev_str)
            print('Runner is setup using CUDA enabled device: {}'.format(torch.cuda.get_device_name(dev_str)))
        else:
            device = torch.device('cpu')
            print('Runner is setup using CPU! Training and inference will be very slow!')
        cudnn.benchmark = True  # Should make training go faster for large models
        print('Loading model...')
        models_folder = os.path.join(args.models_folder, args.experiment_name, 'seed_{}'.format(args.seed))
        net = torch.load(os.path.join(models_folder, 'epoch_{}.pt'.format(epoch)))
        net = net.to(device)
        net.eval()

        with torch.no_grad():
            print('TRAINSET: Getting embeddings through model...')
            train_features = []
            train_labels = []
            start = time.time()
            for i, (inputs, targets) in enumerate(trainloader):
                print('Batch {} out of {}...'.format(i, len(trainloader)), end='\r')
                inputs, targets = inputs.to(device), targets.to(device)
                _, features = net(inputs, return_emb=True)
                train_features.append(features)
                train_labels.append(targets)
            print()
            print('Time taken to get embeddings: {}'.format(format_time(time.time() - start)))
            print('TRAINSET: Applying TSNE to embeddings...')
            fea, label = embedding(train_features, train_labels)
            print('TRAINSET: Plotting TSNE results...')
            plot_features(fea, label, 10, epoch, args.split)

    generate_gif(gif_name='result.gif')


def main_single_epoch():
    # Data
    print('==> Preparing data..')
    
    normalize = transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                    std=[x / 255.0 for x in [63.0, 62.1, 66.7]])
    test_transform = transforms.Compose([transforms.ToTensor(), normalize])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=test_transform)
    train_indx = np.array(range(len(trainset.targets)))
    trainset.data = trainset.data[train_indx, :, :, :]
    trainset.targets = np.array(trainset.targets)[train_indx].tolist()
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.bs, shuffle=False, num_workers=1)

    if args.selected_epoch is None:
        raise ValueError('Must select an epoch!')
    epoch = args.selected_epoch

    print('EPOCH: {}'.format(epoch))
    cuda_available = torch.cuda.is_available()
    if cuda_available and args.device != 'cpu':
        dev_str = 'cuda:{}'.format(args.device)
        device = torch.device(dev_str)
        print('Runner is setup using CUDA enabled device: {}'.format(torch.cuda.get_device_name(dev_str)))
    else:
        device = torch.device('cpu')
        print('Runner is setup using CPU! Training and inference will be very slow!')
    cudnn.benchmark = True  # Should make training go faster for large models
    print('Loading model...')
    models_folder = os.path.join(args.models_folder, args.experiment_name, 'seed_{}'.format(args.seed))
    net = torch.load(os.path.join(models_folder, 'epoch_{}.pt'.format(epoch)))
    net = net.to(device)
    net.eval()

    with torch.no_grad():
        print('TRAINSET: Getting embeddings through model...')
        train_features = []
        train_labels = []
        start = time.time()
        for i, (inputs, targets) in enumerate(trainloader):
            print('Batch {} out of {}...'.format(i, len(trainloader)), end='\r')
            inputs, targets = inputs.to(device), targets.to(device)
            _, features = net(inputs, return_emb=True)
            train_features.append(features)
            train_labels.append(targets)
        print()
        print('Time taken to get embeddings: {}'.format(format_time(time.time() - start)))
        print('TRAINSET: Applying TSNE to embeddings...')
        fea, label = embedding(train_features, train_labels)
        print('TRAINSET: Plotting TSNE results...')
        plot_features(fea, label, 10, epoch, args.split)


def embedding(featureList, labelList, mode='tsne'):
    assert len(featureList) == len(labelList)
    assert len(featureList) > 0
    feature = featureList[0]
    label = labelList[0]
    for i in range(1, len(labelList)):
        feature = torch.cat([feature,featureList[i]],dim=0)
        label = torch.cat([label, labelList[i]], dim=0)

    feature = feature.cpu().detach().numpy()
    label = label.cpu().detach().numpy()
    # Using PCA or TSNE to reduce dimension to a reasonable dimension as recommended in
    if mode == 'pca':
        representation = PCA(n_components=50).fit_transform(feature)
    elif mode == 'tsne':
        representation = TSNE(n_components=2).fit_transform(feature)
    else:
        raise ValueError('Mode "{}" not available!'.format(mode))
    return representation, label


def debug_main():
    features = np.random.random((50000, 2))
    labels = np.random.randint(low=0, high=10, size=(50000,))
    plot_features(features,labels,num_classes=10, epoch=1, prefix='train/')


def plot_features(features, labels, num_classes, epoch, prefix):
    """Plot features on 2D plane.

    Args:
        features: (num_instances, num_features).
        labels: (num_instances).
    """

    samis_scores = Proxy(mode='samis_loss', dataset='cifar10').get_dict_form(sort=False)
    train_indx = np.array(list(samis_scores.keys()))

    my_cmap = clr.LinearSegmentedColormap.from_list('custom', ['#008000','#FF0000'], N=256)

    fig = plt.figure()
    ax = plt.subplot(111)   

    shapes = ['o', 'v', '^', '<', '>', 's', 'P', 'p', '*', '+']
    for label_idx in range(num_classes):
        ax.scatter(
            features[labels == label_idx, 0],
            features[labels == label_idx, 1],
            c=[samis_scores[i] for i in train_indx[labels == label_idx]],
            cmap=my_cmap,
            marker=shapes[label_idx],
            s=40,
        )
        # plt.clim(0, 1)
    box = ax.get_position()
    ax.set_position([box.x0, box.y0 + box.height * 0.1,
                    box.width, box.height * 0.9])
    ax.axes.xaxis.set_ticklabels([])
    ax.axes.yaxis.set_ticklabels([])

    # Put a legend below current axis
    ax.legend(['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'], loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=5)
    # plt.legend(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'], loc='upper right')
    # plt.colorbar().set_label('SAMIS', rotation=270)

    norm = plt.Normalize(0, 1)
    sm = plt.cm.ScalarMappable(cmap=my_cmap, norm=norm)
    sm.set_array([])
    ax.figure.colorbar(sm, ax=plt.gca()).set_label('SAMIS', size='large', rotation=270)

    plt.xticks([], [])
    plt.yticks([], [])

    plt.title('EPOCH: {}'.format(epoch))

    dirname = osp.join(args.save_dir, prefix)
    os.makedirs(dirname, exist_ok=True)
    save_name = osp.join(dirname, 'epoch_' + str(epoch) + '.png')
    plt.savefig(save_name, bbox_inches='tight')
    plt.close()


def generate_gif(gif_name):
    outfile_name = osp.join(args.save_dir, gif_name)
    gif_images = []
    for i in range(1, args.total_epochs+1):
        image_path = 'epoch_{}.png'.format(i)
        image_path = osp.join(args.save_dir, args.split, image_path)
        if osp.exists(image_path):
            gif_images.append(imageio.imread(image_path))
        else:
            print(f"Image {image_path} not exists. Ignored.")
    imageio.mimsave(outfile_name, gif_images, fps=5)
    print("Done!")


if __name__ == '__main__':
    # if args.mode == 'sequential':
    #     main()
    # elif args.mode == 'single':
    #     main_single_epoch()
    # else:
    #     raise ValueError('Wrong mode!')
    generate_gif(gif_name='result.gif')