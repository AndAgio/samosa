import os
import argparse
import time
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from tqdm import tqdm
import datasets
from utils import AverageMeter, get_splits, lab_conv, enable_running_stats, disable_running_stats
import matplotlib.pyplot as plt

from openTSNE import TSNE as openTSNE
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

from nets.resnet import *
from mqnet_utils import str_to_bool

parser = argparse.ArgumentParser("samisuk")
# dataset
parser.add_argument('-d', '--dataset', type=str, default='cifar100', choices=['cifar100', 'cifar10', 'tinyimagenet'])
parser.add_argument('-j', '--workers', default=0, type=int,
                    help="number of data loading workers (default: 0)")
# optimization
parser.add_argument('--batch-size', type=int, default=128)
parser.add_argument('--lr-model', type=float, default=0.01, help="learning rate for model")
parser.add_argument('--max-epoch', type=int, default=300)
parser.add_argument('--max-query', type=int, default=11)
parser.add_argument('--query-batch', type=int, default=1500)
parser.add_argument('--stepsize', type=int, default=60)
parser.add_argument('--gamma', type=float, default=0.5, help="learning rate decay")
parser.add_argument('--query-strategy', type=str, default='eoal', choices=['random', 'eoal', 'lfosa', 'samisuk_l', 'samisuk_h', 'samisuk_dynamic'])

# model
parser.add_argument('--model', type=str, default='resnet18')
# misc
parser.add_argument('--eval-freq', type=int, default=300)
parser.add_argument('--gpu', type=str, default='0')
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--use-cpu', action='store_true')
# openset
parser.add_argument('--is-filter', type=bool, default=True)
parser.add_argument('--is-mini', type=bool, default=True)
parser.add_argument('--known-class', type=int, default=20)
parser.add_argument('--known-T', type=float, default=0.5)
parser.add_argument('--unknown-T', type=float, default=0.5)
parser.add_argument('--modelB-T', type=float, default=1)
parser.add_argument('--init-percent', type=int, default=16)
parser.add_argument('--diversity', type=int, default=1)
parser.add_argument('--pareta-alpha', type=float, default=0.8)
parser.add_argument('--reg-w', type=float, default=0.1)
parser.add_argument('--w-unk-cls', type=int, default=1)
parser.add_argument('--w-ent', type=float, default=1)
parser.add_argument('--continue-round', type=int, default=-1)
parser.add_argument("--resume", default=False, type=str_to_bool, help="whether parallel or not")

parser.add_argument('--save-dir', type=str, default='images')

parser.add_argument('--mode', type=str, default='pca')

args = parser.parse_args()



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


def plot_embeddings_same_round_model():
    seed = args.seed
    all_accuracies = []
    all_precisions = []
    all_recalls = []
    knownclass = get_splits(args.dataset, seed, args.known_class)
    num_classes = args.known_class 
    print("Known Classes", knownclass)
    torch.manual_seed(args.seed)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    use_gpu = torch.cuda.is_available()
    if args.use_cpu: use_gpu = False

    if use_gpu:
        print("Currently using GPU: {}".format(args.gpu))
        device_name = torch.cuda.get_device_name()
        print("Device name:", device_name)
        cudnn.benchmark = True
        torch.cuda.manual_seed_all(args.seed)
    else:
        print("Currently using CPU")

    print("Creating dataset: {}".format(args.dataset))
    dataset = datasets.create(
        name=args.dataset, known_class_=args.known_class, knownclass=knownclass, init_percent_=args.init_percent,
        batch_size=args.batch_size, use_gpu=use_gpu,
        num_workers=args.workers, is_filter=args.is_filter, is_mini=args.is_mini,
    )

    testloader, unlabeledloader = dataset.testloader, dataset.unlabeledloader
    trainloader_C = dataset.trainloader
    #trainloader_C = None   # init negativeloader none
    invalidList = []
    labeled_ind_train, unlabeled_ind_train = dataset.labeled_ind_train, dataset.unlabeled_ind_train
    start = 0    
    run = 0
    print("Creating model: {}".format(args.model))
    start_time = time.time()

    Acc = []
    Err = []
    Precision = []
    Recall = []
    last_round=0


    if args.dataset == 'tinyimagenet':
        im_size = (64,64)
    else:
        im_size = (32,32)

    cuda_available = torch.cuda.is_available()
    if cuda_available and not args.use_cpu:
        dev_str = 'cuda:{}'.format(args.gpu)
        device = torch.device(dev_str)
        print('Runner is setup using CUDA enabled device: {}'.format(torch.cuda.get_device_name(dev_str)))
    else:
        device = torch.device('cpu')
        print('Runner is setup using CPU! Training and inference will be very slow!')
    cudnn.benchmark = True  # Should make training go faster for large models

    for query in tqdm(range(last_round, args.max_query)):
        print("Query Round: {}".format(query))
        if query > 0:
            data = np.load("embeddings_indices/{}_kc_{}_{}_seed_{}_query_{}.npz".format(args.query_strategy, args.known_class, args.dataset, args.seed, query))
            #obtained queries - don't use for round 0
            queries = data['queries']
            queries = list(queries)

            data = np.load("inter2/{}_kc_{}_{}_seed_{}_query_{}.npz".format(args.query_strategy, args.known_class, args.dataset, args.seed, query-1)) 
            labeled_ind_train = data["labeled"]
            unlabeled_ind_train = data["unlabeled"] 
            labeled_ind_train = list(labeled_ind_train)
            unlabeled_ind_train = list(unlabeled_ind_train)
            
            #dataset the eval model was trained on           
            train_dataset = datasets.create(
            name=args.dataset, known_class_=args.known_class, knownclass=knownclass, init_percent_=args.init_percent,
            batch_size=args.batch_size, use_gpu=use_gpu,
            num_workers=args.workers, is_filter=args.is_filter, is_mini=args.is_mini,
            unlabeled_ind_train=unlabeled_ind_train, labeled_ind_train=labeled_ind_train,
            )
            trainloader = train_dataset.trainloader
            testloader = train_dataset.testloader

            #dataset with new queries
            new_query_dataset = datasets.create(
            name=args.dataset, known_class_=args.known_class, knownclass=knownclass, init_percent_=args.init_percent,
            batch_size=args.batch_size, use_gpu=use_gpu,
            num_workers=args.workers, is_filter=args.is_filter, is_mini=args.is_mini,
            unlabeled_ind_train=unlabeled_ind_train, labeled_ind_train=queries,
            )
            new_query_trainloader = new_query_dataset.trainloader
            new_query_testloader = new_query_dataset.testloader

            # Model initialization

            save_path = "./checkpoints/embeddings_models/"
            net = ResNet(arch = args.model,channel = 3, num_classes=dataset.num_classes, im_size=im_size)
            net.load_state_dict(torch.load(os.path.join(save_path, 'eval/{}_kc_{}_{}_seed_{}_query_{}.pth'.format(args.query_strategy, args.known_class, args.dataset, args.seed, query)), map_location=device))
            net = net.to(device)
            net.eval()

            with torch.no_grad():
                print('TRAINSET: Getting embeddings through model...')
                train_features = []
                train_labels = []
                start = time.time()
                len_train_features = 0
                # for i, (inputs, targets) in enumerate(trainloader):
                for batch_idx, (index, (inputs, targets)) in enumerate(trainloader):
                    # if batch_idx > 20:
                    #     break
                    print('Batch {} out of {}...'.format(batch_idx, len(trainloader)), end='\r')
                    targets = lab_conv(knownclass, targets)
                    inputs, targets = inputs.to(device), targets.to(device)
                    _, features = net(inputs)
                    len_train_features += len(features)
                    train_features.append(features.cpu().detach().numpy())
                    train_labels.append(targets.cpu().detach().numpy())
                print()
                print('Time taken to get embeddings: {}'.format(format_time(time.time() - start)))

                del trainloader
                
                print('SAMPLED_SET: Getting embeddings through model...')
                sampledset_features = []
                sampledset_labels = []
                start = time.time()
                # for i, (inputs, targets) in enumerate(new_query_trainloader):
                for batch_idx, (index, (inputs, targets)) in enumerate(new_query_trainloader):
                    print('Batch {} out of {}...'.format(batch_idx, len(new_query_trainloader)), end='\r')
                    targets = lab_conv(knownclass, targets)
                    inputs, targets = inputs.to(device), targets.to(device)
                    _, features = net(inputs)
                    sampledset_features.append(features.cpu().detach().numpy())
                    sampledset_labels.append(targets.cpu().detach().numpy())
                print()
                print('Time taken to get embeddings: {}'.format(format_time(time.time() - start)))

                del new_query_trainloader
                del net

                mode = args.mode
                if mode == 'tsne':
                    compressor = openTSNE(
                        perplexity=10,
                        metric="euclidean",
                        n_jobs=8,
                        random_state=42,
                        verbose=False,
                    )
                elif mode == 'pca':
                    compressor = PCA(n_components=2)
                else:
                    raise ValueError('Invalid mode!')

                max_embeddings_trainset = 5000

                print('TRAINSET: Applying {} to embeddings...'.format(mode.upper()))
                # fea, label = embedding(train_features, train_labels)
                feature = train_features[0]
                label = train_labels[0]
                for i in range(1, len(train_labels)):
                    # feature = torch.cat([feature,train_features[i]],dim=0)
                    feature = np.concatenate([feature, train_features[i]],0)
                    # label = torch.cat([label, train_labels[i]], dim=0)
                    label = np.concatenate([label, train_labels[i]],0)
                    if len(label) > max_embeddings_trainset:
                        break

                print('Number of training set samples: {}'.format(len(label)))

                # # feature = feature.cpu().detach().numpy()[:max_embeddings_trainset]
                # feature = feature[:max_embeddings_trainset]
                # # label = label.cpu().detach().numpy()[:max_embeddings_trainset]
                # label = label[:max_embeddings_trainset]
                # Using PCA or TSNE to reduce dimension to a reasonable dimension as recommended in
                fit_compressor = compressor.fit(feature)
                transformed_train_features = fit_compressor.transform(feature)
                print('TRAINSET: Plotting {} results...'.format(mode.upper()))
                # plot_features(fea, label, 10, epoch, args.split)

                fig = plt.figure()
                ax = plt.subplot(111)

                shapes = ['o', 'v', '^', '<', '>', 's', 'P', 'p', '*', '+']
                for label_idx in range(num_classes):
                    ax.scatter(
                        transformed_train_features[label == label_idx, 0],
                        transformed_train_features[label == label_idx, 1],
                        c='green',
                        marker=shapes[label_idx],
                        s=40,
                    )
                    # plt.clim(0, 1)

                del feature
                del label

                max_embeddings_sampledset = 500

                print('SAMPLED_SET: Applying {} to embeddings...'.format(mode.upper()))
                # fea, label = embedding(train_features, train_labels)
                feature = sampledset_features[0]
                label = sampledset_labels[0]
                for i in range(1, len(sampledset_labels)):
                    # feature = torch.cat([feature, sampledset_features[i]],dim=0)
                    feature = np.concatenate([feature, sampledset_features[i]], 0)
                    # label = torch.cat([label, sampledset_labels[i]], dim=0)
                    label = np.concatenate([label, sampledset_labels[i]], 0)
                    if len(label) > max_embeddings_sampledset:
                        break

                print('Number of sampled set samples: {}'.format(len(label)))

                # feature = feature.cpu().detach().numpy()[:max_embeddings_sampledset]
                # label = label.cpu().detach().numpy()[:max_embeddings_sampledset]
                # Using PCA or TSNE to reduce dimension to a reasonable dimension as recommended in
                transformed_sampledset_features = fit_compressor.transform(feature)
                print('SAMPLED_SET: Plotting {} results...'.format(mode.upper()))

                shapes = ['o', 'v', '^', '<', '>', 's', 'P', 'p', '*', '+']
                for label_idx in range(num_classes):
                    ax.scatter(
                        transformed_sampledset_features[label == label_idx, 0],
                        transformed_sampledset_features[label == label_idx, 1],
                        c='red',
                        marker=shapes[label_idx],
                        s=40,
                    )


                box = ax.get_position()
                ax.set_position([box.x0, box.y0 + box.height * 0.1,
                                box.width, box.height * 0.9])
                ax.axes.xaxis.set_ticklabels([])
                ax.axes.yaxis.set_ticklabels([])

                # Put a legend below current axis
                original_classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
                knownclass_labels = [original_classes[k] for k in knownclass]
                ax.legend(knownclass_labels, loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=len(knownclass_labels))
                # plt.legend(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'], loc='upper right')
                # plt.colorbar().set_label('SAMIS', rotation=270)

                plt.xticks([], [])
                plt.yticks([], [])

                plt.title('Round: {}'.format(query))

                dirname = os.path.join(args.save_dir, 'tsne_embeddings/same_round/{}/{}_{}_{}_{}'.format(mode, args.query_strategy, args.known_class, args.dataset, args.seed))
                os.makedirs(dirname, exist_ok=True)
                save_name = os.path.join(dirname, 'query_' + str(query) + '.png')
                plt.savefig(save_name, bbox_inches='tight')
                plt.close()



def plot_embeddings_next_round_model():
    seed = args.seed
    all_accuracies = []
    all_precisions = []
    all_recalls = []
    knownclass = get_splits(args.dataset, seed, args.known_class)
    num_classes = args.known_class 
    print("Known Classes", knownclass)
    torch.manual_seed(args.seed)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    use_gpu = torch.cuda.is_available()
    if args.use_cpu: use_gpu = False

    if use_gpu:
        print("Currently using GPU: {}".format(args.gpu))
        device_name = torch.cuda.get_device_name()
        print("Device name:", device_name)
        cudnn.benchmark = True
        torch.cuda.manual_seed_all(args.seed)
    else:
        print("Currently using CPU")

    print("Creating dataset: {}".format(args.dataset))
    dataset = datasets.create(
        name=args.dataset, known_class_=args.known_class, knownclass=knownclass, init_percent_=args.init_percent,
        batch_size=args.batch_size, use_gpu=use_gpu,
        num_workers=args.workers, is_filter=args.is_filter, is_mini=args.is_mini,
    )

    testloader, unlabeledloader = dataset.testloader, dataset.unlabeledloader
    trainloader_C = dataset.trainloader
    #trainloader_C = None   # init negativeloader none
    invalidList = []
    labeled_ind_train, unlabeled_ind_train = dataset.labeled_ind_train, dataset.unlabeled_ind_train
    start = 0    
    run = 0
    print("Creating model: {}".format(args.model))
    start_time = time.time()

    Acc = []
    Err = []
    Precision = []
    Recall = []
    last_round=0


    if args.dataset == 'tinyimagenet':
        im_size = (64,64)
    else:
        im_size = (32,32)

    cuda_available = torch.cuda.is_available()
    if cuda_available and not args.use_cpu:
        dev_str = 'cuda:{}'.format(args.gpu)
        device = torch.device(dev_str)
        print('Runner is setup using CUDA enabled device: {}'.format(torch.cuda.get_device_name(dev_str)))
    else:
        device = torch.device('cpu')
        print('Runner is setup using CPU! Training and inference will be very slow!')
    cudnn.benchmark = True  # Should make training go faster for large models

    for query in tqdm(range(last_round, args.max_query)):
        print("Query Round: {}".format(query))
        if query > 0:
            data = np.load("embeddings_indices/{}_kc_{}_{}_seed_{}_query_{}.npz".format(args.query_strategy, args.known_class, args.dataset, args.seed, query-1))
            #obtained queries - don't use for round 0
            queries = data['queries']
            queries = list(queries)

            data = np.load("inter2/{}_kc_{}_{}_seed_{}_query_{}.npz".format(args.query_strategy, args.known_class, args.dataset, args.seed, query-1)) 
            labeled_ind_train = data["labeled"]
            unlabeled_ind_train = data["unlabeled"] 
            labeled_ind_train = list(labeled_ind_train)
            unlabeled_ind_train = list(unlabeled_ind_train)
            
            #dataset the eval model was trained on           
            train_dataset = datasets.create(
            name=args.dataset, known_class_=args.known_class, knownclass=knownclass, init_percent_=args.init_percent,
            batch_size=args.batch_size, use_gpu=use_gpu,
            num_workers=args.workers, is_filter=args.is_filter, is_mini=args.is_mini,
            unlabeled_ind_train=unlabeled_ind_train, labeled_ind_train=labeled_ind_train,
            )
            trainloader = train_dataset.trainloader
            testloader = train_dataset.testloader

            #dataset with new queries
            new_query_dataset = datasets.create(
            name=args.dataset, known_class_=args.known_class, knownclass=knownclass, init_percent_=args.init_percent,
            batch_size=args.batch_size, use_gpu=use_gpu,
            num_workers=args.workers, is_filter=args.is_filter, is_mini=args.is_mini,
            unlabeled_ind_train=unlabeled_ind_train, labeled_ind_train=queries,
            )
            new_query_trainloader = new_query_dataset.trainloader
            new_query_testloader = new_query_dataset.testloader

            # Model initialization

            save_path = "./checkpoints/embeddings_models/"
            net = ResNet(arch = args.model,channel = 3, num_classes=dataset.num_classes, im_size=im_size)
            net.load_state_dict(torch.load(os.path.join(save_path, 'eval/{}_kc_{}_{}_seed_{}_query_{}.pth'.format(args.query_strategy, args.known_class, args.dataset, args.seed, query)), map_location=device))
            net = net.to(device)
            net.eval()

            with torch.no_grad():
                print('TRAINSET: Getting embeddings through model...')
                train_features = []
                train_labels = []
                start = time.time()
                len_train_features = 0
                # for i, (inputs, targets) in enumerate(trainloader):
                for batch_idx, (index, (inputs, targets)) in enumerate(trainloader):
                    # if batch_idx > 20:
                    #     break
                    print('Batch {} out of {}...'.format(batch_idx, len(trainloader)), end='\r')
                    targets = lab_conv(knownclass, targets)
                    inputs, targets = inputs.to(device), targets.to(device)
                    _, features = net(inputs)
                    len_train_features += len(features)
                    train_features.append(features.cpu().detach().numpy())
                    train_labels.append(targets.cpu().detach().numpy())
                print()
                print('Time taken to get embeddings: {}'.format(format_time(time.time() - start)))

                del trainloader
                
                print('SAMPLED_SET: Getting embeddings through model...')
                sampledset_features = []
                sampledset_labels = []
                start = time.time()
                # for i, (inputs, targets) in enumerate(new_query_trainloader):
                for batch_idx, (index, (inputs, targets)) in enumerate(new_query_trainloader):
                    # if batch_idx > 20:
                    #     break
                    print('Batch {} out of {}...'.format(batch_idx, len(new_query_trainloader)), end='\r')
                    targets = lab_conv(knownclass, targets)
                    inputs, targets = inputs.to(device), targets.to(device)
                    _, features = net(inputs)
                    sampledset_features.append(features.cpu().detach().numpy())
                    sampledset_labels.append(targets.cpu().detach().numpy())
                print()
                print('Time taken to get embeddings: {}'.format(format_time(time.time() - start)))

                del new_query_trainloader
                del net

                mode = args.mode
                if mode == 'tsne':
                    compressor = openTSNE(
                        perplexity=10,
                        metric="euclidean",
                        n_jobs=8,
                        random_state=42,
                        verbose=False,
                    )
                elif mode == 'pca':
                    compressor = PCA(n_components=2)
                else:
                    raise ValueError('Invalid mode!')

                max_embeddings_trainset = 5000

                print('TRAINSET: Applying {} to embeddings...'.format(mode.upper()))
                # fea, label = embedding(train_features, train_labels)
                feature = train_features[0]
                label = train_labels[0]
                for i in range(1, len(train_labels)):
                    # feature = torch.cat([feature,train_features[i]],dim=0)
                    feature = np.concatenate([feature, train_features[i]],0)
                    # label = torch.cat([label, train_labels[i]], dim=0)
                    label = np.concatenate([label, train_labels[i]],0)
                    if len(label) > max_embeddings_trainset:
                        break

                print('Number of training set samples: {}'.format(len(label)))

                # # feature = feature.cpu().detach().numpy()[:max_embeddings_trainset]
                # feature = feature[:max_embeddings_trainset]
                # # label = label.cpu().detach().numpy()[:max_embeddings_trainset]
                # label = label[:max_embeddings_trainset]
                # Using PCA or TSNE to reduce dimension to a reasonable dimension as recommended in
                fit_compressor = compressor.fit(feature)
                transformed_train_features = fit_compressor.transform(feature)
                print('TRAINSET: Plotting {} results...'.format(mode.upper()))
                # plot_features(fea, label, 10, epoch, args.split)

                fig = plt.figure()
                ax = plt.subplot(111)

                shapes = ['o', 'v', '^', '<', '>', 's', 'P', 'p', '*', '+']
                for label_idx in range(num_classes):
                    ax.scatter(
                        transformed_train_features[label == label_idx, 0],
                        transformed_train_features[label == label_idx, 1],
                        c='green',
                        marker=shapes[label_idx],
                        s=40,
                    )
                    # plt.clim(0, 1)

                del feature
                del label

                max_embeddings_sampledset = 500

                print('SAMPLED_SET: Applying {} to embeddings...'.format(mode.upper()))
                # fea, label = embedding(train_features, train_labels)
                feature = sampledset_features[0]
                label = sampledset_labels[0]
                for i in range(1, len(sampledset_labels)):
                    # feature = torch.cat([feature, sampledset_features[i]],dim=0)
                    feature = np.concatenate([feature, sampledset_features[i]], 0)
                    # label = torch.cat([label, sampledset_labels[i]], dim=0)
                    label = np.concatenate([label, sampledset_labels[i]], 0)
                    if len(label) > max_embeddings_sampledset:
                        break

                print('Number of sampled set samples: {}'.format(len(label)))

                # feature = feature.cpu().detach().numpy()[:max_embeddings_sampledset]
                # label = label.cpu().detach().numpy()[:max_embeddings_sampledset]
                # Using PCA or TSNE to reduce dimension to a reasonable dimension as recommended in
                transformed_sampledset_features = fit_compressor.transform(feature)
                print('SAMPLED_SET: Plotting {} results...'.format(mode.upper()))

                shapes = ['o', 'v', '^', '<', '>', 's', 'P', 'p', '*', '+']
                for label_idx in range(num_classes):
                    ax.scatter(
                        transformed_sampledset_features[label == label_idx, 0],
                        transformed_sampledset_features[label == label_idx, 1],
                        c='red',
                        marker=shapes[label_idx],
                        s=40,
                    )


                box = ax.get_position()
                ax.set_position([box.x0, box.y0 + box.height * 0.1,
                                box.width, box.height * 0.9])
                ax.axes.xaxis.set_ticklabels([])
                ax.axes.yaxis.set_ticklabels([])

                # Put a legend below current axis
                original_classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
                knownclass_labels = [original_classes[k] for k in knownclass]
                ax.legend(knownclass_labels, loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=len(knownclass_labels))
                # plt.legend(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'], loc='upper right')
                # plt.colorbar().set_label('SAMIS', rotation=270)

                plt.xticks([], [])
                plt.yticks([], [])

                plt.title('Round: {}'.format(query))

                dirname = os.path.join(args.save_dir, 'tsne_embeddings/next_round/{}/{}_{}_{}_{}'.format(mode, args.query_strategy, args.known_class, args.dataset, args.seed))
                os.makedirs(dirname, exist_ok=True)
                save_name = os.path.join(dirname, 'query_' + str(query) + '.png')
                plt.savefig(save_name, bbox_inches='tight')
                plt.close()




def visualize_queried_samples():
    seed = args.seed
    knownclass = get_splits(args.dataset, seed, args.known_class)
    label_to_be_visualized = knownclass[0]
    original_classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    print('Images to visualize are of {}'.format(original_classes[label_to_be_visualized]))
    num_classes = args.known_class 
    print("Known Classes", knownclass)
    torch.manual_seed(args.seed)
    
    use_gpu = False

    last_round=0


    if args.dataset == 'tinyimagenet':
        im_size = (64,64)
    else:
        im_size = (32,32)

    for query in tqdm(range(last_round, args.max_query)):
        print("Query Round: {}".format(query))
        if query > 0:
            data = np.load("embeddings_indices/{}_kc_{}_{}_seed_{}_query_{}.npz".format(args.query_strategy, args.known_class, args.dataset, args.seed, query-1))
            #obtained queries - don't use for round 0
            queries = data['queries']
            queries = list(queries)

            #dataset with new queries
            new_query_dataset = datasets.create(
            name=args.dataset, known_class_=args.known_class, knownclass=knownclass, init_percent_=args.init_percent,
            batch_size=args.batch_size, use_gpu=use_gpu,
            num_workers=args.workers, is_filter=args.is_filter, is_mini=args.is_mini,
            unlabeled_ind_train=queries, labeled_ind_train=queries,
            )
            new_query_trainloader = new_query_dataset.trainloader

            inputs_to_be_visualized = []
            for batch_idx, (index, (inputs, targets)) in enumerate(new_query_trainloader):
                # targets = lab_conv(knownclass, targets)
                # print(targets)
                # print(label_to_be_visualized)
                inputs = inputs[targets == label_to_be_visualized]
                inputs_to_be_visualized.append(inputs)

            images = inputs_to_be_visualized[0]
            for i in range(1, len(inputs_to_be_visualized)):
                images = torch.cat([images,inputs_to_be_visualized[i]],dim=0)
            images = images.cpu().detach().numpy()

            print(len(images))

            fig = plt.figure(0, figsize=(8, 6))
            fig.suptitle("Queried {} samples at round {}".format(original_classes[label_to_be_visualized], query), fontsize=16)
            # fig.set_size_inches(18.5, 18.5)
            for i in range(0,32):
                fig.add_subplot(4, 8, i+1)
                plt.imshow(images[i].reshape(3,im_size[0],im_size[1]).transpose(1,2,0))
                plt.axis('off')
                # plt.title("Label: {}".format(test_labels[i]))


            dirname = os.path.join(args.save_dir, 'queried_visualization/{}_{}_{}_{}/{}'.format(args.query_strategy, args.known_class, args.dataset, args.seed, original_classes[label_to_be_visualized]))
            os.makedirs(dirname, exist_ok=True)
            save_name = os.path.join(dirname, 'query_' + str(query) + '.png')
            plt.tight_layout()
            plt.savefig(save_name, bbox_inches='tight')
            plt.close()

            # for to_be_visualized in inputs_to_be_visualized:
            #     print(to_be_visualized)



#Train SGD
def train_A(model, criterion_xent, optimizer_model, trainloader, use_gpu, knownclass, epoch):
    model.train()
    xent_losses = AverageMeter()
    losses = AverageMeter()

    for batch_idx, (index, (data, labels)) in enumerate(trainloader):
        labels = lab_conv(knownclass, labels)
        if use_gpu:
            data, labels = data.cuda(), labels.cuda()
        outputs, _ = model(data)
        loss_xent = criterion_xent(outputs, labels)
        loss = loss_xent 
        optimizer_model.zero_grad()
        loss.backward()
        optimizer_model.step()
    
        losses.update(loss.item(), labels.size(0))
        xent_losses.update(loss_xent.item(), labels.size(0))

    if epoch%50 == 0:
        print(f" loss: {losses.avg} xent_loss: {xent_losses.avg}")

#Train SAM
def train_B(model, criterion_xent, optimizer_model, trainloader, use_gpu, knownclass, epoch):
    model.train()
    xent_losses = AverageMeter()
    losses = AverageMeter()

    for batch_idx, (index, (data, labels)) in enumerate(trainloader):
        labels = lab_conv(knownclass, labels)
        if use_gpu:
            data, labels = data.cuda(), labels.cuda()
        enable_running_stats(model)
        outputs, _ = model(data)
        loss_xent = criterion_xent(outputs, labels)
        loss = loss_xent 
        loss.backward()
        optimizer_model.first_step(zero_grad=True)             
        # second forward-backward step
        disable_running_stats(model)
        criterion_xent(model(data)[0], labels).backward()

        optimizer_model.second_step(zero_grad=True) 
    
        losses.update(loss.item(), labels.size(0))
        xent_losses.update(loss_xent.item(), labels.size(0))

    if epoch%50 == 0:
        print(f" loss: {losses.avg} xent_loss: {xent_losses.avg}")
        
def test(model, testloader, use_gpu, knownclass):
    model.eval()
    correct, total = 0, 0
    
    with torch.no_grad():
        for index, (data, labels) in testloader:
            labels = lab_conv(knownclass, labels)
            if use_gpu:
                data, labels = data.cuda(), labels.cuda()
            outputs, _ = model(data)
            predictions = outputs.data.max(1)[1]
            total += labels.size(0)
            correct += (predictions == labels.data).sum()

    acc = correct * 100. / total
    err = 100. - acc
    return acc, err

if __name__ == '__main__':
    plot_embeddings_same_round_model()
    # plot_embeddings_next_round_model()
    # visualize_queried_samples()




