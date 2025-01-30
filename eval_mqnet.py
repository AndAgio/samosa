import argparse
import copy
import datetime
from utils import AverageMeter as AM
from utils import get_splits

# Python
import os
import time
import random

# Torch
import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torch.optim import lr_scheduler

# Utils
from mqnet_utils import *

import nets
from resnet import ResNet18, ResClassifier_MME

import methods as methods


import torch.backends.cudnn as cudnn
import datasets
import glob

parser = argparse.ArgumentParser("samisuk")
# dataset
parser.add_argument('-d', '--dataset', type=str, default='cifar100', choices=['cifar100', 'cifar10', 'tinyimagenet'])
parser.add_argument('-j', '--workers', default=0, type=int,
                    help="number of data loading workers (default: 0)")
# optimization
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--test_batch_size', type=int, default=128)
parser.add_argument('--lr-model', type=float, default=0.01, help="learning rate for model")
parser.add_argument('--max-epoch', type=int, default=300)
parser.add_argument('--max-query', type=int, default=11)
parser.add_argument('--query-batch', type=int, default=1500)
parser.add_argument('--stepsize', type=int, default=60)
# parser.add_argument('--gamma', type=float, default=0.5, help="learning rate decay")
parser.add_argument('--query-strategy', type=str, default='eoal', choices=['MQNet', 'Uncertainty', 'Coreset', 'LL', 'BADGE', 'CCAL', 'SIMILAR']) # Uncertainty, Coreset, LL, BADGE, CCAL, MQNet


# Optimizer and scheduler
parser.add_argument('--optimizer', default="SGD", help='optimizer to use, e.g. SGD, Adam')
parser.add_argument('--lr', type=float, default=0.01, help='learning rate for updating network parameters')
parser.add_argument('--lr-mqnet', type=float, default=0.001, help='learning rate for updating mqnet')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum (default: 0.9)')
parser.add_argument('-wd', '--weight_decay', default=5e-04, type=float, metavar='W', help='weight decay (default: 5e-4)', dest='weight_decay')
parser.add_argument("--scheduler", default="StepLR", type=str, help="Learning rate scheduler") #CosineAnnealingLR, StepLR, MultiStepLR
parser.add_argument('--min_lr', type=float, default=1e-4, help='minimum learning rate for CosineAnnealingLR')
parser.add_argument("--gamma", type=float, default=.5, help="Gamma value for StepLR")
parser.add_argument("--step_size", type=float, default=60, help="Step size for StepLR")
parser.add_argument('--milestone', type=list, default=[100, 150], metavar='M', help='Milestone for MultiStepLR')
parser.add_argument('--warmup', type=int, default=10, metavar='warmup', help='warmup epochs')

# model
parser.add_argument('--model', type=str, default='ResNet18')
# misc
parser.add_argument('--eval-freq', type=int, default=300)
parser.add_argument('--gpu', default=None, nargs="+", type=int, help='GPU id to use')
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

parser.add_argument('--ood-rate', type=float, default=0.6, metavar='N', help='OOD rate in unlabeled set')
# parser.add_argument('--n-query', type=int, default=1000, help='# of query samples')
parser.add_argument('--subset', type=int, default=50000, help='subset')
parser.add_argument('--ccal-batch-size', default=32, type=int, metavar='N')
parser.add_argument('--csi-batch-size', default=32, type=int, metavar='N')

# AL Algorithm
parser.add_argument('--submodular', default="logdetcmi", help="specifiy submodular function to use") #flcmi, logdetcmi
parser.add_argument('--submodular_greedy', default="LazyGreedy", help="specifiy greedy algorithm for submodular optimization")
parser.add_argument('--uncertainty', default="CONF", help="specifiy uncertanty score to use") #CONF, Margin, Entropy
# for CCAL
parser.add_argument("--resize_factor", help='resize scale is sampled from [resize_factor, 1.0]',default=0.08, type=float)
parser.add_argument("--resize_fix", help='resize scale is fixed to resize_factor (not (resize_factor, 1.0])',action='store_true')
parser.add_argument('--sim_lambda', help='Weight for SimCLR loss',default=1.0, type=float)
parser.add_argument('--shift_trans_type', help='shifting transformation type', default='rotation',choices=['rotation', 'cutperm', 'none'], type=str)
parser.add_argument("--ood_samples", help='number of samples to compute OOD score',default=1, type=int)
parser.add_argument('--k', help='Initial learning rate', default=100.0, type=float)
parser.add_argument('--t', help='Initial learning rate', default=0.9, type=float)
# for MQNet
parser.add_argument('--mqnet-mode', default="CONF", help="specifiy the mode of MQNet to use") #CONF, LL

parser.add_argument('--epoch-loss', default=60, type=int, help='number of epochs for training loss module in LL')
parser.add_argument('--epochs-ccal', default=70, type=int, help='number of epochs for training contrastive coders in CCAL')
parser.add_argument('--epochs-csi', default=100, type=int, help='number of epochs for training CSI')
parser.add_argument('--epochs-mqnet', default=100, type=int, help='number of epochs for training mqnet')
parser.add_argument('--steps-per-epoch', type=int, default=100, metavar='N', help='number of steps per epoch')

parser.add_argument("--data-parallel", default=False, type=str_to_bool, help="whether parallel or not")
parser.add_argument("--ssl-save", default=True, type=str_to_bool, help="whether save ssl model or not")

parser.add_argument('--print_freq', '-p', default=100, type=int, help='print frequency (default: 20)')
args = parser.parse_args()


def main():
    os.makedirs('new_fair_intermediaries', exist_ok=True)
    os.makedirs('new_fair_results', exist_ok=True)
    os.makedirs('mqnet_models', exist_ok=True)
    seed = args.seed
    all_accuracies = []
    all_precisions = []
    all_recalls = []
    run_once = 0
    knownclass = get_splits(args.dataset, seed, args.known_class)  
    print("Known Classes", knownclass)
    torch.manual_seed(args.seed)
    if len(args.gpu) == 1:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu[0])
    use_gpu = torch.cuda.is_available()
    if args.use_cpu: use_gpu = False

    if use_gpu:
        print("Currently using GPU: {}".format(args.gpu))
        cudnn.benchmark = True
        torch.cuda.manual_seed_all(args.seed)
    else:
        print("Currently using CPU")

    args.gpu = torch.device(args.gpu[0])
    if args.query_strategy == 'Uncertainty':
        args.query_strategy = '{}_{}'.format(args.query_strategy, args.uncertainty)
    print("Creating dataset: {}".format(args.dataset))
    dataset = datasets.create(
        name=args.dataset, known_class_=args.known_class, knownclass=knownclass, init_percent_=args.init_percent,
        batch_size=args.batch_size, use_gpu=use_gpu,
        num_workers=args.workers, is_filter=args.is_filter, is_mini=args.is_mini,
    )

    testloader_eval, unlabeledloader = dataset.testloader, dataset.unlabeledloader
    trainloader_eval, trainloader_B, trainloader_C = dataset.trainloader, dataset.trainloader, dataset.trainloader
    #trainloader_C = None   # init negativeloader none
    invalidList = []
    labeled_ind_train, unlabeled_ind_train = dataset.labeled_ind_train, dataset.unlabeled_ind_train
    start = 0

    # train_dst = torch.utils.data.Subset(dataset, labeled_ind_train)
    train_dst = dataset.trainset
    # unlabeled_dst = torch.utils.data.Subset(dataset, unlabeled_ind_train)
    unlabeled_dst = dataset.trainset
    # test_dst = torch.utils.data.Subset(dataset, dataset.filter_ind_test)
    test_dst = dataset.testset
    # filter_ind_test
    # trainset_2 = torch.utils.data.Subset(trainset, odds)
    # train_dst = dataset.trainset
    # test_dst = dataset.testset
    # unlabeled_dst = B_dataset.trainset

    # Initialize a labeled dataset by randomly sampling K=1,000 points from the entire dataset.
    # I_index, O_index, U_index, Q_index = [], [], [], []
    # prev_I_index, prev_O_index, prev_U_index, prev_Q_index = [], [], [], []
    # I_index, O_index, U_index = get_sub_train_dataset(args, train_dst, I_index, O_index, U_index, Q_index, initial=True)
    # test_I_index = get_sub_test_dataset(args, test_dst)


    if args.continue_round > -1:
        start = args.continue_round

    Acc = []
    Err = []
    Precision = []
    Recall = []

    # Active learning

    for query in tqdm(range(start, args.max_query)):
        print("Query Round: {}".format(query))
        if query > 0:
            data = np.load("intermediaries/{}_kc_{}_{}_seed_{}_query_{}.npz".format(args.query_strategy, args.known_class, args.dataset, args.seed, query-1))
            if args.continue_round > -1 and run == 0:
                prev = np.load('new_fair_intermediaries/{}_kc_{}_{}_seed_{}_{}.npz'.format(args.query_strategy, args.known_class, args.dataset, args.seed, query-1))
                Acc = prev['acc']
                Acc = list(Acc)
                run = run + 1
            Precision = data["precision"]
            Recall = data["recall"]
            labeled_ind_train = data["labeled"]
            unlabeled_ind_train = data["unlabeled"]
            invalidList = data["invalidList"]

            Precision = list(Precision)
            Recall = list(Recall)
            unlabeled_ind_train = list(unlabeled_ind_train)
            labeled_ind_train = list(labeled_ind_train)
            invalidList = list(invalidList)

            dataset = datasets.create(
            name=args.dataset, known_class_=args.known_class, knownclass=knownclass, init_percent_=args.init_percent,
            batch_size=args.batch_size, use_gpu=use_gpu,
            num_workers=args.workers, is_filter=args.is_filter, is_mini=args.is_mini,
            unlabeled_ind_train=list(set(unlabeled_ind_train) - set(invalidList)), labeled_ind_train=labeled_ind_train,
            )
            trainloader_eval = dataset.trainloader
            testloader_eval = dataset.testloader

        model_eval = ResNet18(n_class = dataset.num_classes)
        model_eval = model_eval.cuda()
        optimizer_model_eval = torch.optim.SGD(model_eval.parameters(), lr=args.lr_model, weight_decay=5e-04, momentum=0.9)
        scheduler_eval = lr_scheduler.StepLR(optimizer_model_eval, step_size=args.stepsize, gamma=args.gamma)
        criterion_xent = nn.CrossEntropyLoss()
        criterion_xent.cuda()

        t = time.time()
        
        # Test
        for epoch in tqdm(range(args.max_epoch)):    
            train_SGD(model_eval, criterion_xent,
                optimizer_model_eval,
                trainloader_eval, knownclass)
    
            if args.stepsize > 0:
                scheduler_eval.step()
    
        # Record results
        acc, err = test(model_eval, testloader_eval, knownclass)
        print("Accuracy (%): {}\t Error rate (%): {}".format(acc, err))
                
        Acc.append(float(acc))
        Err.append(float(err))


        
        # Save in case of interruption
        np.savez('new_fair_intermediaries/{}_kc_{}_{}_seed_{}_query_{}.npz'.format(args.query_strategy, args.known_class, args.dataset, args.seed, query), unlabeled=unlabeled_ind_train, labeled=labeled_ind_train, invalidList=invalidList, acc=Acc, precision=Precision, recall=Recall)



    all_accuracies.append(Acc)
    all_precisions.append(Precision)
    all_recalls.append(Recall)
    print("Accuracies", all_accuracies)
    print("Precisions", all_precisions)
    print("Recalls", all_recalls)
    np.savez('new_fair_results/{}_kc_{}_{}_seed_{}.npz'.format(args.query_strategy, args.known_class, args.dataset, args.seed), acc=Acc, precision=Precision, recall=Recall)   

def train_SGD(model, criterion_xent, optimizer_model, trainloader, knownclass):
    model.train()

    for batch_idx, (index, (data, labels)) in enumerate(trainloader):
        labels = lab_conv(knownclass, labels)
        data, labels = data.cuda(), labels.cuda()
        outputs, _ = model(data)
        loss_xent = criterion_xent(outputs, labels)
        loss = loss_xent 
        optimizer_model.zero_grad()
        loss.backward()
        optimizer_model.step()
    


def test(model, testloader, knownclass):
    model.eval()
    correct, total = 0, 0
    
    with torch.no_grad():
        for index, (data, labels) in testloader:
            labels = lab_conv(knownclass, labels)

            data, labels = data.cuda(), labels.cuda()
            outputs, _ = model(data)
            predictions = outputs.data.max(1)[1]
            total += labels.size(0)
            correct += (predictions == labels.data).sum()

    acc = correct * 100. / total
    err = 100. - acc
    return acc, err

if __name__ == '__main__':
    main()
