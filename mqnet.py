import argparse
import copy
import datetime
from utils import AverageMeter, get_splits

# Python
import os
import time
import random

# Torch
import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

# Utils
from mqnet_utils import *

import nets

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

parser.add_argument('--wait', default=0, type=int)

args = parser.parse_args()

print('Waiting for {} hours...'.format(args.wait))
time.sleep(args.wait*60*60)
print('Done waiting. Now executing')


def main():
    os.makedirs('intermediaries', exist_ok=True)
    os.makedirs('results', exist_ok=True)
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

    print("Creating dataset: {}".format(args.dataset))
    dataset = datasets.create(
        name=args.dataset, known_class_=args.known_class, knownclass=knownclass, init_percent_=args.init_percent,
        batch_size=args.batch_size, use_gpu=use_gpu,
        num_workers=args.workers, is_filter=args.is_filter, is_mini=args.is_mini,
    )

    testloader, unlabeledloader = dataset.testloader, dataset.unlabeledloader
    trainloader_A, trainloader_B, trainloader_C = dataset.trainloader, dataset.trainloader, dataset.trainloader
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

    I_index = labeled_ind_train
    O_index = invalidList
    U_index = unlabeled_ind_train
    Q_index = []
    prev_I_index = I_index
    prev_O_index = O_index 
    prev_U_index = U_index 
    prev_Q_index = Q_index
    test_I_index = dataset.filter_ind_test


    # DataLoaders
    if args.dataset in ['cifar10', 'cifar100', 'tinyimagenet']:
        sampler_labeled = SubsetRandomSampler(I_index)  # make indices initial to the samples
        sampler_test = SubsetSequentialSampler(test_I_index)
        train_loader = DataLoader(train_dst, sampler=sampler_labeled, batch_size=args.batch_size, num_workers=args.workers)
        test_loader = DataLoader(test_dst, sampler=sampler_test, batch_size=args.test_batch_size, num_workers=args.workers)
    dataloaders = {'train': train_loader, 'test': test_loader}

    print("Creating model: {}".format(args.model))
    start_time = time.time()

    Acc = []
    Err = []
    Precision = []
    Recall = []

    #Resuming from prev round
    if args.resume:
        found_files = glob.glob(os.path.join('intermediaries', '*.npz'))
        print('found_files: {}'.format(found_files))
        if args.query_strategy == 'Uncertainty':
            query_strategy = '{}_{}'.format(args.query_strategy, args.uncertainty)
        elif args.query_strategy == 'MQNet':
            query_strategy = '{}_{}'.format(args.query_strategy, args.mqnet_mode)
        else:
            query_strategy = args.query_strategy
        skeemed_files = [f for f in found_files if '{}_'.format(query_strategy) in f and 'kc_{}_'.format(args.known_class) in f and '_{}_'.format(args.dataset) in f and 'seed_{}_'.format(args.seed) in f]
        print('skeemed_files: {}'.format(skeemed_files))
        if len(skeemed_files) == 0:
            last_round = 0
        else:
            last_round = max([int(f.split('_')[-1].split('.')[0]) for f in skeemed_files])
    if args.resume and last_round > 0:
        start = last_round + 1
        if args.query_strategy == 'Uncertainty':
            query_strategy = '{}_{}'.format(args.query_strategy, args.uncertainty)
        elif args.query_strategy == 'MQNet':
            query_strategy = '{}_{}'.format(args.query_strategy, args.mqnet_mode)
        else:
            query_strategy = args.query_strategy
        data = np.load("intermediaries/{}_kc_{}_{}_seed_{}_query_{}.npz".format(query_strategy, args.known_class, args.dataset, args.seed, last_round))
        Acc = data["acc"]
        Err = data["err"]
        Precision = data["precision"]
        Recall = data["recall"]
        labeled_ind_train = data["labeled"]
        unlabeled_ind_train = data["unlabeled"]
        invalidList = data["invalidList"] 
        I_index = data["I_index"]
        O_index = data["O_index"]
        U_index = data["U_index"]
        Q_index = data["Q_index"]
        prev_I_index = data["prev_I_index"]
        prev_O_index = data["prev_O_index"]
        prev_U_index = data["prev_U_index"]
        prev_Q_index = data["prev_Q_index"]

        Acc = list(Acc)
        Precision = list(Precision)
        Recall = list(Recall)
        Err = list(Err)
        unlabeled_ind_train = list(unlabeled_ind_train)
        labeled_ind_train = list(labeled_ind_train)
        invalidList = list(invalidList)

        # DataLoaders
        if args.dataset in ['cifar10', 'cifar100', 'tinyimagenet']:
            sampler_labeled = SubsetRandomSampler(I_index)  # make indices initial to the samples
            sampler_test = SubsetSequentialSampler(test_I_index)
            train_loader = DataLoader(train_dst, sampler=sampler_labeled, batch_size=args.batch_size, num_workers=args.workers)
            test_loader = DataLoader(test_dst, sampler=sampler_test, batch_size=args.test_batch_size, num_workers=args.workers)
        dataloaders = {'train': train_loader, 'test': test_loader}
        

    # Active learning

    models = None
    for query in tqdm(range(start, args.max_query)):
        print("Query Round: {}".format(query))

        models = get_models(args, nets, args.model, models)
        torch.backends.cudnn.benchmark = False

        # Loss, criterion and scheduler (re)initialization
        criterion, optimizers, schedulers = get_optim_configurations(args, models)

        # Self-supervised Training (for CCAL and MQ-Net with CSI)
        if query == 0:
            models = self_sup_train(args, seed, models, optimizers, schedulers, train_dst, I_index, O_index, U_index)

        # Training
        t = time.time()
        train(args, models, criterion, optimizers, schedulers, dataloaders)
        print("query: {}, elapsed time: {}".format(query, (time.time() - t)))

        # Test
        acc = test(args, models, dataloaders)
        err = 100. - acc
        print("Accuracy (%): {}\t Error rate (%): {}".format(acc, err))
        Acc.append(float(acc))
        Err.append(float(err))

        print('Round {}/{} || Labeled IN size {}: Test acc {}'.format(
                query + 1, args.max_query, len(I_index), acc), flush=True)

        #### AL Query ####
        print("==========Start Querying==========")
        selection_args = dict(I_index=I_index,
                                O_index=O_index,
                                selection_method=args.uncertainty,
                                dataloaders=dataloaders,
                                cur_cycle=query)

        ALmethod = methods.__dict__[args.query_strategy](args, models, unlabeled_dst, U_index, **selection_args)
        Q_index, Q_scores = ALmethod.select()

        # Update Indices
        I_index, O_index, U_index, in_cnt = get_sub_train_dataset(args, train_dst, I_index, O_index, U_index, Q_index, initial=False)
        print("# Labeled_in: {}, # Labeled_ood: {}, # Unlabeled: {}".format(
            len(set(I_index)), len(set(O_index)), len(set(U_index))))
        
        queryIndex = list(set(I_index) - set(prev_I_index))
        Prec = len(set(I_index) - set(prev_I_index)) / (len(set(I_index) - set(prev_I_index)) + len(set(O_index) - set(prev_O_index)))
        if args.dataset=='cifar10':
            num_total_known = 5000 * args.known_class
        else:
            num_total_known = 500 * args.known_class

        Rec = len(I_index)/ num_total_known
        Precision.append(Prec)
        Recall.append(Rec)
        
        # Update labeled, unlabeled and invalid set
        unlabeled_ind_train = list(set(unlabeled_ind_train) - set(I_index))
        labeled_ind_train = list(set(labeled_ind_train)) + queryIndex
        invalidList = list(set(invalidList)) + list(set(O_index))

        prev_I_index = copy.copy(I_index)
        prev_O_index = copy.copy(O_index)
        prev_U_index = copy.copy(U_index)

        # Meta-training MQNet
        if args.query_strategy == 'MQNet':
            models, optimizers, schedulers = init_mqnet(args, nets, models, optimizers, schedulers)
            if args.resume and run_once == 0:
                model_path = 'mqnet_models/'+ str(args.dataset)+'_kc_'+str(args.known_class)+'_seed_' + str(args.seed) + '_query_' + str(last_round)+'.pt'
                models['mqnet'].load_state_dict(torch.load(model_path))
                run_once = 1
            unlabeled_loader = DataLoader(unlabeled_dst, sampler=SubsetRandomSampler(U_index), batch_size=args.test_batch_size, num_workers=args.workers)
            delta_loader = DataLoader(train_dst, sampler=SubsetRandomSampler(Q_index), batch_size=max(1, args.csi_batch_size), num_workers=args.workers)
            models = meta_train(args, models, optimizers, schedulers, criterion, dataloaders['train'], unlabeled_loader, delta_loader)
            model_path = 'mqnet_models/'+ str(args.dataset)+'_kc_'+str(args.known_class)+'_seed_' + str(args.seed) + '_query_' + str(query)+'.pt'
            torch.save(models['mqnet'].state_dict(), model_path)

        # Update trainloader
        if args.dataset in ['cifar10', 'cifar100', 'tinyimagenet']:
            sampler_labeled = SubsetRandomSampler(I_index)  # make indices initial to the samples
            dataloaders['train'] = DataLoader(train_dst, sampler=sampler_labeled, batch_size=args.batch_size, num_workers=args.workers)
        
        # Save in case of interruption
        if args.query_strategy == 'Uncertainty':
            query_strategy = '{}_{}'.format(args.query_strategy, args.uncertainty)
        elif args.query_strategy == 'MQNet':
            query_strategy = '{}_{}'.format(args.query_strategy, args.mqnet_mode)
        else:
            query_strategy = args.query_strategy
        np.savez('intermediaries/{}_kc_{}_{}_seed_{}_query_{}.npz'.format(query_strategy, args.known_class, args.dataset, args.seed, query), unlabeled=unlabeled_ind_train, labeled=labeled_ind_train, invalidList=invalidList, acc=Acc, precision=Precision, recall=Recall, err=Err, I_index = I_index, O_index = O_index, U_index = U_index, Q_index = Q_index, prev_I_index = prev_I_index, prev_O_index = prev_O_index, prev_U_index = prev_U_index, prev_Q_index = prev_Q_index)

        print("Query Strategy: "+args.query_strategy+" | Query Budget: "+str(args.query_batch)+" | Valid Query Nums: "+str(len(queryIndex))+" | Query Precision: "+str(Precision[query])+" | Query Recall: "+str(Recall[query])+" | Training Nums: "+str(len(labeled_ind_train)))

        # DataLoaders
        if args.dataset in ['cifar10', 'cifar100', 'tinyimagenet']:
            sampler_labeled = SubsetRandomSampler(I_index)  # make indices initial to the samples
            sampler_test = SubsetSequentialSampler(test_I_index)
            train_loader = DataLoader(train_dst, sampler=sampler_labeled, batch_size=args.batch_size, num_workers=args.workers)
            test_loader = DataLoader(test_dst, sampler=sampler_test, batch_size=args.test_batch_size, num_workers=args.workers)
        dataloaders = {'train': train_loader, 'test': test_loader}

    all_accuracies.append(Acc)
    all_precisions.append(Precision)
    all_recalls.append(Recall)
    print("Accuracies", all_accuracies)
    print("Precisions", all_precisions)
    print("Recalls", all_recalls)
    if args.query_strategy == 'Uncertainty':
        query_strategy = '{}_{}'.format(args.query_strategy, args.uncertainty)
    elif args.query_strategy == 'MQNet':
        query_strategy = '{}_{}'.format(args.query_strategy, args.mqnet_mode)
    else:
        query_strategy = args.query_strategy
    np.savez('results/{}_kc_{}_{}_seed_{}.npz'.format(query_strategy, args.known_class, args.dataset, args.seed), all_acc=Acc, all_precision=Precision, all_recall=Recall)   
    elapsed = round(time.time() - start_time)
    elapsed = str(datetime.timedelta(seconds=elapsed))
    print("Finished. Total elapsed time (h:m:s): {}".format(elapsed))


if __name__ == '__main__':
    main()