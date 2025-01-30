import os
import argparse
import datetime
import time
import numpy as np
import torch
import torch.nn as nn
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
from tqdm import tqdm
from sam import SAM
import copy
from resnet import ResNet18, ResClassifier_MME
import query_strategies
import datasets
from utils import AverageMeter, get_splits, open_entropy, entropic_bc_loss, reg_loss, lab_conv, unknown_clustering, enable_running_stats, disable_running_stats
import glob
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
parser.add_argument('--query-strategy', type=str, default='eoal', choices=['random', 'eoal', 'samisuk_l', 'samisuk_h', 'samisuk_dynamic', 'samis_ent', 'perfect_precision', 'custom_precision_random', 'custom_precision_mem'])

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


parser.add_argument('--custom_precision', type=float, default=1.0)


args = parser.parse_args()

def main():
    os.makedirs('new_fair_intermediaries', exist_ok=True)
    os.makedirs('new_fair_results', exist_ok=True)
    seed = args.seed
    all_accuracies = []
    all_precisions = []
    all_recalls = []
    knownclass = get_splits(args.dataset, seed, args.known_class)  
    print("Known Classes", knownclass)
    torch.manual_seed(args.seed)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    use_gpu = torch.cuda.is_available()
    if args.use_cpu: use_gpu = False

    if use_gpu:
        print("Currently using GPU: {}".format(args.gpu))
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
    trainloader_A, trainloader_B, trainloader_C = dataset.trainloader, dataset.trainloader, dataset.trainloader
    #trainloader_C = None   # init negativeloader none
    invalidList = []
    labeled_ind_train, unlabeled_ind_train = dataset.labeled_ind_train, dataset.unlabeled_ind_train
    start = 0    

    print("Creating model: {}".format(args.model))
    start_time = time.time()

    Acc = []
    Err = []
    Precision = []
    Recall = []

    #Resuming from prev round
    if args.resume:
        found_files = glob.glob(os.path.join('new_fair_intermediaries', '*.npz'))
        print('found_files: {}'.format(found_files))
        if args.query_strategy in ['custom_precision_random', 'custom_precision_mem']:
            query_strategy = '{}_{}'.format(args.query_strategy, args.custom_precision)
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
        if args.query_strategy in ['custom_precision_random', 'custom_precision_mem']:
            query_strategy = '{}_{}'.format(args.query_strategy, args.custom_precision)
        else:
            query_strategy = args.query_strategy
        data = np.load("new_fair_intermediaries/{}_kc_{}_{}_seed_{}_query_{}.npz".format(query_strategy, args.known_class, args.dataset, args.seed, last_round))
        Acc = data["acc"]
        Err = data["err"]
        Precision = data["precision"]
        Recall = data["recall"]
        labeled_ind_train = data["labeled"]
        unlabeled_ind_train = data["unlabeled"]
        invalidList = data["invalidList"] 
      
        Acc = list(Acc)
        Precision = list(Precision)
        Recall = list(Recall)
        Err = list(Err)
        unlabeled_ind_train = list(unlabeled_ind_train)
        labeled_ind_train = list(labeled_ind_train)
        invalidList = list(invalidList)
 
        B_dataset = datasets.create(
            name=args.dataset, known_class_=args.known_class, knownclass=knownclass, init_percent_=args.init_percent,
            batch_size=args.batch_size, use_gpu=use_gpu,
            num_workers=args.workers, is_filter=args.is_filter, is_mini=args.is_mini,
            unlabeled_ind_train=list(set(unlabeled_ind_train) - set(invalidList)), labeled_ind_train=labeled_ind_train+invalidList,
        )
        trainloader_A = B_dataset.trainloader
        trainloader_B = B_dataset.trainloader
        testloader = B_dataset.testloader
        unlabeledloader = B_dataset.unlabeledloader
        C_dataset = datasets.create(
            name=args.dataset, known_class_=args.known_class, knownclass=knownclass, init_percent_=args.init_percent,
            batch_size=args.batch_size, use_gpu=use_gpu,
            num_workers=args.workers, is_filter=args.is_filter, is_mini=args.is_mini,
            unlabeled_ind_train=unlabeled_ind_train, labeled_ind_train=labeled_ind_train,
        )
        trainloader_C = C_dataset.trainloader

 
    for query in tqdm(range(start, args.max_query)):
        print("Query Round: {}".format(query))

        # Model initialization
        #SGD Model for SAMIS
        model_A = ResNet18(n_class=dataset.num_classes + 1)
        #SAM Model for SAMIS
        model_B = ResNet18(n_class=dataset.num_classes + 1)
        #SGD Model for Testing
        model_C = ResNet18(n_class=dataset.num_classes)
       
        if use_gpu:
            model_A = model_A.cuda()
            model_B = model_B.cuda()
            model_C = model_C.cuda()
            
        criterion_xent = nn.CrossEntropyLoss()
        criterion_xent.cuda()
        optimizer_model_A = torch.optim.SGD(model_A.parameters(), lr=args.lr_model, weight_decay=5e-04, momentum=0.9)
        optimizer_model_B = SAM(params=model_B.parameters(),
                                base_optimizer=torch.optim.SGD,
                                lr=args.lr_model,
                                momentum=0.9,
                                weight_decay=5e-04,
                                nesterov=False,
                                rho=0.05)
        optimizer_model_C = torch.optim.SGD(model_C.parameters(), lr=args.lr_model, weight_decay=5e-04, momentum=0.9)

        if args.stepsize > 0:
            scheduler_A = lr_scheduler.StepLR(optimizer_model_A, step_size=args.stepsize, gamma=args.gamma)
            scheduler_B = lr_scheduler.StepLR(optimizer_model_B, step_size=args.stepsize, gamma=args.gamma)
            scheduler_C = lr_scheduler.StepLR(optimizer_model_C, step_size=args.stepsize, gamma=args.gamma)

        # Model training 
        for epoch in tqdm(range(args.max_epoch)):
            # Train model B for classifying known classes
            if args.query_strategy != 'perfect_precision':
                train_A(model_A, criterion_xent,
                    optimizer_model_A,
                    trainloader_A, use_gpu, knownclass, epoch)
                
                train_B(model_B, criterion_xent,
                    optimizer_model_B,
                    trainloader_B, use_gpu, knownclass, epoch)

            train_A(model_C, criterion_xent,
                optimizer_model_C,
                trainloader_C, use_gpu, knownclass, epoch)

            if args.stepsize > 0:
                if args.query_strategy != 'perfect_precision':
                    scheduler_A.step()
                    scheduler_B.step()
                scheduler_C.step()

            if args.eval_freq > 0 and (epoch+1) % args.eval_freq == 0 or (epoch+1) == args.max_epoch:
                print("==> Test")
                if args.query_strategy != 'perfect_precision':
                    acc_A, err_A = test(model_A, testloader, use_gpu, knownclass)
                    acc_B, err_B = test(model_B, testloader, use_gpu, knownclass)
                    print("Model_A | Accuracy (%): {}\t Error rate (%): {}".format(acc_A, err_A))
                    print("Model_B | Accuracy (%): {}\t Error rate (%): {}".format(acc_B, err_B))
                acc_C, err_C = test(model_C, testloader, use_gpu, knownclass)
                print("Model_C | Accuracy (%): {}\t Error rate (%): {}".format(acc_C, err_C))

        # Record results
        acc, err = test(model_C, testloader, use_gpu, knownclass)
        Acc.append(float(acc))
        Err.append(float(err))
        
        queryIndex = []
        if args.query_strategy == "random":
            queryIndex, invalidIndex, Prec, Rec = query_strategies.random_sampling(args, unlabeledloader, len(labeled_ind_train), model_B, knownclass)
        elif args.query_strategy == 'samis_ent':
            queryIndex, invalidIndex, Prec, Rec = query_strategies.samis(args, unlabeledloader, len(labeled_ind_train), model_A, model_B, knownclass, use_gpu, query)
        elif args.query_strategy == 'perfect_precision':
            queryIndex, invalidIndex, Prec, Rec = query_strategies.perfect_precision(args, unlabeledloader, len(labeled_ind_train), knownclass)
        elif args.query_strategy == 'custom_precision_random':
            queryIndex, invalidIndex, Prec, Rec = query_strategies.custom_precision_random(args, unlabeledloader, len(labeled_ind_train), knownclass)
        elif args.query_strategy == 'custom_precision_mem':
            queryIndex, invalidIndex, Prec, Rec = query_strategies.custom_precision_mem(args, unlabeledloader, len(labeled_ind_train), knownclass)
        else:
            queryIndex, invalidIndex, Prec, Rec = query_strategies.samisuk(args, unlabeledloader, len(labeled_ind_train), model_A, model_B, knownclass, use_gpu, query, this_round=query, total_rounds=args.max_query)

        Precision.append(Prec)
        Recall.append(Rec)
        
        # Update labeled, unlabeled and invalid set
        unlabeled_ind_train = list(set(unlabeled_ind_train)-set(queryIndex))
        labeled_ind_train = list(labeled_ind_train) + list(queryIndex)
        invalidList = list(invalidList) + list(invalidIndex)
        # Save in case of interruption
        if args.query_strategy in ['custom_precision_random', 'custom_precision_mem']:
            query_strategy = '{}_{}'.format(args.query_strategy, args.custom_precision)
        else:
            query_strategy = args.query_strategy
        np.savez('new_fair_intermediaries/{}_kc_{}_{}_seed_{}_query_{}.npz'.format(query_strategy, args.known_class, args.dataset, args.seed, query), unlabeled=unlabeled_ind_train, labeled=labeled_ind_train, invalidList=invalidList, acc=Acc, precision=Precision, recall=Recall, err=Err)

        print("Query Strategy: "+args.query_strategy+" | Query Budget: "+str(args.query_batch)+" | Valid Query Nums: "+str(len(queryIndex))+" | Query Precision: "+str(Precision[query])+" | Query Recall: "+str(Recall[query])+" | Training Nums: "+str(len(labeled_ind_train)))
        B_dataset = datasets.create(
            name=args.dataset, known_class_=args.known_class, knownclass=knownclass, init_percent_=args.init_percent,
            batch_size=args.batch_size, use_gpu=use_gpu,
            num_workers=args.workers, is_filter=args.is_filter, is_mini=args.is_mini,
            unlabeled_ind_train=list(set(unlabeled_ind_train) - set(invalidList)), labeled_ind_train=labeled_ind_train+invalidList,
        )
        trainloader_A = B_dataset.trainloader
        trainloader_B = B_dataset.trainloader
        testloader = B_dataset.testloader
        unlabeledloader = B_dataset.unlabeledloader
        C_dataset = datasets.create(
            name=args.dataset, known_class_=args.known_class, knownclass=knownclass, init_percent_=args.init_percent,
            batch_size=args.batch_size, use_gpu=use_gpu,
            num_workers=args.workers, is_filter=args.is_filter, is_mini=args.is_mini,
            unlabeled_ind_train=unlabeled_ind_train, labeled_ind_train=labeled_ind_train,
        )
        trainloader_C = C_dataset.trainloader

    all_accuracies.append(Acc)
    all_precisions.append(Precision)
    all_recalls.append(Recall)
    print("Accuracies", all_accuracies)
    print("Precisions", all_precisions)
    print("Recalls", all_recalls)
    if args.query_strategy in ['custom_precision_random', 'custom_precision_mem']:
        query_strategy = '{}_{}'.format(args.query_strategy, args.custom_precision)
    else:
        query_strategy = args.query_strategy
    np.savez('new_fair_results/{}_kc_{}_{}_seed_{}.npz'.format(query_strategy, args.known_class, args.dataset, args.seed), all_acc=Acc, all_precision=Precision, all_recall=Recall)   
    elapsed = round(time.time() - start_time)
    elapsed = str(datetime.timedelta(seconds=elapsed))
    print("Finished. Total elapsed time (h:m:s): {}".format(elapsed))

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
    main()





