import os
import errno
import shutil
import os.path as osp
import torch.nn.functional as F
import torch
import numpy as np
from finch import FINCH

def get_splits(dataset, seed, mismatch):
    # one can modify the seed to get different splits
    if dataset == 'cifar10':
        
        shuffled_list = [8, 2, 7, 4, 3, 5, 9, 6, 0, 1]
        knownclass = shuffled_list[:mismatch]
    elif dataset == 'cifar100':
        
        shuffled_list = [27, 56, 53, 69, 57, 89, 77, 21, 37, 86, 51, 46, 30, 68, 49, 18, 20, 43, 54, 19, 92, 31, 3, 82, 26, 12, 67, 17, 63, 55, 91, 62, 99, 38, 47, 50, 78, 24, 0, 44, 76, 16, 75, 71, 11, 94, 6, 73, 65, 32, 64, 66, 1, 15, 40, 87, 2, 96, 7, 23, 84, 72, 79, 74, 59, 85, 39, 28, 52, 48, 14, 35, 61, 81, 29, 36, 25, 9, 97, 42, 83, 70, 90, 10, 8, 98, 4, 41, 13, 22, 80, 95, 93, 58, 5, 33, 45, 88, 34, 60]
        knownclass = shuffled_list[:mismatch]
    elif dataset == 'tinyimagenet':
        shuffled_list = [172, 122, 10, 131, 8, 63, 30, 86, 135, 29, 166, 18, 54, 42, 70, 97, 4, 66, 163, 109, 92, 7, 120, 171, 40, 136, 3, 197, 104, 179, 190, 82, 72, 80, 114, 149, 36, 43, 14, 85, 133, 78, 112, 108, 139, 125, 148, 132, 118, 98, 6, 140, 25, 103, 101, 50, 93, 110, 33, 129, 182, 123, 77, 198, 175, 89, 159, 174, 31, 193, 90, 128, 141, 156, 184, 168, 111, 34, 88, 41, 5, 9, 26, 154, 49, 17, 87, 144, 15, 56, 113, 147, 189, 62, 167, 127, 100, 64, 180, 76, 0, 16, 121, 185, 153, 55, 79, 12, 116, 150, 67, 106, 107, 191, 178, 151, 60, 83, 32, 53, 176, 146, 143, 13, 105, 27, 24, 157, 137, 19, 74, 130, 102, 158, 186, 188, 126, 2, 35, 75, 165, 124, 69, 59, 84, 11, 71, 161, 22, 152, 196, 169, 160, 183, 44, 194, 115, 164, 199, 162, 61, 37, 170, 95, 134, 46, 99, 181, 20, 91, 52, 119, 28, 96, 117, 1, 73, 51, 142, 94, 58, 155, 192, 138, 177, 145, 65, 23, 21, 81, 45, 38, 48, 57, 47, 195, 187, 39, 173, 68]
        knownclass = shuffled_list[:mismatch]
    return knownclass

def open_entropy(out_open):
    assert len(out_open.size()) == 3
    assert out_open.size(1) == 2
    out_open = F.softmax(out_open, 1)
    ent_open = torch.mean(torch.sum(-out_open * torch.log(out_open + 1e-8), 1), 1)
    return ent_open

def min_max_normalization(data, new_min=0, new_max=1):
    """
    Perform min-max normalization on a NumPy array of numbers.
    :param data: NumPy array of numbers to be normalized
    :param new_min: Minimum value of the new range
    :param new_max: Maximum value of the new range
    :return: NumPy array of normalized numbers
    """
    # Ensure the data is a NumPy array
    data = np.asarray(data)

    # Find the minimum and maximum values in the data
    min_val = data.min()
    max_val = data.max()

    # Perform vectorized min-max normalization
    normalized_data = new_min + (data - min_val) * (new_max - new_min) / (max_val - min_val)

    return normalized_data

def calculate_cluster_centers(features, cluster_labels):
    unique_clusters = torch.unique(cluster_labels)
    cluster_centers = torch.zeros((len(unique_clusters), features.shape[1])).cuda()
    
    for i, cluster_id in enumerate(unique_clusters):
        cluster_indices = torch.where(cluster_labels == cluster_id)[0]
        cluster_features = features[cluster_indices]
        # Calculate the center of the cluster using the mean of features
        cluster_center = torch.mean(cluster_features, dim=0)
        cluster_centers[i] = cluster_center
    return cluster_centers

def unknown_clustering(args, model, model_bc, trainloader_C, use_gpu, knownclass):
    model.eval()
    model_bc.eval()
    feat_all = torch.zeros([1, 128], device='cuda')
    labelArr, labelArr_true, queryIndex, y_pred = [], [], [], []
    
    for batch_idx, (index, (data, labels)) in enumerate(trainloader_C):
        labels_true = labels
        labelArr_true += list(labels_true.cpu().data.numpy())
        labels = lab_conv(knownclass, labels)
        if use_gpu:
            data, labels = data.cuda(), labels.cuda()
        outputs, features = model(data)
        softprobs = torch.softmax(outputs, dim=1)
        _, predicted = torch.max(softprobs, 1)
        y_pred += list(predicted.cpu().data.numpy())
        feat_all = torch.cat([feat_all, features.data], 0)
        queryIndex += index
        labelArr += list(labels.cpu().data.numpy())
    
    queryIndex = np.array(queryIndex)
    y_pred = np.array(y_pred)

    embeddings = feat_all[1:].cpu().numpy()
    _, _, req_c = FINCH(embeddings, req_clust= args.w_unk_cls * len(knownclass), verbose=False)
    cluster_labels = req_c
    # Convert back to tensors after clustering
    embeddings = torch.tensor(embeddings, device='cuda')
    labelArr_true = torch.tensor(labelArr_true)
    queryIndex = torch.tensor(queryIndex)
    cluster_labels = torch.tensor(cluster_labels)
    cluster_centers = calculate_cluster_centers(embeddings, cluster_labels)
    return cluster_centers, embeddings, cluster_labels, queryIndex

def reg_loss(features, labels, cluster_centers, cluster_labels, num_classes):
    features_k, _ = features[labels<num_classes], labels[labels<num_classes]
    features_u, _ = features[labels==num_classes], labels[labels==num_classes]
    k_dists = torch.cdist(features_k, cluster_centers)
    uk_dists = torch.cdist(features_u, cluster_centers)
    pk = torch.softmax(-k_dists, dim=1)
    pu = torch.softmax(-uk_dists, dim=1)

    k_ent = -torch.sum(pk*torch.log(pk+1e-20), 1)
    u_ent = -torch.sum(pu*torch.log(pu+1e-20), 1)
    true = torch.gather(uk_dists, 1, cluster_labels.long().view(-1, 1)).view(-1)
    non_gt = torch.tensor([[i for i in range(len(cluster_centers)) if cluster_labels[x] != i] for x in range(len(uk_dists))]).long().cuda()
    others = torch.gather(uk_dists, 1, non_gt)
    intra_loss = torch.mean(true)
    inter_loss = torch.exp(-others+true.unsqueeze(1))
    inter_loss = torch.mean(torch.log(1+torch.sum(inter_loss, dim = 1)))
    loss = 0.1*intra_loss + 1*inter_loss
    return loss, k_ent.sum(), u_ent.sum()

def entropic_bc_loss(out_open, label, pareto_alpha, num_classes, query, weight):
    assert len(out_open.size()) == 3
    assert out_open.size(1) == 2

    out_open = F.softmax(out_open, 1)
    label_p = torch.zeros((out_open.size(0),
                        out_open.size(2)+1)).cuda()  
    label_range = torch.arange(0, out_open.size(0))  
    label_p[label_range, label] = 1  
    label_n = 1 - label_p
    if query > 0:
        label_p[label==num_classes,:] = pareto_alpha/num_classes
        label_n[label==num_classes,:] = pareto_alpha/num_classes
    label_p = label_p[:,:-1]
    label_n = label_n[:,:-1]
    valid_labels_mask = (label < num_classes)

    if (query > 0) and (weight!=0):
        
        open_loss_pos = torch.mean(torch.sum(-torch.log(out_open[label<num_classes, 1, :]
                                                        + 1e-8) * (1 - pareto_alpha) * label_p[label<num_classes], 1))
        if valid_labels_mask.any():
            open_loss_neg = torch.mean(torch.max(-torch.log(out_open[label<num_classes, 0, :] +
                                                    1e-8) * (1 - pareto_alpha) * label_n[label<num_classes], 1)[0]) ##### take max negative alone
        else: 
            open_loss_neg = torch.tensor(0.0).cuda()
        open_loss_pos_ood = torch.mean(torch.sum(-torch.log(out_open[label==num_classes, 1, :] +
                                                    1e-8) * label_p[label==num_classes], 1))
        open_loss_neg_ood = torch.mean(torch.sum(-torch.log(out_open[label==num_classes, 0, :] +
                                                    1e-8) * label_n[label==num_classes], 1))
        
        return open_loss_pos, open_loss_neg, open_loss_neg_ood, open_loss_pos_ood
    else:
        open_loss_pos = torch.mean(torch.sum(-torch.log(out_open[:, 1, :]
                                                        + 1e-8) * (1 - 0) * label_p, 1))
        open_loss_neg = torch.mean(torch.max(-torch.log(out_open[:, 0, :] +
                                                    1e-8) * (1 - 0) * label_n, 1)[0]) ##### take max negative alone
        return open_loss_pos, open_loss_neg, 0, 0

def bc_loss(out_open, label):
    assert len(out_open.size()) == 3
    assert out_open.size(1) == 2

    out_open = F.softmax(out_open, 1)
    label_p = torch.zeros((out_open.size(0),
                        out_open.size(2))).long().cuda()  ##### torch.Size([36, 20]) - zeros
    label_range = torch.arange(0, out_open.size(0)).long()  ##### label_range - batch size
    label_p[label_range, label] = 1  ###### set label to 1 - [0,0,0,0,....,1,0,0,0]
    label_n = 1 - label_p  ###### label_n - reamining all 1 - [1,1,1,1,,,,,0,1,1,1]
    open_loss_pos = torch.mean(torch.sum(-torch.log(out_open[:, 1, :]
                                                    + 1e-8) * label_p, 1))
    open_loss_neg = torch.mean(torch.max(-torch.log(out_open[:, 0, :] +
                                                1e-8) * label_n, 1)[0]) ##### take max negative alone
    # open_loss_neg = torch.mean(torch.sum(-torch.log(out_open[:, 0, :] +
    #                                             1e-8) * label_n, 1)[0])
    return open_loss_pos, open_loss_neg

def lab_conv(knownclass, label):
    knownclass = sorted(knownclass)
    label_convert = torch.zeros(len(label), dtype=int)
    for j in range(len(label)):
        for i in range(len(knownclass)):

            if label[j] == knownclass[i]:
                label_convert[j] = int(knownclass.index(knownclass[i]))
                break
            else:
                label_convert[j] = int(len(knownclass))   
                #label_convert[j] = label[j]
    return label_convert

def mkdir_if_missing(directory):
    if not osp.exists(directory):
        try:
            os.makedirs(directory)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise

class AverageMeter(object):
    """Computes and stores the average and current value.
       
       Code imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def save_checkpoint(state, is_best, fpath='checkpoint.pth.tar'):
    mkdir_if_missing(osp.dirname(fpath))
    torch.save(state, fpath)
    if is_best:
        shutil.copy(fpath, osp.join(osp.dirname(fpath), 'best_model.pth.tar'))

def save_model(model_A, model_B, model_bc, save_path):
    save_dic = {
        'A_state_dict': model_A.state_dict(),
        'B_state_dict': model_B.state_dict(),
        'bc_state_dict': model_bc.state_dict(),
    }
    torch.save(save_dic, save_path)

def load_model(model_A, model_B, model_bc, load_path):
    checkpoint = torch.load(load_path)
    model_A.load_state_dict(checkpoint['A_state_dict'])
    model_B.load_state_dict(checkpoint['B_state_dict'])
    model_bc.load_state_dict(checkpoint['bc_state_dict'])
    return model_A, model_B, model_bc

def disable_running_stats(model):
    def _disable(module):
        if isinstance(module, torch.nn.BatchNorm2d):
            module.backup_momentum = module.momentum
            module.momentum = 0
    model.apply(_disable)


def enable_running_stats(model):
    def _enable(module):
        if isinstance(module, torch.nn.BatchNorm2d) and hasattr(module, "backup_momentum"):
            module.momentum = module.backup_momentum
    model.apply(_enable)
