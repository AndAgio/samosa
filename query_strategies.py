import numpy as np
import torch
from math import log
import numpy as np
from finch import FINCH
import torch.nn.functional as F
from utils import open_entropy, lab_conv, min_max_normalization
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import RobustScaler
from sklearn.utils import shuffle

from src.metrics import Proxy


def eoal_sampling(args, unlabeledloader, Len_labeled_ind_train, model, model_bc, knownclass, use_gpu, cluster_centers=None, cluster_labels=None, first_rd=True, diversity=True):
    
    model.eval()
    model_bc.eval()
    labelArr, queryIndex, entropy_list, y_pred, unk_entropy_list = [], [], [], [], []
    feat_all = torch.zeros([1, 128]).cuda()
    precision, recall = 0, 0

    for batch_idx, (index, (data, labels)) in enumerate(unlabeledloader):
        labels = lab_conv(knownclass, labels)
        if use_gpu:
            data, labels = data.cuda(), labels.cuda()
        outputs, features = model(data)
        softprobs = torch.softmax(outputs, dim=1)
        _, predicted = torch.max(softprobs, 1)
        queryIndex += index
        y_pred += list(np.array(predicted.cpu().data))
        labelArr += list(np.array(labels.cpu().data))
        feat_all = torch.cat([feat_all, features.data],0)
        out_open = model_bc(features)
        out_open = out_open.view(outputs.size(0), 2, -1)

        ####### closed-set entropy score
        entropy_data = open_entropy(out_open)
        entropy_list.append(entropy_data.cpu().data)
        ####### distance-based entropy score
        if not first_rd:
            dists = torch.cdist(features, cluster_centers)
            similarity_scores_cj = torch.softmax(-dists, dim=1)
            pred_ent = -torch.sum(similarity_scores_cj*torch.log(similarity_scores_cj+1e-20), 1)
            unk_entropy_list.append(pred_ent.cpu().data)

    entropy_list = torch.cat(entropy_list).cpu()
    entropy_list = entropy_list / log(2)

    y_pred = np.array(y_pred)
    labelArr = torch.tensor(labelArr)
    labelArr_k = labelArr[y_pred < args.known_class]
    
    if not first_rd:
        unk_entropy_list = torch.cat(unk_entropy_list).cpu()
        unk_entropy_list = unk_entropy_list / log(len(cluster_centers))
        entropy_list = entropy_list - unk_entropy_list
        
    embeddings = feat_all[1:].cpu().numpy()
    embeddings_k = embeddings[y_pred < args.known_class]

    uncertaintyArr_k = entropy_list[y_pred < args.known_class]
    queryIndex = torch.tensor(queryIndex)
    queryIndex_k = queryIndex[y_pred < args.known_class]
    
    if not diversity:
        sorted_idx = uncertaintyArr_k.sort()[1][:args.query_batch]
        selected_idx = queryIndex_k[sorted_idx]
        selected_gt = labelArr_k[sorted_idx]
        selected_gt = selected_gt.numpy()
        selected_idx = selected_idx.numpy()

    else:        
        labels_c, num_clust, _ = FINCH(embeddings_k, req_clust= len(knownclass), verbose=True)
        tmp_var = 0
        while num_clust[tmp_var] > args.query_batch:
            tmp_var += 1
        cluster_labels = labels_c[:, tmp_var]
        num_clusters = num_clust[tmp_var]

        rem = min(args.query_batch, len(queryIndex_k))
        num_per_cluster = int(rem/num_clusters)
        selected_idx = []
        selected_gt = []

        ax = [0 for i in range(num_clusters)]
        while rem > 0:
            print("Remaining Budget to Sample:  ", rem)
            for cls in range(num_clusters):
                temp_ent = uncertaintyArr_k[cluster_labels == cls]
                temp_index = queryIndex_k[cluster_labels == cls]
                temp_gt = labelArr_k[cluster_labels == cls]
                if rem >= num_per_cluster:
                    sorted_idx = temp_ent.sort()[1][ax[cls]:ax[cls]+min(num_per_cluster, len(temp_ent))]
                    ax[cls] += len(sorted_idx)
                    rem -= len(sorted_idx)
                else:
                    sorted_idx = temp_ent.sort()[1][ax[cls]:ax[cls]+min(rem, len(temp_ent))]
                    ax[cls] += len(sorted_idx)
                    rem -= len(sorted_idx)
                q_idxs = temp_index[sorted_idx.cpu()]
                selected_idx.extend(list(q_idxs.numpy()))
                gt_cls = temp_gt[sorted_idx.cpu()]
                selected_gt.extend(list(gt_cls.numpy()))
        print("clustering finished")
        selected_gt = np.array(selected_gt)
        selected_idx = np.array(selected_idx)
    
    if len(selected_gt) < args.query_batch:
        rem_budget = args.query_batch - len(set(selected_idx))
        print("Not using all the budget...")
        uncertaintyArr_u = entropy_list[y_pred >= args.known_class]
        queryIndex_u = queryIndex[y_pred >= args.known_class]
        queryIndex_u = np.array(queryIndex_u)
        labelArr_u = labelArr[y_pred >= args.known_class]
        labelArr_u = np.array(labelArr_u)
        tmp_data = np.vstack((queryIndex_u, labelArr_u)).T
        print("Choosing from the K+1 classifier's rejected samples...")
        sorted_idx_extra = uncertaintyArr_u.sort()[1][:rem_budget]
        tmp_data = tmp_data.T
        rand_idx = tmp_data[0][sorted_idx_extra.cpu().numpy()]
        rand_LabelArr = tmp_data[1][sorted_idx_extra.cpu().numpy()]
        selected_gt = np.concatenate((selected_gt, rand_LabelArr))
        selected_idx = np.concatenate((selected_idx, rand_idx))
    
    precision = len(np.where(selected_gt < args.known_class)[0]) / len(selected_gt)
    recall = (len(np.where(selected_gt < args.known_class)[0]) + Len_labeled_ind_train) / (
                len(np.where(labelArr < args.known_class)[0]) + Len_labeled_ind_train)
    return selected_idx[np.where(selected_gt < args.known_class)[0]], selected_idx[np.where(selected_gt >= args.known_class)[0]], precision, recall

def random_sampling(args, unlabeledloader, Len_labeled_ind_train, model, knownclass):
    model.eval()
    queryIndex = []
    labelArr = []
    precision, recall = 0, 0
    for batch_idx, (index, (_, labels)) in enumerate(unlabeledloader):
        labels = lab_conv(knownclass, labels)
        queryIndex += index
        labelArr += list(np.array(labels.cpu().data))

    tmp_data = np.vstack((queryIndex, labelArr)).T
    np.random.shuffle(tmp_data)
    tmp_data = tmp_data.T
    queryIndex = tmp_data[0][:args.query_batch]
    labelArr = tmp_data[1]
    queryLabelArr = tmp_data[1][:args.query_batch]
    precision = len(np.where(queryLabelArr < args.known_class)[0]) / len(queryLabelArr)
    recall = (len(np.where(queryLabelArr < args.known_class)[0]) + Len_labeled_ind_train) / (len(np.where(labelArr < args.known_class)[0]) + Len_labeled_ind_train)
    return queryIndex[np.where(queryLabelArr < args.known_class)[0]], queryIndex[np.where(queryLabelArr >= args.known_class)[0]], precision, recall



def samis(args, unlabeledloader, Len_labeled_ind_train, sgd_model, sam_model, knownclass, use_gpu, query):
    sgd_model.eval()
    sam_model.eval()
    labelArr, queryIndex, samis_list, y_pred, entropy_list, sam_entropy_list, sam_pred, sgd_pred = [], [], [], [], [], [], [], []
    feat_all = torch.zeros([1, 128]).cuda()
    precision, recall = 0, 0

    for batch_idx, (index, (data, labels)) in enumerate(unlabeledloader):
        labels = lab_conv(knownclass, labels)
        if use_gpu:
            data, labels = data.cuda(), labels.cuda()
        sgd_outputs, features = sgd_model(data)
        sam_outputs, features = sam_model(data)
        sgd_probs = torch.softmax(sgd_outputs, dim=1)
        sam_probs = torch.softmax(sam_outputs, dim=1)
        _, sgd_predicted = torch.max(sgd_probs, 1)
        _, sam_predicted = torch.max(sam_probs, 1)
        sgd_pred.extend(sgd_predicted.tolist())
        sam_pred.extend(sam_predicted.tolist())
        # Calculate the absolute differences of the probabilities for each data point in the batch
        differences = torch.abs(sgd_probs - sam_probs).sum(dim=1)

        # Calculate KL divergence
        #differences = F.kl_div(log_sgd_probs, sam_probs, reduction='none').sum(dim=1)
        sgd_probs = sgd_probs[:, :args.known_class]
        sam_probs = sam_probs[:, :args.known_class]

        # Normalize the probabilities so they sum to 1 across the known classes
        sgd_probs /= sgd_probs.sum(dim=1, keepdim=True)
        sam_probs /= sam_probs.sum(dim=1, keepdim=True)
        entropies = -torch.sum(sgd_probs * torch.log(sgd_probs + 1e-9), dim=1)
        sam_entropies = -torch.sum(sam_probs * torch.log(sam_probs + 1e-9), dim=1) 
        # Convert differences to Python scalars and append to samis_list
        samis_list.extend(differences.tolist())
        entropy_list.extend(entropies.tolist()) 
        sam_entropy_list.extend(sam_entropies.tolist())  
        queryIndex += index
        labelArr += list(np.array(labels.cpu().data))


    samis_list = np.array(samis_list)
    entropy_list = np.array(entropy_list)
    sam_entropy_list = np.array(sam_entropy_list)
    y_pred = np.array(y_pred)
    labelArr = torch.tensor(labelArr)
    queryIndex = torch.tensor(queryIndex)

    sam_pred = np.array(sam_pred)
    sgd_pred = np.array(sgd_pred)
    entropy_list_k = entropy_list[(sgd_pred < args.known_class) & (sam_pred < args.known_class)]
    entropy_list_k = min_max_normalization(entropy_list_k)
    sam_entropy_list_k = sam_entropy_list[(sgd_pred < args.known_class) & (sam_pred < args.known_class)]
    sam_entropy_list_k = min_max_normalization(sam_entropy_list_k)
    samis_list_k = samis_list[(sgd_pred < args.known_class) & (sam_pred < args.known_class)]
    samis_list_k = min_max_normalization(samis_list_k)
    labelArr_k = labelArr[(sgd_pred < args.known_class) & (sam_pred < args.known_class)]
    queryIndex_k = queryIndex[(sgd_pred < args.known_class) & (sam_pred < args.known_class)]
    #np.savez('log/low_samisuk_{}_round{}.npz'.format(args.dataset, query), sgd_pred=sgd_pred, sam_pred=sam_pred, samis=samis_list, sgd_entropy=entropy_list, sam_entropy=sam_entropy_list, labelArr=labelArr)
    if query == 0:
        sorted_idx = np.argsort(samis_list_k + entropy_list_k/2 + sam_entropy_list_k/2)
        sorted_idx = sorted_idx[:args.query_batch]
        selected_idx = queryIndex_k[sorted_idx]
        selected_gt = labelArr_k[sorted_idx]
        selected_gt = selected_gt.numpy()
        selected_idx = selected_idx.numpy()
    else:
        sorted_idx = np.argsort(samis_list_k + entropy_list_k/2 + sam_entropy_list_k/2)
        top = sorted_idx[:int(args.query_batch / 2)]
        bottom = sorted_idx[-(args.query_batch - int(args.query_batch / 2)):]
        combined = np.concatenate([top, bottom])
        sorted_idx = np.unique(combined)
        #sorted_idx = sorted_idx[:args.query_batch]
        selected_idx = queryIndex_k[sorted_idx]
        selected_gt = labelArr_k[sorted_idx]
        selected_gt = selected_gt.numpy()
        selected_idx = selected_idx.numpy()
    # sorted_idx = sorted_idx[:args.query_batch]

    #If rejected samples are less than samples to query
    if len(selected_gt) < args.query_batch:
        print("Not using all the budget...")
        rem_budget = args.query_batch - len(set(selected_idx))
        queryIndex_u = queryIndex[~((sgd_pred < args.known_class) & (sam_pred < args.known_class))]
        samis_list_u = samis_list[~((sgd_pred < args.known_class) & (sam_pred < args.known_class))]
        labelArr_u = labelArr[~((sgd_pred < args.known_class) & (sam_pred < args.known_class))]
        sorted_idx = np.argsort(-samis_list_u)[:rem_budget]
        selected_gt = np.concatenate((selected_gt, labelArr_u[sorted_idx]))
        selected_idx = np.concatenate((selected_idx, queryIndex_u[sorted_idx]))
    precision = len(np.where(selected_gt < args.known_class)[0]) / len(selected_gt)
    recall = (len(np.where(selected_gt < args.known_class)[0]) + Len_labeled_ind_train) / (
                len(np.where(labelArr < args.known_class)[0]) + Len_labeled_ind_train)


    
    return selected_idx[np.where(selected_gt < args.known_class)[0]], selected_idx[np.where(selected_gt >= args.known_class)[0]], precision, recall    


def samisuk(args, unlabeledloader, Len_labeled_ind_train, sgd_model, sam_model, knownclass, use_gpu, query, this_round=None, total_rounds=None):
    sgd_model.eval()
    sam_model.eval()
    labelArr, queryIndex, samis_list, y_pred, entropy_list, sam_entropy_list, sam_pred, sgd_pred = [], [], [], [], [], [], [], []
    feat_all = torch.zeros([1, 128]).cuda()
    precision, recall = 0, 0

    for batch_idx, (index, (data, labels)) in enumerate(unlabeledloader):
        labels = lab_conv(knownclass, labels)
        if use_gpu:
            data, labels = data.cuda(), labels.cuda()
        sgd_outputs, features = sgd_model(data)
        sam_outputs, features = sam_model(data)
        sgd_probs = torch.softmax(sgd_outputs, dim=1)
        sam_probs = torch.softmax(sam_outputs, dim=1)
        _, sgd_predicted = torch.max(sgd_probs, 1)
        _, sam_predicted = torch.max(sam_probs, 1)
        sgd_pred.extend(sgd_predicted.tolist())
        sam_pred.extend(sam_predicted.tolist())
        # Calculate the absolute differences of the probabilities for each data point in the batch
        differences = torch.abs(sgd_probs - sam_probs).sum(dim=1)

        # Calculate KL divergence
        #differences = F.kl_div(log_sgd_probs, sam_probs, reduction='none').sum(dim=1)
        # Convert differences to Python scalars and append to samis_list
        samis_list.extend(differences.tolist())
        queryIndex += index
        labelArr += list(np.array(labels.cpu().data))


    samis_list = np.array(samis_list)
    y_pred = np.array(y_pred)
    labelArr = torch.tensor(labelArr)
    queryIndex = torch.tensor(queryIndex)

    sam_pred = np.array(sam_pred)
    sgd_pred = np.array(sgd_pred)
    samis_list_k = samis_list[(sgd_pred < args.known_class) & (sam_pred < args.known_class)]
    samis_list_k = min_max_normalization(samis_list_k)
    labelArr_k = labelArr[(sgd_pred < args.known_class) & (sam_pred < args.known_class)]
    queryIndex_k = queryIndex[(sgd_pred < args.known_class) & (sam_pred < args.known_class)]
    #np.savez('log/low_samisuk_{}_round{}.npz'.format(args.dataset, query), sgd_pred=sgd_pred, sam_pred=sam_pred, samis=samis_list, sgd_entropy=entropy_list, sam_entropy=sam_entropy_list, labelArr=labelArr)
    
    if args.query_strategy == "samisuk_l":
        #SAMIS Low
        sorted_idx = np.argsort(samis_list_k)
        sorted_idx = sorted_idx[:args.query_batch]
    elif args.query_strategy == "samisuk_h":
        #SAMIS High
        sorted_idx = np.argsort(-samis_list_k)
        sorted_idx = sorted_idx[:args.query_batch]
    elif args.query_strategy == 'samisuk_dynamic':
        #SAMIS dynamic
        assert this_round is not None
        assert total_rounds is not None
        round_index = float(this_round)/float(total_rounds)
        print('round_index: {}'.format(round_index))
        #print('samis_list_k: {}'.format(samis_list_k))
        #print('np.sort(samis_list_k): {}'.format(np.sort(samis_list_k)))
        sorted_idx = np.argsort(samis_list_k)
        sample_weight = [(samis_list_k[ind]+(1-round_index))*((1-samis_list_k[ind])+round_index) * ((samis_list_k[ind] - 0.5) ** 2 + (round_index - 0.5) ** 2 + 0.5) for ind in sorted_idx]
        #print('sample_weight: {}'.format(sample_weight))
        sample_ps = sample_weight/np.sum(sample_weight)
        #print('sample_ps: {}'.format(sample_ps))
        sorted_idx = np.random.choice(sorted_idx, size=args.query_batch, p=sample_ps)
     
    # sorted_idx = sorted_idx[:args.query_batch]

    # Use sorted_idx to select indices
    selected_idx = queryIndex_k[sorted_idx]
    selected_gt = labelArr_k[sorted_idx]
    selected_gt = selected_gt.numpy()
    selected_idx = selected_idx.numpy()
    #If rejected samples are less than samples to query
    if len(selected_gt) < args.query_batch:
        print("Not using all the budget...")
        rem_budget = args.query_batch - len(set(selected_idx))
        queryIndex_u = queryIndex[~((sgd_pred < args.known_class) & (sam_pred < args.known_class))]
        samis_list_u = samis_list[~((sgd_pred < args.known_class) & (sam_pred < args.known_class))]
        labelArr_u = labelArr[~((sgd_pred < args.known_class) & (sam_pred < args.known_class))]
        sorted_idx = np.argsort(-samis_list_u)[:rem_budget]
        selected_gt = np.concatenate((selected_gt, labelArr_u[sorted_idx]))
        selected_idx = np.concatenate((selected_idx, queryIndex_u[sorted_idx]))
    precision = len(np.where(selected_gt < args.known_class)[0]) / len(selected_gt)
    recall = (len(np.where(selected_gt < args.known_class)[0]) + Len_labeled_ind_train) / (
                len(np.where(labelArr < args.known_class)[0]) + Len_labeled_ind_train)
    return selected_idx[np.where(selected_gt < args.known_class)[0]], selected_idx[np.where(selected_gt >= args.known_class)[0]], precision, recall   

def AV_sampling_temperature(args, unlabeledloader, Len_labeled_ind_train, model, use_gpu, knownclass):
    model.eval()
    queryIndex = []
    labelArr = []
    uncertaintyArr = []
    S_ij = {}
    for batch_idx, (index, (data, labels)) in enumerate(unlabeledloader):
        labels = lab_conv(knownclass, labels)
        if use_gpu:
            data, labels = data.cuda(), labels.cuda()
        features, outputs = model(data)
        queryIndex += index
        labelArr += list(np.array(labels.cpu().data))
        # activation value based
        v_ij, predicted = outputs.max(1)
        for i in range(len(predicted.data)):
            tmp_class = np.array(predicted.data.cpu())[i]
            tmp_index = index[i]
            tmp_label = np.array(labels.data.cpu())[i]
            tmp_value = np.array(v_ij.data.cpu())[i]
            if tmp_class not in S_ij:
                S_ij[tmp_class] = []
            S_ij[tmp_class].append([tmp_value, tmp_index, tmp_label])


    # fit a two-component GMM for each class
    tmp_data = []
    for tmp_class in S_ij:
        S_ij[tmp_class] = np.array(S_ij[tmp_class])
        activation_value = S_ij[tmp_class][:, 0]
        max_value = np.nanmax(activation_value)
        # Replace NaNs with the maximum value found
        activation_value = np.nan_to_num(activation_value, nan=max_value)
        if np.any(activation_value > np.finfo(np.float64).max):
            scaler = RobustScaler()
            activation_value = scaler.fit_transform(activation_value)
        if len(activation_value) < 2:
            continue
        gmm = GaussianMixture(n_components=2, max_iter=10, tol=1e-2, reg_covar=5e-4)
        gmm.fit(np.array(activation_value).reshape(-1, 1))
        prob = gmm.predict_proba(np.array(activation_value).reshape(-1, 1))
        # 得到为known类别的概率
        prob = prob[:, gmm.means_.argmax()]
        # 如果为unknown类别直接为0
        if tmp_class == args.known_class:
            prob = [0]*len(prob)
            prob = np.array(prob)

        if len(tmp_data) == 0:
            tmp_data = np.hstack((prob.reshape(-1, 1), S_ij[tmp_class]))
        else:
            tmp_data = np.vstack((tmp_data, np.hstack((prob.reshape(-1, 1), S_ij[tmp_class]))))


    tmp_data = tmp_data[np.argsort(tmp_data[:, 0])]
    tmp_data = tmp_data.T
    queryIndex = tmp_data[2][-args.query_batch:].astype(int)
    labelArr = tmp_data[3].astype(int)
    queryLabelArr = tmp_data[3][-args.query_batch:]
    precision = len(np.where(queryLabelArr < args.known_class)[0]) / len(queryLabelArr)
    recall = (len(np.where(queryLabelArr < args.known_class)[0]) + Len_labeled_ind_train) / (
            len(np.where(labelArr < args.known_class)[0]) + Len_labeled_ind_train)
    return queryIndex[np.where(queryLabelArr < args.known_class)[0]], queryIndex[np.where(queryLabelArr >= args.known_class)[0]], precision, recall


def lfosamis(args, unlabeledloader, Len_labeled_ind_train, model, model_c, use_gpu, knownclass):
    model.eval()
    model_c.eval()
    samis_list = []
    queryIndex = []
    count = 0
    labelArr = []
    uncertaintyArr = []
    S_ij = {}
    for batch_idx, (index, (data, labels)) in enumerate(unlabeledloader):
        labels = lab_conv(knownclass, labels)
        if use_gpu:
            data, labels = data.cuda(), labels.cuda()
        features, outputs = model(data)
        features, sam_outputs = model_c(data)
        sgd_probs = torch.softmax(outputs, dim=1)
        sam_probs = torch.softmax(sam_outputs, dim=1)
        differences = torch.abs(sgd_probs - sam_probs).sum(dim=1)

        # Calculate KL divergence
        #differences = F.kl_div(log_sgd_probs, sam_probs, reduction='none').sum(dim=1)
        # Convert differences to Python scalars and append to samis_list
        samis_list.extend(differences.tolist())
        queryIndex += index
        labelArr += list(np.array(labels.cpu().data))
        # activation value based
        v_ij, predicted = outputs.max(1)
        for i in range(len(predicted.data)):
            tmp_class = np.array(predicted.data.cpu())[i]
            tmp_index = index[i]
            tmp_label = np.array(labels.data.cpu())[i]
            tmp_value = np.array(v_ij.data.cpu())[i]
            tmp_samis = np.array(differences.tolist()[i])
            if tmp_class not in S_ij:
                S_ij[tmp_class] = []
            S_ij[tmp_class].append([tmp_value, tmp_index, tmp_label, tmp_samis])

    samis_list = np.array(samis_list)
    # fit a two-component GMM for each class
    tmp_data = []
    for tmp_class in S_ij:
        S_ij[tmp_class] = np.array(S_ij[tmp_class])
        activation_value = S_ij[tmp_class][:, 0]
        max_value = np.nanmax(activation_value)
        # Replace NaNs with the maximum value found
        activation_value = np.nan_to_num(activation_value, nan=max_value)
        if np.any(activation_value > np.finfo(np.float64).max):
            scaler = RobustScaler()
            activation_value = scaler.fit_transform(activation_value)
        if len(activation_value) < 2:
            continue
        gmm = GaussianMixture(n_components=2, max_iter=10, tol=1e-2, reg_covar=5e-4)
        gmm.fit(np.array(activation_value).reshape(-1, 1))
        prob = gmm.predict_proba(np.array(activation_value).reshape(-1, 1))
        # 得到为known类别的概率
        prob = prob[:, gmm.means_.argmax()]
        # 如果为unknown类别直接为0
        if tmp_class == args.known_class:
            prob = [0]*len(prob)
            prob = np.array(prob)

        if len(tmp_data) == 0:
            tmp_data = np.hstack((prob.reshape(-1, 1), S_ij[tmp_class]))
        else:
            tmp_data = np.vstack((tmp_data, np.hstack((prob.reshape(-1, 1), S_ij[tmp_class]))))

    tmp_ind = np.argsort(tmp_data[:, 0])
    samis_list = samis_list
    tmp_data = tmp_data[np.argsort(tmp_data[:, 0])]
    tmp_data = tmp_data[-args.query_batch*3:]
    tmp_data = tmp_data[np.argsort(tmp_data[:, 4])]
    tmp_data = tmp_data[-args.query_batch:]
    tmp_data = tmp_data.T

    queryIndex = tmp_data[2].astype(int)
    labelArr = tmp_data[3].astype(int)
    queryLabelArr = tmp_data[3]
    precision = len(np.where(queryLabelArr < args.known_class)[0]) / len(queryLabelArr)
    recall = (len(np.where(queryLabelArr < args.known_class)[0]) + Len_labeled_ind_train) / (
            len(np.where(labelArr < args.known_class)[0]) + Len_labeled_ind_train)
    return queryIndex[np.where(queryLabelArr < args.known_class)[0]], queryIndex[np.where(queryLabelArr >= args.known_class)[0]], precision, recall


def uncertainty_sampling(args, unlabeledloader, Len_labeled_ind_train, model, knownclass):
    model.eval()
    queryIndex = []
    labelArr = []
    uncertaintyArr = []
    precision, recall = 0, 0
    for batch_idx, (index, (data, labels)) in enumerate(unlabeledloader):
        labels = lab_conv(knownclass, labels)
        
        data, labels = data.cuda(), labels.cuda()
        outputs, features = model(data)

        uncertaintyArr += list(np.array((-torch.softmax(outputs,1)*torch.log(torch.softmax(outputs,1))).sum(1).cpu().data))
        queryIndex += index
        labelArr += list(np.array(labels.cpu().data))

    tmp_data = np.vstack((uncertaintyArr, queryIndex, labelArr)).T
    tmp_data = tmp_data[np.argsort(tmp_data[:,0])]
    tmp_data = tmp_data.T
    queryIndex = tmp_data[1][-args.query_batch:].astype(int)
    labelArr = tmp_data[2].astype(int)
    queryLabelArr = tmp_data[2][-args.query_batch:]
    precision = len(np.where(queryLabelArr < args.known_class)[0]) / len(queryLabelArr)
    recall = (len(np.where(queryLabelArr < args.known_class)[0]) + Len_labeled_ind_train) / (
                len(np.where(labelArr < args.known_class)[0]) + Len_labeled_ind_train)
    return queryIndex[np.where(queryLabelArr < args.known_class)[0]], queryIndex[np.where(queryLabelArr >= args.known_class)[0]], precision, recall

def perfect_precision(args, unlabeledloader, Len_labeled_ind_train, knownclass):
    precision, recall = 0, 0
    labelArr = []
    mem_scores = Proxy(mode='mem', dataset=args.dataset).get_dict_form(sort=False)
    unlabeled_mem_scores = {}
    index_label_dict = {}
    for batch_idx, (index, (data, labels)) in enumerate(unlabeledloader):
        # labels = lab_conv(knownclass, labels)
        for i, label in enumerate(labels):
            # print('index[i]: {}'.format(index[i]))
            # print('label: {}'.format(label))
            # print('knownclass: {}'.format(knownclass))
            index_label_dict[index[i]] = label
            if label in knownclass:
                unlabeled_mem_scores[index[i]] = mem_scores[index[i]]
        labelArr += list(np.array(labels.cpu().data))

    # print('unlabeled_mem_scores: {}'.format(unlabeled_mem_scores))
    sorted_items = sorted(unlabeled_mem_scores.items(), key=lambda t: t[1])
    # print('sorted_items: {}'.format(sorted_items))
    queryIndex = list([item[0] for item in sorted_items][:args.query_batch])
    # print('queryIndex: {}'.format(queryIndex))
    # print(len(queryIndex))
    # print('Queried mem scores: {}'.format([unlabeled_mem_scores[quer] for quer in queryIndex]))

    for index in queryIndex:
        # print('index: {}'.format(index))
        # print('knownclass: {}'.format(knownclass))
        # print('class: {}'.format(index_label_dict[index]))
        # print('unlabeled_mem_scores[index]: {}'.format(unlabeled_mem_scores[index]))
        assert index_label_dict[index] in knownclass

    labelArr = np.array(labelArr).astype(int)
    labelArr = lab_conv(knownclass, torch.from_numpy(labelArr))
    queryLabelArr = np.array([index_label_dict[index] for index in queryIndex])
    queryLabelArr = lab_conv(knownclass, torch.from_numpy(queryLabelArr))
    precision = len(np.where(queryLabelArr < args.known_class)[0]) / len(queryLabelArr)
    recall = (len(np.where(queryLabelArr < args.known_class)[0]) + Len_labeled_ind_train) / (
                len(np.where(labelArr < args.known_class)[0]) + Len_labeled_ind_train)
    # print('precision: {}'.format(precision))
    # print('recall: {}'.format(recall))

    return queryIndex, [], precision, recall


def custom_precision_random(args, unlabeledloader, Len_labeled_ind_train, knownclass):
    correct_queryIndex, wrong_queryIndex, queryIndex = [], [], []
    correct_labelArr, wrong_labelArr, labelArr = [], [], []
    precision, recall = 0, 0
    for batch_idx, (index, (_, labels)) in enumerate(unlabeledloader):
        # labels = lab_conv(knownclass, labels)
        for i, label in enumerate(labels):
            # print('index[i]: {}'.format(index[i]))
            # print('label: {}'.format(label))
            # print('knownclass: {}'.format(knownclass))
            if label in knownclass:
                correct_queryIndex += [index[i]]
                correct_labelArr += [label]
            else:
                wrong_queryIndex += [index[i]]
                wrong_labelArr += [label]
        queryIndex += index
        labelArr += list(np.array(labels.cpu().data))

    # print('correct_queryIndex shape: {}'.format(np.shape(correct_queryIndex)))
    # print('correct_labelArr shape: {}'.format(np.shape(correct_labelArr)))
    correct_queryIndex, correct_labelArr = shuffle(correct_queryIndex, correct_labelArr)
    correct_queried_indeces = correct_queryIndex[:int(args.query_batch*args.custom_precision)]
    # print(np.shape(correct_queried_indeces))
    correct_queried_labels = correct_labelArr[:int(args.query_batch*args.custom_precision)]
    # print(np.shape(correct_queried_labels))
    # np.random.shuffle(tmp_correct_data)
    # tmp_correct_data = tmp_correct_data.T
    # correct_queried_indeces = tmp_correct_data[0][:int(args.query_batch*args.custom_precision)]
    # correct_queried_labels = tmp_correct_data[1][:int(args.query_batch*args.custom_precision)]

    
    wrong_queryIndex, wrong_labelArr = shuffle(wrong_queryIndex, wrong_labelArr)
    wrong_queried_indeces = wrong_queryIndex[:int(args.query_batch*(1-args.custom_precision))]
    wrong_queried_labels = wrong_labelArr[:int(args.query_batch*(1-args.custom_precision))]
    # tmp_wrong_data = np.vstack((wrong_queryIndex, wrong_labelArr)).T
    # np.random.shuffle(tmp_wrong_data)
    # tmp_wrong_data = tmp_wrong_data.T
    # wrong_queried_indeces = tmp_wrong_data[0][:int(args.query_batch*(1-args.custom_precision))]
    # wrong_queried_labels = tmp_wrong_data[1][:int(args.query_batch*(1-args.custom_precision))]

    # print(np.shape(correct_queried_labels))
    # print(np.shape(wrong_queried_labels))
    queryLabelArr = np.hstack([correct_queried_labels, wrong_queried_labels])
    queryLabelArr = lab_conv(knownclass, torch.from_numpy(queryLabelArr))
    # print(np.shape(queryLabelArr))
    labelArr = np.array(labelArr).astype(int)
    labelArr = lab_conv(knownclass, torch.from_numpy(labelArr))
    # print(np.shape(labelArr))

    # print(np.where(queryLabelArr < args.known_class)[0])
    # print(len(np.where(queryLabelArr < args.known_class)[0]))
    # print(len(queryLabelArr))
    precision = len(np.where(queryLabelArr < args.known_class)[0]) / len(queryLabelArr)
    recall = (len(np.where(queryLabelArr < args.known_class)[0]) + Len_labeled_ind_train) / (len(np.where(labelArr < args.known_class)[0]) + Len_labeled_ind_train)
    # print('precision: {}'.format(precision))
    # print('recall: {}'.format(recall))
    return correct_queried_indeces, wrong_queried_indeces, precision, recall



def custom_precision_mem(args, unlabeledloader, Len_labeled_ind_train, knownclass):
    correct_queryIndex, wrong_queryIndex, queryIndex = {}, {}, []
    correct_labelArr, wrong_labelArr, labelArr = {}, {}, []
    precision, recall = 0, 0
    mem_scores = Proxy(mode='mem', dataset=args.dataset).get_dict_form(sort=False)
    correct_mem_scores = {}
    wrong_mem_scores = {}
    for batch_idx, (index, (_, labels)) in enumerate(unlabeledloader):
        # labels = lab_conv(knownclass, labels)
        for i, label in enumerate(labels):
            if label in knownclass:
                correct_queryIndex[index[i]] = index[i]
                correct_labelArr[index[i]] = label
                correct_mem_scores[index[i]] = mem_scores[index[i]]
            else:
                wrong_queryIndex[index[i]] = index[i]
                wrong_labelArr[index[i]] = label
                wrong_mem_scores[index[i]] = mem_scores[index[i]]
        queryIndex += index
        labelArr += list(np.array(labels.cpu().data))

    # tmp_correct_data = np.vstack((correct_queryIndex, correct_labelArr)).T
    sorted_items = sorted(correct_mem_scores.items(), key=lambda t: t[1])
    # print('sorted_items: {}'.format(sorted_items))
    queryIndex = list([item[0] for item in sorted_items][-int(args.query_batch*args.custom_precision):])
    correct_queried_indeces = [correct_queryIndex[ind] for ind in queryIndex]
    correct_queried_labels = [correct_labelArr[ind] for ind in queryIndex]
    # print('correct_queried_indeces: {}'.format(correct_queried_indeces))
    # print('correct_queried_labels: {}'.format(correct_queried_labels))

    # tmp_wrong_data = np.vstack((wrong_queryIndex, wrong_labelArr)).T
    sorted_items = sorted(wrong_mem_scores.items(), key=lambda t: t[1])
    queryIndex = list([item[0] for item in sorted_items][-int(args.query_batch*(1-args.custom_precision)):])
    wrong_queried_indeces = [wrong_queryIndex[ind] for ind in queryIndex]
    wrong_queried_labels = [wrong_labelArr[ind] for ind in queryIndex]
    # print('wrong_queried_indeces: {}'.format(wrong_queried_indeces))
    # print('wrong_queried_labels: {}'.format(wrong_queried_labels))

    # print(np.shape(correct_queried_labels))
    # print(np.shape(wrong_queried_labels))
    queryLabelArr = np.hstack([correct_queried_labels, wrong_queried_labels])
    queryLabelArr = lab_conv(knownclass, torch.from_numpy(queryLabelArr))
    labelArr = np.array(labelArr).astype(int)
    labelArr = lab_conv(knownclass, torch.from_numpy(labelArr))

    precision = len(np.where(queryLabelArr < args.known_class)[0]) / len(queryLabelArr)
    recall = (len(np.where(queryLabelArr < args.known_class)[0]) + Len_labeled_ind_train) / (len(np.where(labelArr < args.known_class)[0]) + Len_labeled_ind_train)
    # print('precision: {}'.format(precision))
    # print('recall: {}'.format(recall))
    return correct_queried_indeces, wrong_queried_indeces, precision, recall
