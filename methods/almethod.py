import torch

class ALMethod(object):
    def __init__(self, args, models, unlabeled_dst, U_index, **kwargs):
        self.unlabeled_dst = unlabeled_dst
        self.U_index = U_index
        self.unlabeled_set = torch.utils.data.Subset(unlabeled_dst, U_index)
        self.n_unlabeled = len(self.unlabeled_set)
        self.num_classes = 10 if args.dataset == 'cifar10' else 100 if args.dataset == 'cifar100' else 200 if args.dataset == 'tinyimagenet' else None
        if self.num_classes is None:
            raise ValueError('Dataset not found!')
        self.models = models
        self.index = []
        self.args = args

    def select(self, **kwargs):
        return