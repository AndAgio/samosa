from .almethod import ALMethod
import torch
import numpy as np

class Coreset(ALMethod):
    def __init__(self, args, models, unlabeled_dst, U_index, I_index, **kwargs):
        super().__init__(args, models, unlabeled_dst, U_index, **kwargs)
        self.I_index = I_index
        self.labeled_in_set = torch.utils.data.Subset(self.unlabeled_dst, self.I_index)

    def get_features(self):
        self.models['backbone'].eval()
        labeled_features, unlabeled_features = None, None
        with torch.no_grad():
            labeled_in_loader = torch.utils.data.DataLoader(self.labeled_in_set, batch_size=self.args.test_batch_size, num_workers=self.args.workers)
            unlabeled_loader = torch.utils.data.DataLoader(self.unlabeled_set, batch_size=self.args.test_batch_size, num_workers=self.args.workers)

            # generate entire labeled_in features set
            for data in labeled_in_loader:
                # inputs = data[0].to(self.args.gpu)
                inputs = data[1][0].to(self.args.gpu)
                out, features = self.models['backbone'](inputs)

                if labeled_features is None:
                    labeled_features = features
                else:
                    labeled_features = torch.cat((labeled_features, features), 0)

            # generate entire unlabeled features set
            for data in unlabeled_loader:
                # inputs = data[0].to(self.args.gpu)
                inputs = data[1][0].to(self.args.gpu)
                out, features = self.models['backbone'](inputs)
                if unlabeled_features is None:
                    unlabeled_features = features
                else:
                    unlabeled_features = torch.cat((unlabeled_features, features), 0)
        return labeled_features, unlabeled_features

    def k_center_greedy(self, labeled, unlabeled, n_query):
        # get the minimum distances between the labeled and unlabeled examples (iteratively, to avoid memory issues):
        min_dist = torch.min(torch.cdist(labeled[0:2, :], unlabeled), 0).values
        for j in range(2, labeled.shape[0], 100):
            if j + 100 < labeled.shape[0]:
                dist_matrix = torch.cdist(labeled[j:j + 100, :], unlabeled)
            else:
                dist_matrix = torch.cdist(labeled[j:, :], unlabeled)
            min_dist = torch.stack((min_dist, torch.min(dist_matrix, 0).values))
            min_dist = torch.min(min_dist, 0).values

        min_dist = min_dist.reshape((1, min_dist.size(0)))
        farthest = torch.argmax(min_dist)

        greedy_indices = torch.tensor([farthest])
        for i in range(n_query - 1):
            dist_matrix = torch.cdist(unlabeled[greedy_indices[-1], :].reshape((1, -1)), unlabeled)
            min_dist = torch.stack((min_dist, dist_matrix))
            min_dist = torch.min(min_dist, 0).values

            farthest = torch.tensor([torch.argmax(min_dist)])
            greedy_indices = torch.cat((greedy_indices, farthest), 0)

        return greedy_indices.cpu().numpy()

    def select(self, **kwargs):
        labeled_features, unlabeled_features = self.get_features()
        selected_indices = self.k_center_greedy(labeled_features, unlabeled_features, self.args.query_batch)
        scores = list(np.ones(len(selected_indices))) # equally assign 1 (meaningless)

        Q_index = [self.U_index[idx] for idx in selected_indices]

        return Q_index, scores