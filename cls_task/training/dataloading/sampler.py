import math
import numpy as np

import torch
from torch.utils.data import Sampler
import torch.distributed as dist

from torch.utils.data.sampler import WeightedRandomSampler

class DistributedWeightedSampler(Sampler):
    def __init__(self, dataset, num_replicas=None, rank=None, replacement=True, shuffle=True):
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.num_samples = int(math.ceil(len(self.dataset) * 1.0 / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas
        self.replacement = replacement
        self.shuffle = shuffle

    def calculate_weights(self, targets):
        class_sample_count = torch.tensor(
            [(targets == t).sum() for t in np.unique(targets)]
        )
        weight = 1. / class_sample_count.double()
        samples_weight = torch.tensor([weight[t] for t in targets])
        return samples_weight

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.epoch)
        if self.shuffle:
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = list(range(len(self.dataset)))

        indices += indices[:(self.total_size - len(indices))]
        assert len(indices) == self.total_size

        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples

        targets = self.dataset.targets
        targets = torch.tensor(targets)[indices]
        assert len(targets) == self.num_samples
        weights = self.calculate_weights(targets)
        subsample_balanced_indices = torch.multinomial(weights, self.num_samples, self.replacement)
        dataset_indices = torch.tensor(indices)[subsample_balanced_indices]
        return iter(dataset_indices.tolist())

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch


    def get_weighted_random_sampler(dataset):
        targets = dataset.targets
        class_sample_count = torch.tensor([targets == t].sum() for t in np.unique(targets))
        weight = 1. / class_sample_count.double()
        samples_weight = torch.tensor([weight[t] for t in targets])
        return WeightedRandomSampler(samples_weight, len(samples_weight))