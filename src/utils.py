#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import copy
import torch
import numpy as np

from torch.utils.data import Dataset

class DatasetSplit(Dataset):
    """An abstract Dataset class wrapped around Pytorch Dataset class.
    """

    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = [int(i) for i in idxs]

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return torch.tensor(image), torch.tensor(label)

class DatasetRelabel(DatasetSplit):
    """An abstract Dataset class wrapped around Pytorch Dataset class.
    """

    def __init__(self, dataset, idxs, target_labels=None):
        super(DatasetRelabel, self).__init__(dataset, idxs)

        self.target_labels = target_labels

        ### Mapping from original label to new label
        if target_labels is not None:
            ### Mapping to continuous new labels
            ### Key: orinal label, Value: new label
            self.to_new_label_dict = dict([(target_label, i) for i, target_label in enumerate(self.target_labels)])
            self.minor_class_label = len(self.target_labels)
        else:
            self.to_new_label_dict = self.minor_class_label = None

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        if self.target_labels is not None:
            label = int(label)
            if label not in self.target_labels:
                label = self.minor_class_label ### Relabel as minor class\
            else:
                label = self.to_new_label_dict[int(label)]
        return torch.tensor(image), torch.tensor(label)

# class DatasetLabelSpecific(DatasetSplit):
#     """An abstract Dataset class wrapped around Pytorch Dataset class.
#     """

#     def __init__(self, dataset, target_labels):
#         self.target_labels = target_labels
#         if isinstance(dataset, DatasetSplit):
#             self.dataset = dataset.dataset
#             idx_of_split = []
#             for i in range(len(dataset)):
#                 _, tenser_y = dataset[i]
#                 if tenser_y in target_labels:
#                     idx_of_split.append(i)
#             self.idxs = np.array(dataset.idxs)[idx_of_split]
#         else:
#             self.dataset = dataset
#             target_idx = []
#             for i in range(len(dataset)):
#                 _, tenser_y = dataset[i]
#                 if tenser_y in target_labels:
#                     target_idx.append(i)
#             self.idxs = [int(i) for i in target_idx]


def average_weights(w):
    """
    Returns the average of the weights.
    """
    if len(w) == 1:
        return w[0]
    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[key] += w[i][key]
        w_avg[key] = torch.div(w_avg[key], len(w))
    return w_avg


def exp_details(args):
    print('\nExperimental details:')
    print(f'    Model     : {args.model}')
    print(f'    Optimizer : {args.optimizer}')
    print(f'    Learning  : {args.lr}')
    print(f'    Global Rounds   : {args.epochs}\n')

    print('    Federated parameters:')
    if args.iid:
        print('    IID')
    else:
        print('    Non-IID')
    print(f'    Fraction of users  : {args.frac}')
    print(f'    Local Batch size   : {args.local_bs}')
    print(f'    Local Epochs       : {args.local_ep}\n')
    return
