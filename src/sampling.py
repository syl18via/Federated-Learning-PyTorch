#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import numpy as np
from torchvision import datasets, transforms
from typing import Union, Dict
from collections import Counter

def check_dist(name, _dataset):
    _, lables = zip(*list((_dataset)))
    lables = [int(x) for x in lables]
    counter = Counter(lables)
    print(f"{name}, distribution: {counter}")
    return counter

def mnist_iid(dataset, num_users):
    """
    Sample I.I.D. client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    num_items = int(len(dataset)/num_users)
    client2dataidxs, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        client2dataidxs[i] = set(np.random.choice(all_idxs, num_items,
                                             replace=False))
        all_idxs = list(set(all_idxs) - client2dataidxs[i])
    return client2dataidxs


def mnist_noniid_v1(dataset, num_users):
    """
    Sample non-I.I.D client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return:
    """
    # 60,000 training imgs -->  200 imgs/shard X 300 shards
    num_shards, num_imgs = 200, 300
    idx_shard = [i for i in range(num_shards)]
    client2dataidxs = {i: np.array([]) for i in range(num_users)}
    idxs = np.arange(num_shards*num_imgs)
    labels = dataset.train_labels.numpy()

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]

    # divide and assign 2 shards/client
    for i in range(num_users):
        rand_set = set(np.random.choice(idx_shard, 2, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            client2dataidxs[i] = np.concatenate(
                (client2dataidxs[i], idxs[rand*num_imgs:(rand+1)*num_imgs]), axis=0)
    return client2dataidxs

def mnist_noniid_v2(dataset, num_users):
    """
    Sample non-I.I.D client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return:
    """
    # 60,000 training imgs -->  200 imgs/shard X 300 shards
    client2dataidxs = {i: np.array([]) for i in range(num_users)}
    # idxs = np.arange(len(dataset))
    labels = dataset.train_labels.numpy()

    CLASS_NUM = 10
    MAJOR2MINOR_SAMPLE_RATIO = 99
    MAJOR_CLASS_NUM = 5
    total_sample_num = len(labels)
    sample_num_per_client = total_sample_num // num_users
    sample_num_per_major_class = int(MAJOR2MINOR_SAMPLE_RATIO * sample_num_per_client / (MAJOR2MINOR_SAMPLE_RATIO * MAJOR_CLASS_NUM + 1 * (CLASS_NUM - MAJOR_CLASS_NUM)))
    sample_num_per_minor_class = int(1 * sample_num_per_client / (MAJOR2MINOR_SAMPLE_RATIO * MAJOR_CLASS_NUM + 1 * (CLASS_NUM - MAJOR_CLASS_NUM)))

    # Group labels
    label2idxs = []
    for digit in range(CLASS_NUM):
        indexs = [i for i, label in enumerate(labels) if label == digit]
        # indexs = indexs[0:5421]
        label2idxs.append(indexs)

    ### Construct an imbalanced dataset
    ### For each client, randomly select 2 labels as the major class
    for i in range(num_users):

        if i in [0, 1]:
            major_class = list(range(5))
        elif i in [2, 3]:
            major_class = list(range(5, 10))
        else:
            ### make sure the major classes do not belong to [0, 5) or [5, 10) at the same time
            major_class1 = np.random.choice(int(CLASS_NUM/2)-2, 
                int(MAJOR_CLASS_NUM/2), replace=False)
            major_class2 = np.random.choice(int(CLASS_NUM/2)-2,
                MAJOR_CLASS_NUM-int(MAJOR_CLASS_NUM/2), replace=False) + int(CLASS_NUM/2)
            major_class = list(major_class1) + list(major_class2)
            
        # major_class = np.random.choice(CLASS_NUM, MAJOR_CLASS_NUM, replace=False)
        # major_class = [0, 1]

        for _class in range(CLASS_NUM):
            sample_idxs_of_this_class = label2idxs[_class]
            if _class in major_class:
                client2dataidxs[i] = np.concatenate((client2dataidxs[i],
                    np.random.choice(sample_idxs_of_this_class, sample_num_per_major_class, replace=False)), axis=0)
            else:
                ### In minor class
                client2dataidxs[i] = np.concatenate((client2dataidxs[i],
                    np.random.choice(sample_idxs_of_this_class, sample_num_per_minor_class, replace=False)), axis=0)

    return client2dataidxs

CLIENT_DATA_DIST = np.array([
    #  0,  1,  2,   3,   4,   5,  6,   7,    8,  9
    [1.4, 1.4, 30, 1.4, 1.4, 30, 1.4, 1.4, 1.6, 30],#client0
    [1.4, 1.4, 1.4, 30, 1.4, 1.4, 30, 1.4, 1.6, 30],
    [1.4, 1.4, 30, 1.4, 1.4, 1.4, 1.4, 30, 30, 1.6],
    [1.4, 1.4, 30, 1.4, 1.4, 1.4, 1.4, 30, 30, 1.6],
    [30 ,  30, 1.4,1.4,  30, 1.4, 1.4,1.4,1.4, 1.6],
    [1.4, 30, 1.4, 1.4, 30,  30, 1.4, 1.4, 1.4, 1.6],
    [1.4, 1.4, 1.4, 30, 1.4, 1.4, 30, 1.4, 1.6, 30],
    [30, 1.4, 1.4, 30, 1.4, 1.4, 1.4, 30, 1.4, 1.6],
    [1.4, 1.4, 1.4, 1.4, 1.4, 30, 30, 1.4, 30, 1.6],
    [30 ,  30, 1.4,1.4,  30, 1.4, 1.4,1.4,1.4, 1.6]
])

CLIENT_NUM = CLIENT_DATA_DIST.shape[0]
CLASS_NUM = CLIENT_DATA_DIST.shape[1]

def load_custom_dataset(dataset, num_users):
    """
    Sample non-I.I.D client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return:
    """

    counter = check_dist("Origin Dataset", dataset)

    total_data_size = sum(counter.values())

    ### Assume each client has the same size of data
    data_size_per_client = int(total_data_size / CLIENT_NUM)
    CLIENT_DATA_NUM = [data_size_per_client] * CLIENT_NUM

    client2dataidxs: Dict[int, np.ndarray] = {i: np.array([]) for i in range(num_users)}

    # Group labels
    label2idxs = {}
    for idx, (_, y) in enumerate(dataset):
        if y not in label2idxs:
            label2idxs[y] = []
        label2idxs[y].append(idx)

    ### When traversing all data by label, store the current 
    # position of the idx list of each label
    label2bias = [0] * CLASS_NUM
    
    for client_id in range(num_users):

        _dist: np.ndarray = CLIENT_DATA_DIST[client_id]
        _data_num: int = CLIENT_DATA_NUM[client_id]
        _data_num_per_label: np.ndarray = _dist * _data_num / 100

        for _class_id in range(CLASS_NUM):
            all_data_idxs_of_this_class: list = label2idxs[_class_id]
            required_data_num_this_class: int = int(_data_num_per_label[_class_id])
            required_data_idxs = all_data_idxs_of_this_class[label2bias[_class_id]:(label2bias[_class_id]+required_data_num_this_class)]
            label2bias[_class_id] += required_data_num_this_class
            client2dataidxs[client_id] = np.concatenate((client2dataidxs[client_id], required_data_idxs), axis=0)

    return client2dataidxs

def mnist_noniid(dataset, num_users):
    # return mnist_noniid_v1(dataset, num_users)
    return mnist_noniid_v2(dataset, num_users)

def mnist_iid_noniid(dataset, num_users):
    dict_iid = mnist_iid(dataset, num_users)
    dict_non_iid = mnist_noniid(dataset, num_users)
    client2dataidxs = {i: np.array([]) for i in range(num_users)}
    for i in range(num_users):
        if i < num_users/2:
            client2dataidxs[i] = dict_iid[i]
        else:
            client2dataidxs[i] = dict_non_iid[i]
    return client2dataidxs

def mnist_noniid_unequal(dataset, num_users):
    """
    Sample non-I.I.D client data from MNIST dataset s.t clients
    have unequal amount of data
    :param dataset:
    :param num_users:
    :returns a dict of clients with each clients assigned certain
    number of training imgs
    """
    # 60,000 training imgs --> 50 imgs/shard X 1200 shards
    num_shards, num_imgs = 1200, 50
    idx_shard = [i for i in range(num_shards)]
    client2dataidxs = {i: np.array([]) for i in range(num_users)}
    idxs = np.arange(num_shards*num_imgs)
    labels = dataset.train_labels.numpy()

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]

    # Minimum and maximum shards assigned per client:
    min_shard = 1
    max_shard = 30

    # Divide the shards into random chunks for every client
    # s.t the sum of these chunks = num_shards
    random_shard_size = np.random.randint(min_shard, max_shard+1,
                                          size=num_users)
    random_shard_size = np.around(random_shard_size /
                                  sum(random_shard_size) * num_shards)
    random_shard_size = random_shard_size.astype(int)

    # Assign the shards randomly to each client
    if sum(random_shard_size) > num_shards:

        for i in range(num_users):
            # First assign each client 1 shard to ensure every client has
            # atleast one shard of data
            rand_set = set(np.random.choice(idx_shard, 1, replace=False))
            idx_shard = list(set(idx_shard) - rand_set)
            for rand in rand_set:
                client2dataidxs[i] = np.concatenate(
                    (client2dataidxs[i], idxs[rand*num_imgs:(rand+1)*num_imgs]),
                    axis=0)

        random_shard_size = random_shard_size-1

        # Next, randomly assign the remaining shards
        for i in range(num_users):
            if len(idx_shard) == 0:
                continue
            shard_size = random_shard_size[i]
            if shard_size > len(idx_shard):
                shard_size = len(idx_shard)
            rand_set = set(np.random.choice(idx_shard, shard_size,
                                            replace=False))
            idx_shard = list(set(idx_shard) - rand_set)
            for rand in rand_set:
                client2dataidxs[i] = np.concatenate(
                    (client2dataidxs[i], idxs[rand*num_imgs:(rand+1)*num_imgs]),
                    axis=0)
    else:

        for i in range(num_users):
            shard_size = random_shard_size[i]
            rand_set = set(np.random.choice(idx_shard, shard_size,
                                            replace=False))
            idx_shard = list(set(idx_shard) - rand_set)
            for rand in rand_set:
                client2dataidxs[i] = np.concatenate(
                    (client2dataidxs[i], idxs[rand*num_imgs:(rand+1)*num_imgs]),
                    axis=0)

        if len(idx_shard) > 0:
            # Add the leftover shards to the client with minimum images:
            shard_size = len(idx_shard)
            # Add the remaining shard to the client with lowest data
            k = min(client2dataidxs, key=lambda x: len(client2dataidxs.get(x)))
            rand_set = set(np.random.choice(idx_shard, shard_size,
                                            replace=False))
            idx_shard = list(set(idx_shard) - rand_set)
            for rand in rand_set:
                client2dataidxs[k] = np.concatenate(
                    (client2dataidxs[k], idxs[rand*num_imgs:(rand+1)*num_imgs]),
                    axis=0)

    return client2dataidxs


def cifar_iid(dataset, num_users):
    """
    Sample I.I.D. client data from CIFAR10 dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    num_items = int(len(dataset)/num_users)
    client2dataidxs, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        client2dataidxs[i] = set(np.random.choice(all_idxs, num_items,
                                             replace=False))
        all_idxs = list(set(all_idxs) - client2dataidxs[i])
    return client2dataidxs

def cifar_noniid(dataset, num_users):
    """
    Sample non-I.I.D client data from CIFAR10 dataset
    :param dataset:
    :param num_users:
    :return:
    """
    num_shards, num_imgs = 200, 250
    idx_shard = [i for i in range(num_shards)]
    client2dataidxs = {i: np.array([]) for i in range(num_users)}
    idxs = np.arange(num_shards*num_imgs)
    # labels = dataset.train_labels.numpy()
    labels = np.array(dataset.targets)

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]

    # divide and assign
    for i in range(num_users):
        rand_set = set(np.random.choice(idx_shard, 2, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            client2dataidxs[i] = np.concatenate(
                (client2dataidxs[i], idxs[rand*num_imgs:(rand+1)*num_imgs]), axis=0)
    return client2dataidxs


def get_dataset(args):
    """ Returns train and test datasets and a user group which is a dict where
    the keys are the user index and the values are the corresponding data for
    each of those users.
    """

    if args.dataset == 'cifar':
        data_dir = '../data/cifar/'
        apply_transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        train_dataset = datasets.CIFAR10(data_dir, train=True, download=True,
                                       transform=apply_transform)

        test_dataset = datasets.CIFAR10(data_dir, train=False, download=True,
                                      transform=apply_transform)

        # check_dist("Cifar Train", train_dataset)
        # check_dist("Cifar Test", test_dataset)           

        # sample training data amongst users
        if True:
            client2dataidxs = load_custom_dataset(dataset=train_dataset, num_users=args.num_users)
        elif args.iid:
            # Sample IID user data from Mnist
            client2dataidxs = cifar_iid(train_dataset, args.num_users)
        else:
            # Sample Non-IID user data from Mnist
            if args.unequal:
                # Chose uneuqal splits for every user
                raise NotImplementedError()
            else:
                # Chose euqal splits for every user
                client2dataidxs = cifar_noniid(train_dataset, args.num_users)

    elif args.dataset == 'mnist' or 'fmnist':
        if args.dataset == 'mnist':
            data_dir = '../data/mnist/'
        else:
            data_dir = '../data/fmnist/'

        apply_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))])

        train_dataset = datasets.MNIST(data_dir, train=True, download=True,
                                       transform=apply_transform)

        test_dataset = datasets.MNIST(data_dir, train=False, download=True,
                                      transform=apply_transform)

        # sample training data amongst users
        if True:
            client2dataidxs = load_custom_dataset(dataset=train_dataset, num_users=args.num_users)
        elif args.halfiid:
            client2dataidxs = mnist_iid_noniid(train_dataset, args.num_users)
        elif args.iid:
            # Sample IID user data from Mnist
            client2dataidxs = mnist_iid(train_dataset, args.num_users)
        else:
            # Sample Non-IID user data from Mnist
            if args.unequal:
                # Chose uneuqal splits for every user
                client2dataidxs = mnist_noniid_unequal(train_dataset, args.num_users)
            else:
                # Chose euqal splits for every user
                client2dataidxs = mnist_noniid(train_dataset, args.num_users)

    return train_dataset, test_dataset, client2dataidxs


if __name__ == '__main__':
    dataset_train = datasets.MNIST('./data/mnist/', train=True, download=True,
                                   transform=transforms.Compose([
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.1307,),
                                                            (0.3081,))
                                   ]))
    num = 100
    d = mnist_noniid(dataset_train, num)
