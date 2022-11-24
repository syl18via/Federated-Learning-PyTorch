#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6


from unicodedata import numeric
import numpy as np
from torchvision import datasets, transforms

def mnist_iid(dataset, num_users):
    """
    Sample I.I.D. client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    num_items = int(len(dataset)/num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items,
                                             replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users


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
    dict_users = {i: np.array([]) for i in range(num_users)}
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
            dict_users[i] = np.concatenate(
                (dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]), axis=0)
    return dict_users

def mnist_noniid_v2(dataset, num_users):
    """
    Sample non-I.I.D client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return:
    """
    # 60,000 training imgs -->  200 imgs/shard X 300 shards
    dict_users = {i: np.array([]) for i in range(num_users)}
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
                dict_users[i] = np.concatenate((dict_users[i],
                    np.random.choice(sample_idxs_of_this_class, sample_num_per_major_class, replace=False)), axis=0)
            else:
                ### In minor class
                dict_users[i] = np.concatenate((dict_users[i],
                    np.random.choice(sample_idxs_of_this_class, sample_num_per_minor_class, replace=False)), axis=0)

    return dict_users

def mnist_noniid(dataset, num_users):
    # return mnist_noniid_v1(dataset, num_users)
    return mnist_noniid_v2(dataset, num_users)

def mnist_iid_noniid(dataset, num_users):
    dict_iid = mnist_iid(dataset, num_users)
    dict_non_iid = mnist_noniid(dataset, num_users)
    dict_users = {i: np.array([]) for i in range(num_users)}
    for i in range(num_users):
        if i < num_users/2:
            dict_users[i] = dict_iid[i]
        else:
            dict_users[i] = dict_non_iid[i]
    return dict_users




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
    dict_users = {i: np.array([]) for i in range(num_users)}
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
                dict_users[i] = np.concatenate(
                    (dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]),
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
                dict_users[i] = np.concatenate(
                    (dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]),
                    axis=0)
    else:

        for i in range(num_users):
            shard_size = random_shard_size[i]
            rand_set = set(np.random.choice(idx_shard, shard_size,
                                            replace=False))
            idx_shard = list(set(idx_shard) - rand_set)
            for rand in rand_set:
                dict_users[i] = np.concatenate(
                    (dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]),
                    axis=0)

        if len(idx_shard) > 0:
            # Add the leftover shards to the client with minimum images:
            shard_size = len(idx_shard)
            # Add the remaining shard to the client with lowest data
            k = min(dict_users, key=lambda x: len(dict_users.get(x)))
            rand_set = set(np.random.choice(idx_shard, shard_size,
                                            replace=False))
            idx_shard = list(set(idx_shard) - rand_set)
            for rand in rand_set:
                dict_users[k] = np.concatenate(
                    (dict_users[k], idxs[rand*num_imgs:(rand+1)*num_imgs]),
                    axis=0)

    return dict_users


def cifar_iid(dataset, num_users):
    """
    Sample I.I.D. client data from CIFAR10 dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    num_items = int(len(dataset)/num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items,
                                             replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users


def cifar_noniid(dataset, num_users):
    """
    Sample non-I.I.D client data from CIFAR10 dataset
    :param dataset:
    :param num_users:
    :return:
    """
    num_shards, num_imgs = 200, 250
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([]) for i in range(num_users)}
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
            dict_users[i] = np.concatenate(
                (dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]), axis=0)
    return dict_users

client_label_dist = [
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
]

client_data_num = [1200*5] * 10
CLASS_NUM = 10

def mnist_noniid_v3(dataset, num_users):
    """
    Sample non-I.I.D client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return:
    """
    # 60,000 training imgs -->  200 imgs/shard X 300 shards
    dict_users = {i: np.array([]) for i in range(num_users)}
    # idxs = np.arange(len(dataset))
    labels = dataset.train_labels.numpy()


    # Group labels
    label2idxs = []
    for digit in range(CLASS_NUM):
        indexs = [i for i, label in enumerate(labels) if label == digit]
        # indexs = indexs[0:5421]
        label2idxs.append(indexs)

    
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
                dict_users[i] = np.concatenate((dict_users[i],
                    np.random.choice(sample_idxs_of_this_class, sample_num_per_major_class, replace=False)), axis=0)
            else:
                ### In minor class
                dict_users[i] = np.concatenate((dict_users[i],
                    np.random.choice(sample_idxs_of_this_class, sample_num_per_minor_class, replace=False)), axis=0)

    return dict_users
def get_dataset(args):
    """ Returns train and test datasets and a user group which is a dict where
    the keys are the user index and the values are the corresponding data for
    each of those users.
    """

    if args.dataset == 'cifar':
        raise
        data_dir = '../data/cifar/'
        apply_transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        train_dataset = datasets.CIFAR10(data_dir, train=True, download=True,
                                       transform=apply_transform)

        test_dataset = datasets.CIFAR10(data_dir, train=False, download=True,
                                      transform=apply_transform)
                                      

        # sample training data amongst users
        if args.iid:
            # Sample IID user data from Mnist
            user_groups = cifar_iid(train_dataset, args.num_users)
        else:
            # Sample Non-IID user data from Mnist
            if args.unequal:
                # Chose uneuqal splits for every user
                raise NotImplementedError()
            else:
                # Chose euqal splits for every user
                user_groups = cifar_noniid(train_dataset, args.num_users)

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
        if args.halfiid:
            user_groups = mnist_iid_noniid(train_dataset,args.num_users)
        elif args.iid:
            # Sample IID user data from Mnist
            user_groups = mnist_iid(train_dataset, args.num_users)
        else:
            # Sample Non-IID user data from Mnist
            if args.unequal:
                # Chose uneuqal splits for every user
                user_groups = mnist_noniid_unequal(train_dataset, args.num_users)
            else:
                # Chose euqal splits for every user
                user_groups = mnist_noniid(train_dataset, args.num_users)

    return train_dataset, test_dataset, user_groups


if __name__ == '__main__':
    dataset_train = datasets.MNIST('./data/mnist/', train=True, download=True,
                                   transform=transforms.Compose([
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.1307,),
                                                            (0.3081,))
                                   ]))
    num = 100
    d = mnist_noniid(dataset_train, num)
