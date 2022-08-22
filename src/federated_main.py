#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6


import os
import copy
import time
import pickle
import numpy as np
from tqdm import tqdm

import torch
from tensorboardX import SummaryWriter

from options import args_parser
from update import LocalUpdate, test_inference
from models import MLP, CNNMnist, CNNFashion_Mnist, CNNCifar
from utils import get_dataset, average_weights, exp_details

class ClientProj:
    def __init__(self, num_users):
        self.client2proj = np.array([-math.inf] * num_users)
    
    def update_proj_list(self, idxs_users, global_weights, global_weights_before, local_weights):
        #calculate projection of client local gradient on global gradient
        global_grad={}
        for key in global_weights.keys():
            global_grad[key] = (global_weights[key]- global_weights_before[key]).data.cpu().numpy()

        clientid_to_grad = {}
        for i, idx in enumerate(idxs_users):
            clientid_to_grad[idx] = {}
            for key in local_weights[i].keys():
                clientid_to_grad[idx][key] = (local_weights[i][key]- global_weights_before[key]).data.cpu().numpy()

        # clientID2proj = {}
        for idx in idxs_users:
            #### Method 2
            proj_dict = {}
            for key in global_weights.keys():
                _global_grad = global_grad[key].flatten()
                g_norm = np.sqrt(sum(_global_grad**2))
                # print(type(g_norm), g_norm.shape)
                local_grad = clientid_to_grad[idx][key].flatten()
                try:
                    proj_dict[key]= np.dot(local_grad, _global_grad) / g_norm
                except:
                    import code; code.interact(local=locals())
            # clientID2proj[idx] = np.array(list(proj_dict.values())).mean()
            self.client2proj[idx] = np.array(list(proj_dict.values())).mean()
        # print('clientID2proj', clientID2proj)

if __name__ == '__main__':
    start_time = time.time()

    # define paths
    path_project = os.path.abspath('..')
    logger = SummaryWriter('../logs')

    args = args_parser()
    exp_details(args)

    if args.gpu:
        torch.cuda.set_device(args.gpu)
    device = 'cuda' if args.gpu else 'cpu'

    # load dataset and user groups
    train_dataset, test_dataset, user_groups = get_dataset(args)

    # BUILD MODEL
    if args.model == 'cnn':
        # Convolutional neural netork
        if args.dataset == 'mnist':
            global_model = CNNMnist(args=args)
        elif args.dataset == 'fmnist':
            global_model = CNNFashion_Mnist(args=args)
        elif args.dataset == 'cifar':
            global_model = CNNCifar(args=args)

    elif args.model == 'mlp':
        # Multi-layer preceptron
        img_size = train_dataset[0][0].shape
        len_in = 1
        for x in img_size:
            len_in *= x
            global_model = MLP(dim_in=len_in, dim_hidden=64,
                               dim_out=args.num_classes)
    else:
        exit('Error: unrecognized model')

    # Set the model to train and send it to device.
    # print(args.model)
    # print(args.dataset)
    global_model.to(device)
    global_model.train()
    # print(global_model)

    # copy weights
    global_weights = global_model.state_dict()

    # Training
    train_loss, train_accuracy = [], []
    val_acc_list, net_list = [], []
    cv_loss, cv_acc = [], []
    print_every = 10
    val_loss_pre, counter = 0, 0

    import math
    client_proj = ClientProj(args.num_users)

    # client 选择函数，返回idxs_users，表示被选到的client的ID， 对的，
    # 然后后面变复杂之后，这里的选择函数的输入参数会更加复杂，要结合我们统计的momentum-based gradient projection 选择
    def momemtum_based(num_users, client_proj):
        momemtum_based_grad_proj = client_proj.client2proj
        ''' momemtum_based_grad_proj 是一个list，长度等于 总的client数量，挑出momemtum_based_grad_proj最小的num_users client
        '''
        assert isinstance(momemtum_based_grad_proj, list) or isinstance(momemtum_based_grad_proj, np.ndarray)
        assert len(momemtum_based_grad_proj) == args.num_users
        momemtum_based_grad_proj = np.array(momemtum_based_grad_proj)
        return momemtum_based_grad_proj.argsort()[-num_users:][::-1]
    
    def update_client_idx(use_all_users):
        m = max(int(args.frac * args.num_users), 1)
        if args.policy == "random":
            idxs_users = np.random.choice(range(args.num_users), m, replace=False) ### 这句话在选择client， 诶但是他已经实现了，每一个global step update 一次
        elif args.policy == "momentum":
            if use_all_users == True :
                idxs_users = momemtum_based(args.num_users, client_proj)
            else:
                idxs_users = momemtum_based(m, client_proj)
        elif args.policy == "debug":
            idxs_users = np.array(list(range(args.num_users)))[:m]
        else:
            raise ValueError(f"Invalid policy {args.policy}")
        return idxs_users
    
    idxs_users = update_client_idx(use_all_users=True)
    print(f"Initialized clident IDs: {idxs_users}")

    user2local_model = {}
    def get_local_model_fn(idx):
        if idx not in user2local_model:
            user2local_model[idx] = LocalUpdate(
                args=args,
                dataset=train_dataset,
                idxs=user_groups[idx],
                logger=logger,
                model=global_model)
        return user2local_model[idx]

    for epoch in (range(args.epochs)): 
        local_weights, local_losses = [], []
        global_model.train()

        ### NOTE: deepcopy must be used here, or global_weights_before would change according to the weights in global_model
        global_weights_before = copy.deepcopy(global_model.state_dict())


        def test(w1, w2):
            for key in w1:
                equal = (np.abs(w1[key] - w2[key])/w2[key] < 1e-5).all()
                if not equal:
                    print(key, "error")
                    return
            print("all right")

        for idx in idxs_users:
            local_model = get_local_model_fn(idx)
            _weight, loss = local_model.update_weights(global_model, global_round=epoch)
            local_weights.append(copy.deepcopy(_weight))
            local_losses.append(copy.deepcopy(loss))

        ### Update global weights
        global_weights = average_weights(local_weights)
        # Load global weights to the global model
        global_model.load_state_dict(global_weights)

        if args.policy == "momentum":
            # print(f'global_grad:{global_grad}')
            client_proj.update_proj_list(idxs_users, global_weights, global_weights_before, local_weights)
        idxs_users = update_client_idx(use_all_users=False)

        # print global training loss after every 'i' rounds
        if (epoch+1) % print_every == 0:
            loss_avg = sum(local_losses) / len(local_losses)
            train_loss.append(loss_avg)

            # Calculate avg training accuracy over all users at every epoch
            list_acc, list_loss = [], []
            global_model.eval()
            for c in range(args.num_users):
                local_model = get_local_model_fn(idx)
                acc, loss = local_model.inference(model=global_model)
                list_acc.append(acc)
                list_loss.append(loss)
            train_accuracy.append(sum(list_acc)/len(list_acc))
            # print(f' \nlist acc {list_acc} ')
            
            print(f'Avg Training Stats after {epoch+1} global rounds: Training Loss : {np.mean(np.array(train_loss)):.3f}'
             f', Train Accuracy: {100*train_accuracy[-1]:.3f}% selcted idxs {idxs_users}')

    # Test inference after completion of training
    test_acc, test_loss = test_inference(args, global_model, test_dataset)

    print(f' \n Results after {args.epochs} global rounds of training:')
    print("|---- Avg Train Accuracy: {:.2f}%".format(100*train_accuracy[-1]))
    print("|---- Test Accuracy: {:.2f}%".format(100*test_acc))

    # Saving the objects train_loss and train_accuracy:
    save_dir = ".workspace/save/objects"
    os.makedirs(save_dir, exist_ok=True)
    file_name = os.path.join(save_dir, '{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}].pkl'.\
        format(args.dataset, args.model, args.epochs, args.frac, args.iid,
               args.local_ep, args.local_bs))

    with open(file_name, 'wb') as f:
        pickle.dump([train_loss, train_accuracy], f)

    print('\n Total Run Time: {0:0.4f}'.format(time.time()-start_time))

    # PLOTTING (optional)
    # import matplotlib
    # import matplotlib.pyplot as plt
    # matplotlib.use('Agg')

    # Plot Loss curve
    # plt.figure()
    # plt.title('Training Loss vs Communication rounds')
    # plt.plot(range(len(train_loss)), train_loss, color='r')
    # plt.ylabel('Training loss')
    # plt.xlabel('Communication Rounds')
    # plt.savefig('../save/fed_{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}]_loss.png'.
    #             format(args.dataset, args.model, args.epochs, args.frac,
    #                    args.iid, args.local_ep, args.local_bs))
    #
    # # Plot Average Accuracy vs Communication rounds
    # plt.figure()
    # plt.title('Average Accuracy vs Communication rounds')
    # plt.plot(range(len(train_accuracy)), train_accuracy, color='k')
    # plt.ylabel('Average Accuracy')
    # plt.xlabel('Communication Rounds')
    # plt.savefig('../save/fed_{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}]_acc.png'.
    #             format(args.dataset, args.model, args.epochs, args.frac,
    #                    args.iid, args.local_ep, args.local_bs))
