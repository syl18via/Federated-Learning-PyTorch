#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6


from asyncio import tasks
import os
import copy
import time
import pickle
import numpy as np
from tqdm import tqdm
from svfl import calculate_sv
from task import Task
import util

import torch
from tensorboardX import SummaryWriter

from options import args_parser
from update import LocalUpdate, test_inference
from models import MLP, CNNMnist, CNNFashion_Mnist, CNNCifar
from utils import get_dataset, average_weights, exp_details
np.random.seed(1)
class ClientState:
    ''' Store statistic information for all clients, which is used for client selection'''
    def __init__(self, num_users):
        self.client2proj = np.array([-math.inf] * num_users)
        # 在这个类的定义里面新建一个变量 用来存shapley value
        self.sv = np.array([-math.inf] * num_users)
    
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

### Experiment Configs
MIX_RATIO = 0.8
SIMULATE = False
EPOCH_NUM = 35
TRIAL_NUM = 1
TASK_NUM = 2

bid_per_loss_delta_space = [1]
required_client_num_space = [3]
target_labels_space = [[0,1,2,3,4,6,7,8,9]]

if __name__ == '__main__':
    start_time = time.time()

    # define paths
    path_project = os.path.abspath('..')
    logger = SummaryWriter('../logs')

    args = args_parser()
    exp_details(args)
    



    ############################### Task ###########################################
    ### Initialize the global model parameters for both tasks
    ### At the first epoch, both tasks select all clients
    task_list = []
    def create_task(selected_client_idx, required_client_num, bid_per_loss_delta, target_labels=None):
        task = Task(args,logger,
            task_id = len(task_list),
            selected_client_idx=selected_client_idx,
            required_client_num=required_client_num,
            bid_per_loss_delta=bid_per_loss_delta,
            target_labels=target_labels)
        # assert task.target_labels is not None, target_labels
        task_list.append(task)

        ### Init the loss
        task.prev_loss = task.evaluate_model()
    
    for task_id in range(TASK_NUM):
        create_task(
            selected_client_idx=list(range(args.num_users)),
            required_client_num=util.sample_config(required_client_num_space, task_id, use_random=False),
            bid_per_loss_delta=util.sample_config(bid_per_loss_delta_space, task_id, use_random=False),
            target_labels=util.sample_config(target_labels_space, task_id, use_random=False)
        )
    
    
    ############################### Main process of FL ##########################################
    total_reward_list = []
    succ_cnt_list = []
    reward_sum=[]
    for epoch in range(EPOCH_NUM):
        for task in task_list:
            task.epoch = epoch

        for round_idx in range(1):
            ### Train the model parameters distributedly
            #   return a list of model parameters
           
            for task in task_list:
                task.train_one_round()
            

        ### At the end of this epoch
        ### At the first epoch, calculate the Feedback and update clients for each task
        
        

        ### Update price table    

    
        ### Update bid table

        ###select clients for all tasks 
              
        for task in task_list:
            task.end_of_epoch()

        ### caclulate reward
       

        ### Count successful matching






    

    # copy weights
    global_weights = global_model.state_dict()

    # Training

    val_acc_list, net_list = [], []
    cv_loss, cv_acc = [], []
    
    val_loss_pre, counter = 0, 0

    import math
    client_state = ClientState(args.num_users)

    # client 选择函数，返回idxs_users，表示被选到的client的ID， 对的，
    # 然后后面变复杂之后，这里的选择函数的输入参数会更加复杂，要结合我们统计的momentum-based gradient projection 选择
    def momemtum_based(num_users):
        # 这里client_state 不需要传参了， 因为client_state在这个函数定义之前就已经定义了，函数内部可以直接访问client_state ok？
        momemtum_based_grad_proj = client_state.client2proj
        ''' momemtum_based_grad_proj 是一个list，长度等于 总的client数量，挑出momemtum_based_grad_proj最小的num_users client
        '''
        assert isinstance(momemtum_based_grad_proj, list) or isinstance(momemtum_based_grad_proj, np.ndarray)
        assert len(momemtum_based_grad_proj) == args.num_users
        momemtum_based_grad_proj = np.array(momemtum_based_grad_proj)
        return momemtum_based_grad_proj.argsort()[:num_users]

    def shap_based(num_users):
        
        ''' shap_based_grad_proj 是一个list，长度等于 总的client数量，挑出shap_based_grad_proj最大的num_users client
        '''
        sv = client_state.sv
        shap_based_grad_proj = np.array(sv)
        return shap_based_grad_proj.argsort()[-num_users:]
    
    def update_client_idx(use_all_users):
        m = max(int(args.frac * args.num_users), 1)
        if args.policy == "random":
            idxs_users = np.random.choice(range(args.num_users), m, replace=False) ### 这句话在选择client， 诶但是他已经实现了，每一个global step update 一次
        elif args.policy == "momentum":
            if use_all_users == True :
                idxs_users = momemtum_based(args.num_users)
            else:
                idxs_users = momemtum_based(m)
        elif args.policy == "shap":
            if use_all_users == True :
                idxs_users = shap_based(args.num_users)
            else:
                idxs_users = shap_based(m)
        elif args.policy == "debug":
            idxs_users = np.array(list(range(args.num_users)))[:m]
        else:
            raise ValueError(f"Invalid policy {args.policy}")
        return idxs_users
    
    idxs_users = update_client_idx(use_all_users=True)
    print(f"Initialized clident IDs: {idxs_users}")

    
   


    def fed_avg(client2weights):
        # function to merge the model updates into one model for evaluation, ex: FedAvg, FedProx
        # global_weights = average_weights(list(client2weights.values()))
        return average_weights(list(client2weights.values()))
    
    for epoch in (range(args.epochs)): 



        xxx


        ### update clients based on different policies
        if args.policy == "momentum":
            # print(f'global_grad:{global_grad}')
            client_state.update_proj_list(idxs_users, global_weights, global_weights_before, local_weights)
            idxs_users = update_client_idx(use_all_users=False)
        elif args.policy == "shap":
            if epoch == 0:
                client2weights = dict([(idxs_users[i], local_weights[i]) for i in range(len(idxs_users))])
                print(f"Calculate shaple value for {len(idxs_users)} clients")
                sv = calculate_sv(client2weights, evaluate_model, fed_avg)
                client_state.sv=sv
                idxs_users = update_client_idx(use_all_users=False)
            
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
