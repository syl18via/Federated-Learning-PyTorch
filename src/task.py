import os
import copy
import time
import pickle
import numpy as np
from tqdm import tqdm
from svfl import calculate_sv

import torch
from tensorboardX import SummaryWriter

from options import args_parser
from update import LocalUpdate, test_inference
from models import MLP, CNNMnist, CNNFashion_Mnist, CNNCifar
from utils import get_dataset, average_weights, exp_details

class Task:
    def __init__(self, task_id, selected_client_idx,
            required_client_num=None,
            bid_per_loss_delta=None,
            target_labels=None,
            args= None):



        if args.gpu:
            torch.cuda.set_device(args.gpu)
        device = 'cuda' if args.gpu else 'cpu'

        # load dataset and user groups
        train_dataset, test_dataset, user_groups = get_dataset(args)

        # BUILD MODEL
        if args.model == 'cnn':
            # Convolutional neural netork
            if args.dataset == 'mnist':
                self.global_model = CNNMnist(args=args)
            elif args.dataset == 'fmnist':
                self.global_model = CNNFashion_Mnist(args=args)
            elif args.dataset == 'cifar':
                self.global_model = CNNCifar(args=args)

        elif args.model == 'mlp':
            # Multi-layer preceptron
            img_size = train_dataset[0][0].shape
            len_in = 1
            for x in img_size:
                len_in *= x
                self.global_model = MLP(dim_in=len_in, dim_hidden=64,
                                dim_out=args.num_classes)
        else:
            exit('Error: unrecognized model')

        # Set the model to train and send it to device.
        # print(args.model)
        # print(args.dataset)
        self.global_model.to(device)
        self.global_model.train()
        # print(self.global_model)




        self.selected_client_idx = selected_client_idx    # a list of client indexes of selected clients
        self.model_epoch_start = None # model parameters at the start of an epoch
        self.task_id = task_id
        self.learning_rate = None
        self.epoch = 0
        self.required_client_num = required_client_num
        self.bid_per_loss_delta = bid_per_loss_delta

        self.prev_loss = None
        self.total_loss_delta = None
        self.params_per_client = None

        self.target_labels = target_labels
        
    def evaluate_model(self, test_dataset):
        raise
    def log(self, *args, **kwargs):
        print("[Task {} - epoch {}]: ".format(self.task_id, self.epoch), *args, **kwargs)

    def end_of_epoch(self):
        self.params_per_client = None
        self.model_epoch_start = None

    def select_clients(self, agent_shapley, free_client):
        # zip([1, 2, 3], [a, b, c]) --> [(1, a), (2, b), (3, c)]
        # enumerate([a, b, c])  --> [(1, a), (2, b), (3, c)]
        # agent_shapley = list(enumerate(agent_shapley))
        agent_shapley = zip(list(range(NUM_AGENT)), agent_shapley) ### shapley value of all clients, a list of (client_idx, value)
        #sorted_shapley_value = sorted(agent_shapley, key=lambda x: x[1], reverse=True)
        #self.log("Sorted shapley value: {}".format(sorted_shapley_value))
        self.selected_client_idx = []
        for client_idx, _ in agent_shapley:
            if free_client[client_idx] == 0:
                self.selected_client_idx.append(client_idx)
                if self.required_client_num and len(self.selected_client_idx) >= self.required_client_num:
                    break
        
        # ### !!! Select different clients for different tasks
        # ### TODO: the agent_shapley value should be considered 
        # ### E.g., Top-K
        # if task.task_id == 0:
        #     task.selected_client_idx = [0, 1, 2]
        # else:
        #     task.selected_client_idx = [3, 4]

        ### Update the client table
        for idx in self.selected_client_idx:
            free_client[idx] = 1

        self.log("Clients {} are selected.".format(self.selected_client_idx))


    
