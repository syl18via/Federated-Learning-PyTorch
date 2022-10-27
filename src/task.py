import os
import copy
import time
import pickle
import numpy as np
from tqdm import tqdm
import math


import torch
from tensorboardX import SummaryWriter

from options import args_parser
from client import test_inference
from models import MLP, CNNMnist, CNNFashion_Mnist, CNNCifar
from utils import get_dataset, average_weights, exp_details
from client import Client
from svfl import calculate_sv

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

PRINT_EVERY = 10

def fed_avg(client2weights):
    # function to merge the model updates into one model for evaluation, ex: FedAvg, FedProx
    # global_weights = average_weights(list(client2weights.values()))
    return average_weights(list(client2weights.values()))

class Task:
    def __init__(self, args,
            logger,task_id, selected_client_idx,
            required_client_num=None,
            bid_per_loss_delta=None,
            target_labels=None):

        if args.gpu:
            torch.cuda.set_device(args.gpu)
        device = 'cuda' if args.gpu else 'cpu'

        # load dataset and user groups
        self.train_dataset, self.test_dataset, self.user_groups = get_dataset(args)

        # BUILD MODEL
        if args.model == 'cnn':
            # Convolutional neural netork
            if args.dataset == 'mnist':
                self.global_model = CNNMnist(args=args)
            elif args.dataset == 'fmnist':
                self.global_model = CNNFashion_Mnist(args=args)
            elif args.dataset == 'cifar':
                self.global_model = CNNCifar(args=args)
            else:
                raise ValueError(f"Invalid dataset {args.dataset}")
        elif args.model == 'mlp':
            # Multi-layer preceptron
            img_size = self.train_dataset[0][0].shape
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
        self.train_loss, self.train_accuracy = [], []

        self.selected_client_idx = selected_client_idx    # a list of client indexes of selected clients
        self.task_id = task_id
        self.epoch = 0
        self.required_client_num = required_client_num
        self.bid_per_loss_delta = bid_per_loss_delta
        self.target_labels = target_labels

        self.total_loss_delta = None

        self.init_test_model(args, logger)
    
        self.args = args
        self.logger = logger
        self.selected_clients = None
        self.client_state = ClientState(args.num_users)
        self.local_weights = []

        self.init_select_clients()

    def train_one_round(self):
        
        self.local_weights, local_losses = [], []
        self.global_model.train()

        ### NOTE: deepcopy must be used here, or global_weights_before would change according to the weights in global_model
        global_weights_before = copy.deepcopy(self.global_model.state_dict())

        for idx in range(len(self.selected_client_idx)):
            ### Here idx is NOT the client idx
            client = self.selected_clients[idx]
            _weight, loss = client.train_step(self.global_model, self.epoch)
            self.local_weights.append(copy.deepcopy(_weight))
            local_losses.append(copy.deepcopy(loss))
        
        ### Update global weights
        global_weights = average_weights(self.local_weights)
        # Load global weights to the global model
        self.global_model.load_state_dict(global_weights)

        if self.args.policy == "momentum":
            self.client_state.update_proj_list(self.selected_client_idx,
                global_weights, global_weights_before, self.local_weights)

        # print global training loss after every 'i' rounds
        if (self.epoch+1) % PRINT_EVERY == 0:
            loss_avg = sum(local_losses) / len(local_losses)
            self.train_loss.append(loss_avg)

            # Calculate avg training accuracy over all users at every epoch
            list_acc, list_loss = [], []
            self.global_model.eval()
            
            accu= self.evaluate_model(global_weights)
            self.train_accuracy.append(accu)
            # print(f' \nlist acc {list_acc} ')
            
            print(f'Avg Training Stats after {self.epoch+1} global rounds: Training Loss : {np.mean(np.array(self.train_loss)):.3f}'
             f', Train Accuracy: {100*self.train_accuracy[-1]:.3f}% selcted idxs {self.selected_client_idx}')
    
    def init_test_model(self, args, logger):
        self.test_model = Client(
            args=args,
            dataset=self.test_dataset,
            user_groups=self.user_groups,
            client_idx=None,
            logger=logger,
            global_model=self.global_model,
            shuffle=False)

    def evaluate_model(self, weights=None):
        # function to compute evaluation metric, ex: accuracy, precision
        if weights is None:
            self.test_model.load_weights(self.global_model.state_dict())
        else:
            self.test_model.load_weights(weights)
        accu, losss = self.test_model.inference(self.test_model.dataset)
        return accu
   
    def log(self, *args, **kwargs):
        print("[Task {} - epoch {}]: ".format(self.task_id, self.epoch), *args, **kwargs)

    def end_of_epoch(self):
        pass

    def select_clients(self, agent_shapley, free_client):
        raise NotImplementedError()
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

    def shap(self):

        # ### TODO used for debug
        # return [1] * len(self.selected_client_idx)

        client2weights = dict([(self.selected_client_idx[i], self.local_weights[i]) for i in range(len(self.selected_client_idx))])
        print(f"Calculate shaple value for {len(self.selected_client_idx)} clients")
        sv = calculate_sv(client2weights, self.evaluate_model, fed_avg)
        return sv

    def end_train( self, args, test_dataset, start_time):
        # Test inference after completion of training
        test_acc, test_loss = test_inference(args.gpu is not None, self.global_model, test_dataset)

        print(f' \n Task {self.task_id}: Results after {args.epochs} global rounds of training:')
        print("|---- Avg Train Accuracy: {:.2f}%".format(100*self.train_accuracy[-1]))
        print("|---- Test Accuracy: {:.2f}%".format(100*test_acc))

        print('Total Run Time: {0:0.4f}\n'.format(time.time()-start_time))

    def init_select_clients(self):
        self.selected_clients = []
        if self.selected_client_idx is None:
            return
        for client_idx in self.selected_client_idx:
            self.selected_clients.append(
                Client(self.args, self.train_dataset, self.user_groups,
                    client_idx, self.logger, self.global_model))

    @property
    def delta_accu(self):
        if len(self.train_accuracy) == 0:
            return 0
        elif len(self.train_accuracy) == 1:
            return self.train_accuracy[0]
        else:
            return self.train_accuracy[-1] - self.train_accuracy[-2]
        