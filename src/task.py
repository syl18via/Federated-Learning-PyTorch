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
from client import get_local_model_fn
from svfl import calculate_sv

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
        self.train_loss, self.train_accuracy = [], []




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

        self.init_test_model(args, train_dataset, logger, user_groups)
    
        self.local_weights = []

    def train_one_round(self):
        
        self.local_weights, local_losses = [], []
        self.global_model.train()

        ### NOTE: deepcopy must be used here, or global_weights_before would change according to the weights in global_model
        global_weights_before = copy.deepcopy(self.global_model.state_dict())

        for idx in self.selected_client_idx: 
            local_model = get_local_model_fn(idx)
            _weight, loss = local_model.update_weights(self.global_model, global_round=epoch)
            self.local_weights.append(copy.deepcopy(_weight))
            local_losses.append(copy.deepcopy(loss))
        
        ### Update global weights
        global_weights = average_weights(self.local_weights)
        # Load global weights to the global model
        self.global_model.load_state_dict(global_weights)

   

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
            
            print(f'Avg Training Stats after {self.epoch+1} global rounds: Training Loss : {np.mean(np.array(train_loss)):.3f}'
             f', Train Accuracy: {100*self.train_accuracy[-1]:.3f}% selcted idxs {idxs_users}')
    
    def init_test_model(self, args,train_dataset, logger,user_groups):
        dataidx = np.array([np.array(list(user_groups[i])) for i in range(args.num_users)]).flatten()
        self.test_model = LocalUpdate(
            args=args,
            dataset=train_dataset,
            idxs=dataidx,
            logger=logger,
            model=self.global_model)

    def evaluate_model(self, weights=None):
        # function to compute evaluation metric, ex: accuracy, precision
        if weights is None:
            self.test_model.load_weights(self.global_model.state_dict())
        else:
            self.test_model.load_weights(weights)
        accu, losss = self.test_model.inference(self.test_model.model)
        return accu
   
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

    def shap(self):
        client2weights = dict([(self.selected_client_idx[i], self.local_weights[i]) for i in range(len(self.selected_client_idx))])
        print(f"Calculate shaple value for {len(self.selected_client_idx)} clients")
        sv = calculate_sv(client2weights, self.evaluate_model, fed_avg)
        return sv

    
    def end_train( self, args, test_dataset, start_time):
        # Test inference after completion of training
        test_acc, test_loss = test_inference(args, self.global_model, test_dataset)

        print(f' \n Task {self.task_id}: Results after {args.epochs} global rounds of training:')
        print("|---- Avg Train Accuracy: {:.2f}%".format(100*self.train_accuracy[-1]))
        print("|---- Test Accuracy: {:.2f}%".format(100*test_acc))

        print('Total Run Time: {0:0.4f}\n'.format(time.time()-start_time))

        
