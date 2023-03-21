import numpy as np
import copy

import torch
from torch import nn
import torch.optim as optim

from utils import DatasetSplit, DatasetRelabel
from torch.utils.data import DataLoader
from sampling import get_dataset, check_dist

def test_inference(args, model, test_dataset):
    """ Returns the test accuracy and loss.
    """

    model.eval()
    loss, total, correct = [], 0.0, 0.0

    device = 'cuda' if args.gpu is not None else 'cpu'

    if args.dataset == 'cifar':
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = nn.NLLLoss().to(device)
        
    testloader = DataLoader(test_dataset, batch_size=128, shuffle=False)

    for batch_idx, (images, labels) in enumerate(testloader):
        images, labels = images.to(device), labels.to(device)

        # Inference
        outputs = model(images)
        batch_loss = criterion(outputs, labels)
        loss.append(batch_loss.item()) 

        # Prediction
        _, pred_labels = torch.max(outputs, 1)
        pred_labels = pred_labels.view(-1)
        correct += torch.sum(torch.eq(pred_labels, labels)).item()
        total += len(labels)

    accuracy = correct/total
    return accuracy, sum(loss)/len(loss)

class Client(DatasetSplit):
    def __init__(self, id, dataset, data_idxs):
        self.id = id
        self.dataset = dataset
        self.idxs = [int(i) for i in data_idxs]
        
        ### Group local data by labels
        self.lable2data_idxs = {}
        for idx_of_split in self.idxs:
            _, label = self.dataset[idx_of_split]
            if label not in self.lable2data_idxs:
                self.lable2data_idxs[label] = []
            self.lable2data_idxs[label].append(idx_of_split)


def get_clients(args):
    train_dataset, test_dataset, user_groups = get_dataset(args)
    clients = {}
    for client_id in user_groups.keys():
        sample_idxs = user_groups[client_id]
        _client = Client(client_id, train_dataset, list(sample_idxs))
        check_dist(f"Client {client_id}", _client)
        clients[client_id] = _client

    sample_idxs = range(len(test_dataset))
    test_client = Client(-1, test_dataset, list(sample_idxs))
    check_dist(f"Test", test_client)

    return train_dataset, test_client, clients

class VirtualClient:
    def __init__(self, args, dataset, logger, global_model, target_labels=None,
            split=False, shuffle=True, filter=False, required_dist=None):
        self.args = args
        self.logger = logger

        # target_labels = None
        if target_labels is None:
            self.dataset = dataset
        elif filter:
            ### For samples not belonging to target labels, filter out
            assert isinstance(dataset, Client), type(dataset)
            target_idxs = []
            for _label in target_labels:
                target_idxs += dataset.lable2data_idxs[_label]
            self.dataset = DatasetSplit(dataset.dataset, target_idxs)
        else:
            ### For samples not belonging to target labels, label them with -1
            assert isinstance(dataset, Client), type(dataset)
            assert target_labels is not None
            self.dataset = DatasetRelabel(dataset.dataset, dataset.idxs, target_labels, required_dist=required_dist)
        
        if split:
            self.trainloader, self.validloader, self.testloader = self.train_val_test(self.dataset)
        else:
            self.trainloader = DataLoader(self.dataset, batch_size=self.args.local_bs, shuffle=shuffle)
            self.validloader = self.testloader = None

        self.trainloader_iter = iter(self.trainloader)
        self.device = 'cuda' if args.gpu else 'cpu'
        self.target_labels = target_labels
        self.model = None
        self.load_weights(global_model)
        self.local_step = 0
       
        if args.dataset == 'cifar':
            self.criterion = nn.CrossEntropyLoss()
            self.optimizer = optim.SGD(self.model.parameters(), lr=self.args.lr,
                                momentum=0.9, weight_decay=5e-4)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=200)

        else:
            # Default criterion set to NLL loss function
            self.criterion = nn.NLLLoss().to(self.device)
        
            # Set optimizer for the local updates
            if self.args.optimizer == 'sgd':
                self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.args.lr,
                                            momentum=0.5)
            elif self.args.optimizer == 'adam':
                self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.lr,
                                            weight_decay=1e-4)
    
    def train_val_test(self, dataset):
        """
        Returns train, validation and test dataloaders for a given dataset
        and user indexes.
        """
        # split indexes for train, validation, and test (80, 10, 10)
        idxs = np.arange(len(dataset))
        idxs_train = idxs[:int(0.8*len(idxs))]
        idxs_val = idxs[int(0.8*len(idxs)):int(0.9*len(idxs))]
        idxs_test = idxs[int(0.9*len(idxs)):]

        trainloader = DataLoader(DatasetSplit(dataset, idxs_train),
                                 batch_size=self.args.local_bs, shuffle=True)
        validloader = DataLoader(DatasetSplit(dataset, idxs_val),
                                 batch_size=int(len(idxs_val)/10), shuffle=False)
        testloader = DataLoader(DatasetSplit(dataset, idxs_test),
                                batch_size=int(len(idxs_test)/10), shuffle=False)
        return trainloader, validloader, testloader

    def load_weights(self, global_model):
        if isinstance(global_model, dict):
            self.model.load_state_dict(global_model)
        elif isinstance(global_model, torch.nn.Module):
            if self.model is None:
                self.model = copy.deepcopy(global_model)
            else:
                self.model.load_state_dict(copy.deepcopy(global_model.state_dict()))
        else:
            raise ValueError()

    def train_step(self, global_model, epoch):
        self.load_weights(global_model)
        # Set mode to train model
        self.model.train()

        # for iter in range(self.args.local_ep):
        batch_loss = []
        num_batch = 100
        for batch_idx in range(num_batch):
            try:
                images, labels = next(self.trainloader_iter)
            except StopIteration:
                self.trainloader_iter = iter(self.trainloader)
                images, labels = next(self.trainloader_iter)
                self.local_step = 0
            # for batch_idx, (images, labels) in enumerate(self.trainloader):
            # print(labels, self.target_labels)
            images, labels = images.to(self.device), labels.to(self.device)

            if self.args.dataset == 'cifar':
                self.optimizer.zero_grad()
            else:
                self.model.zero_grad()

            outputs = self.model(images)

            if self.args.dataset == 'cifar':
                loss = self.criterion(outputs, labels)
            else:
                value, indices = torch.max(outputs,1)
                # print("Predicted values", indices)
                # import code
                # code.interact(local=locals())
                loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()

            self.local_step += 1

            # if self.args.verbose and (batch_idx % 10 == 0):
            #     accu, _ = self.inference(self.model)
            #     print(f"| Global Round : {global_round} | "
            #         f"[{self.local_step * len(images)}/{len(self.trainloader.dataset)} "
            #         f"({100. * self.local_step / len(self.trainloader):.0f}%)]\t"
            #         f"Loss: {loss.item():.6f}, accu={100*accu:.3f}%")

            # self.logger.add_scalar('loss', loss.item())
            batch_loss.append(loss.item())

        return self.model.state_dict(), sum(batch_loss) / len(batch_loss)

    def inference(self, dataset):
        """ Returns the inference accuracy and loss.
        """
        return test_inference(self.args.gpu is not None, self.model, dataset)
