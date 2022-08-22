#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6
import copy
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

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


class LocalUpdate(object):
    def __init__(self, args, dataset, idxs, logger, model):
        self.args = args
        self.logger = logger
        self.trainloader, self.validloader, self.testloader = self.train_val_test(
            dataset, list(idxs))
        
        self.trainloader_iter = iter(self.trainloader)
        self.device = 'cuda' if args.gpu else 'cpu'
        # Default criterion set to NLL loss function
        self.criterion = nn.NLLLoss().to(self.device)

        self.model = None
        self.load_weights(model)
        self.local_step = 0
        # Set optimizer for the local updates
        if self.args.optimizer == 'sgd':
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.args.lr,
                                        momentum=0.5)
        elif self.args.optimizer == 'adam':
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.lr,
                                         weight_decay=1e-4)

    def train_val_test(self, dataset, idxs):
        """
        Returns train, validation and test dataloaders for a given dataset
        and user indexes.
        """
        # split indexes for train, validation, and test (80, 10, 10)
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
        

    def update_weights(self, global_model, global_round):
        self.load_weights(global_model)
        # Set mode to train model
        self.model.train()

        # for iter in range(self.args.local_ep):
        batch_loss = []
        num_batch = 1
        for batch_idx in range(num_batch):
            try:
                images, labels = next(self.trainloader_iter)
            except StopIteration:
                self.trainloader_iter = iter(self.trainloader)
                images, labels = next(self.trainloader_iter)
                self.local_step = 0
            # for batch_idx, (images, labels) in enumerate(self.trainloader):
            images, labels = images.to(self.device), labels.to(self.device)

            self.model.zero_grad()
            log_probs = self.model(images)
            loss = self.criterion(log_probs, labels)
            loss.backward()
            self.optimizer.step()

            self.local_step += 1

            if self.args.verbose and (batch_idx % 10 == 0):
                accu, _ = self.inference(self.model)
                print(f"| Global Round : {global_round} | "
                    f"[{self.local_step * len(images)}/{len(self.trainloader.dataset)} "
                    f"({100. * self.local_step / len(self.trainloader):.0f}%)]\t"
                    f"Loss: {loss.item():.6f}, accu={100*accu:.3f}%")
            self.logger.add_scalar('loss', loss.item())
            batch_loss.append(loss.item())

        return self.model.state_dict(), sum(batch_loss) / len(batch_loss)

    def inference(self, model):
        """ Returns the inference accuracy and loss.
        """

        model.eval()
        loss, total, correct = 0.0, 0.0, 0.0

        for batch_idx, (images, labels) in enumerate(self.testloader):
            images, labels = images.to(self.device), labels.to(self.device)

            # Inference
            outputs = model(images)
            batch_loss = self.criterion(outputs, labels)
            loss += batch_loss.item()

            # Prediction
            _, pred_labels = torch.max(outputs, 1)
            pred_labels = pred_labels.view(-1)
            correct += torch.sum(torch.eq(pred_labels, labels)).item()
            total += len(labels)

        accuracy = correct/total
        return accuracy, loss


def test_inference(args, model, test_dataset):
    """ Returns the test accuracy and loss.
    """

    model.eval()
    loss, total, correct = 0.0, 0.0, 0.0

    device = 'cuda' if args.gpu else 'cpu'
    criterion = nn.NLLLoss().to(device)
    testloader = DataLoader(test_dataset, batch_size=128,
                            shuffle=False)

    for batch_idx, (images, labels) in enumerate(testloader):
        images, labels = images.to(device), labels.to(device)

        # Inference
        outputs = model(images)
        batch_loss = criterion(outputs, labels)
        loss += batch_loss.item()

        # Prediction
        _, pred_labels = torch.max(outputs, 1)
        pred_labels = pred_labels.view(-1)
        correct += torch.sum(torch.eq(pred_labels, labels)).item()
        total += len(labels)

    accuracy = correct/total
    return accuracy, loss
