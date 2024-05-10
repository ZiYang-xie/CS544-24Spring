from . import BaseModel
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from typing import List

class MLPModel(BaseModel):
    def __init__(self, 
                 width: List[int],
                 act='relu',
                 dropout=0.0,
                 seed=42):
        super(MLPModel, self).__init__()
        act_fn = {
            'relu': nn.ReLU,
            'tanh': nn.Tanh,
            'sigmoid': nn.Sigmoid
        }[act]

        self.model = nn.Sequential(*[
                nn.Sequential(
                    nn.Linear(width[i], width[i+1]), 
                    act_fn(),
                    nn.Dropout(dropout, inplace=True)
                ) 
                for i in range(len(width)-1)]
            )

    def train(self, dataset, opt, iter=100):
        self.model.train()
        train_loss = []
        for _ in range(iter):
            opt.zero_grad()
            pred = self.model(dataset['train_input'])
            loss = F.mse_loss(pred, dataset['train_label'])
            loss.backward()
            opt.step()
            loss_val = np.sqrt(loss.item())
            print("RMSE Loss: ", loss_val)
            train_loss.append(loss_val)  # append the current loss value to the list
        return train_loss
    
    def test(self, dataset):
        self.model.eval()
        pred = self.model(dataset['test_input'])
        loss = F.mse_loss(pred, dataset['test_label'])
        loss_val = np.sqrt(loss.item())
        print("MLP Test Loss: ", loss_val)
        return {'loss': loss_val, 'pred': pred}