from . import BaseModel
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from typing import List
from tqdm import tqdm

class MLPModel(BaseModel):
    def __init__(self, 
                 width: List[int],
                 act='relu',
                 loss_fn=F.mse_loss,
                 dropout=0.0,
                 seed=42):
        super(MLPModel, self).__init__()
        act_fn = {
            'relu': nn.ReLU,
            'tanh': nn.Tanh,
            'sigmoid': nn.Sigmoid
        }[act]

        self.loss_fn = {
            'MSE': F.mse_loss,
            'CE': F.cross_entropy
        }[loss_fn]

        self.model = nn.Sequential(*[
                nn.Sequential(
                    nn.Linear(width[i], width[i+1]), 
                    act_fn(),
                    nn.Dropout(dropout, inplace=True)
                ) 
                for i in range(len(width)-1)]
            )
        self.loss_fn = loss_fn
        self.name = 'MLP'

    def train(self, dataset, opt, iter=100):
        self.model.train()
        train_loss_list = []
        
        pbar = tqdm(range(iter), desc='description', ncols=100)
        for _ in pbar:
            opt.zero_grad()
            self.model.train()
            pred = self.model(dataset['train_input'])
            train_loss = self.loss_fn(pred, dataset['train_label'])
            train_loss.backward()
            opt.step()
            train_loss_val = np.sqrt(train_loss.item())
            
            self.model.eval()
            pred = self.model(dataset['test_input'])
            test_loss = self.loss_fn(pred, dataset['test_label'])
            test_loss_val = np.sqrt(test_loss.item())
            pbar.set_description("train loss: %.2e | test loss: %.2e " % (train_loss_val, test_loss_val,))
            
            train_loss_list.append(train_loss_val)  # append the current loss value to the list
        return train_loss_list
    
    def test(self, dataset):
        self.model.eval()
        pred = self.model(dataset['test_input'])
        loss = self.loss_fn(pred, dataset['test_label'])
        loss_val = np.sqrt(loss.item())
        print("MLP Test Loss: ", loss_val)
        return {'loss': loss_val, 'pred': pred}