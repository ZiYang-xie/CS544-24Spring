from . import BaseModel

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
        for _ in range(iter):
            opt.zero_grad()
            pred = self.model(dataset['train_input'])
            loss = F.mse_loss(pred, dataset['train_label'])
            loss.backward()
            print("Loss: ", loss.item())
            opt.step()