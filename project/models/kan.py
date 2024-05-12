from . import BaseModel
import numpy as np
from models.efficient_kan import KAN
from torch.nn import functional as F
from torch.nn import Module as Module
from typing import List
import torch


class KANModel(BaseModel):
    def __init__(self, 
                 width: List[int],
                 grid=5,
                 k=3,
                 loss_fn=F.mse_loss,
                 seed=42):
        super(KANModel, self).__init__()
        self.model = KAN(layers_hidden=width, grid_size=grid, spline_order=k, seed=seed)
        self.name = 'KAN'
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model.to(self.device)
        self.loss_fn = {
            'MSE': F.mse_loss,
            'CE': F.cross_entropy
        }[loss_fn]

    def vis(self, beta=100, mask=False):
        self.model.plot(beta=beta, mask=mask)

    def train(self, dataset, opt, iter=100, batch=-1):
        result = self.model.fit(dataset, opt, steps=iter, \
                device=self.device, loss_fn=self.loss_fn, batch=batch)
        return result['train_loss']
    
    def test(self, dataset):
        self.model.eval()
        dataset['test_input'] = dataset['test_input'].to(self.device)
        dataset['test_label'] = dataset['test_label'].to(self.device)

        pred = self.model(dataset['test_input'])
        loss = self.loss_fn(pred, dataset['test_label'])
        loss_val = loss.item()
        print("KAN Test Loss: ", loss_val)
        return {'loss': loss_val, 'pred': pred}


    