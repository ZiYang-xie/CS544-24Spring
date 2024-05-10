from . import BaseModel
import numpy as np
from kan import KAN 
from torch.nn import functional as F
from torch.nn import Module as Module
from typing import List
import torch


class KANModel(BaseModel):
    def __init__(self, 
                 width: List[int],
                 grid=5,
                 k=3,
                 seed=42):
        super(KANModel, self).__init__()
        self.model = KAN(width=width, grid=grid, k=k, seed=seed)

    def vis(self, beta=100, mask=False):
        self.model.plot(beta=beta, mask=mask)

    def train(self, dataset, opt, iter=100):
        #  Module.train(self.model)
        result = self.model.train(dataset, opt, steps=iter)
        return result['train_loss']
    
    def test(self, dataset):
        Module.eval(self.model)
        pred = self.model(dataset['test_input'])
        loss = F.mse_loss(pred, dataset['test_label'])
        loss_val = np.sqrt(loss.item())
        print("KAN Test Loss: ", loss_val)
        return {'loss': loss_val, 'pred': pred}


    