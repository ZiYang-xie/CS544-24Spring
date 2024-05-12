from . import BaseModel
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from typing import List
from tqdm import tqdm
from kan.LBFGS import LBFGS

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
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model.to(self.device)

    def train(self, dataset, opt, iter=100, batch=-1, loss_fn=None):
        self.model.train()
        train_loss_list = []
        dataset['train_input'] = dataset['train_input'].to(self.device)
        dataset['train_label'] = dataset['train_label'].to(self.device)
        dataset['test_input'] = dataset['test_input'].to(self.device)
        dataset['test_label'] = dataset['test_label'].to(self.device)
        if loss_fn is None:
            loss_fn = F.mse_loss
        else:
            loss_fn = loss_fn
        
        if batch == -1 or batch > dataset['train_input'].shape[0]:
            batch_size = dataset['train_input'].shape[0]
            batch_size_test = dataset['test_input'].shape[0]
        else:
            batch_size = batch
            batch_size_test = batch
        
        global train_loss

        def closure():
            global train_loss
            opt.zero_grad()
            pred = self.model.forward(dataset['train_input'][train_id])
            train_loss = loss_fn(pred, dataset['train_label'][train_id])
            objective = train_loss
            objective.backward()
            return objective
        
        pbar = tqdm(range(iter), desc='description', ncols=100)
        for _ in pbar:
            
            train_id = np.random.choice(dataset['train_input'].shape[0], batch_size, replace=False)
            test_id = np.random.choice(dataset['test_input'].shape[0], batch_size_test, replace=False)

            if isinstance(opt, LBFGS):
                opt.step(closure)
            elif isinstance(opt, torch.optim.Adam):
                opt.zero_grad()
                self.model.train()
                pred = self.model(dataset['train_input'])
                train_loss = loss_fn(pred, dataset['train_label'])
                train_loss.backward()
                opt.step()
            
            self.model.eval()
            pred = self.model(dataset['test_input'])
            test_loss = loss_fn(pred, dataset['test_label'])
            test_loss_val = test_loss.item()
            pbar.set_description("MLP: train loss: %.2e | test loss: %.2e " % (train_loss.item(), test_loss_val,))
            
            train_loss_list.append(train_loss.item())  # append the current loss value to the list
        return train_loss_list
    
    def test(self, dataset, loss_fn=None):
        self.model.eval()
        dataset['test_input'] = dataset['test_input'].to(self.device)
        dataset['test_label'] = dataset['test_label'].to(self.device)
        if loss_fn is None:
            loss_fn = F.mse_loss
        else:
            loss_fn = loss_fn
        pred = self.model(dataset['test_input'])
        loss = loss_fn(pred, dataset['test_label'])
        loss_val = loss.item()
        print("MLP Test Loss: ", loss_val)
        return {'loss': loss_val, 'pred': pred}