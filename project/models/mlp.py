from . import BaseModel
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from typing import List
from tqdm import tqdm
from kan.LBFGS import LBFGS
from ..optimizers import CG

class MLPModel(BaseModel):
    def __init__(self, 
                 width: List[int],
                 act='relu',
                 loss_fn='MSE',
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
                for i in range(len(width)-1)],
                nn.Linear(width[-1], width[-1])
            )
        self.name = 'MLP'
        self.train_input = None
        self.train_label = None
        self.test_input = None
        self.test_label = None
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model.to(self.device)


    def forward_detached(self, params):
        start_index = 0
        for param in self.model.parameters():
            numel = param.numel()
            size = param.size()
            param.copy_(torch.from_numpy(params[start_index:start_index + numel]).view(size))
            start_index += numel
        self.model.zero_grad()
        pred = self.model(self.train_input)
        loss = self.loss_fn(pred, self.train_label)
        loss.backward()
        grads = np.concatenate([p.grad.data.numpy().flatten() for p in self.model.parameters()])
        return loss.item(), grads

    def train(self, dataset, opt, iter=100, batch=-1):
        self.model.train()
        train_loss_list = []
        self.train_input = dataset['train_input'].to(self.device)
        self.train_label = dataset['train_label'].to(self.device)
        self.test_input = dataset['test_input'].to(self.device)
        self.test_label = dataset['test_label'].to(self.device)
        
        if batch == -1 or batch > dataset['train_input'].shape[0]:
            batch_size = dataset['train_input'].shape[0]
            batch_size_test = dataset['test_input'].shape[0]
        else:
            batch_size = batch
            batch_size_test = batch
        
        global train_loss

        def closure():
            global train_loss
            self.model.train()
            opt.zero_grad()
            pred = self.model.forward(self.train_input[train_id])
            train_loss = self.loss_fn(pred, self.train_label[train_id])
            train_loss.backward()
            return train_loss
        
        pbar = tqdm(range(iter * dataset['train_input'].shape[0] // batch_size), desc='description', ncols=100)
        for _ in pbar:
            train_id = np.random.choice(dataset['train_input'].shape[0], batch_size, replace=False)
            test_id = np.random.choice(dataset['test_input'].shape[0], batch_size_test, replace=False)

            if isinstance(opt, LBFGS):
                opt.step(closure)
            elif isinstance(opt, torch.optim.Adam):
                closure()
                opt.step()
            elif isinstance(opt, CG):
                result = opt.step()
                new_params = result[0]
                train_loss = result[1]
                # update model parameters, convert to torch tensor
                start_index = 0
                for param in self.model.parameters():
                    numel = param.numel()
                    size = param.size()
                    param.copy_(torch.from_numpy(new_params[start_index:start_index + numel]).view(size))
                    start_index += numel

            
            self.model.eval()
            pred = self.model(dataset['test_input'][test_id])
            test_loss = self.loss_fn(pred, dataset['test_label'][test_id])
            test_loss_val = test_loss.item()
            pbar.set_description("MLP: train loss: %.2e | test loss: %.2e " % (train_loss.item(), test_loss_val,))
            
            train_loss_list.append(train_loss.item())  # append the current loss value to the list
        return train_loss_list
    
    def test(self, dataset):
        self.model.eval()
        dataset['test_input'] = dataset['test_input'].to(self.device)
        dataset['test_label'] = dataset['test_label'].to(self.device)
        pred = self.model(dataset['test_input'])
        loss = self.loss_fn(pred, dataset['test_label'])
        loss_val = loss.item()
        print("MLP Test Loss: ", loss_val)
        return {'loss': loss_val, 'pred': pred}
    
    def additional_fields(self):
        d = {
            'x0': np.concatenate([p.detach().numpy().flatten() for p in self.model.parameters()]),
            'fn': lambda params: self.forward_detached(params)[0],
            'grad': lambda params: self.forward_detached(params)[1]
        }
        return d