import torch 
import torch.nn as nn

class BaseModel(nn.Module):
    def __init__(self):
        super(BaseModel, self).__init__()

    def train(self):
        raise NotImplementedError
    
from .kan import KANModel
from .mlp import MLPModel