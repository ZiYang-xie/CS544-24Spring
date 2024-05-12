import torch 
import torch.nn as nn

class BaseModel(nn.Module):
    def __init__(self):
        super(BaseModel, self).__init__()

    def train(self):
        raise NotImplementedError
    
    def additional_fields(self):
        raise NotImplementedError
        return None
from .kan import KANModel
from .mlp import MLPModel