from . import BaseModel

from kan import KAN 
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

    def train(self, dataset, opt):
        self.model.train(dataset, opt)

    