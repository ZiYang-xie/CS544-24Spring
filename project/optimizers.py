import torch
from kan.LBFGS import LBFGS
from torch.optim.optimizer import Optimizer, required
import numpy as np
# from garage.torch.optimizers.conjugate_gradient_optimizer import ConjugateGradientOptimizer
from scipy.optimize import fmin_cg
def build_optimizer(config, param, fn=None, grad=None, x0=None):
    if config['name'] == 'Adam':
        return torch.optim.Adam(param, lr=config['lr'])
    elif config['name'] == 'LBFGS':
        return LBFGS(param, 
                     config['lr'], 
                     config['max_iter'],
                     config['history_size'], 
                     line_search_fn=config['line_search_fn'],
                     tolerance_grad=1e-32, tolerance_change=1e-32, tolerance_ys=1e-32
                    )
    # elif config['name'] == 'cg':
    #     return ConjugateGradientOptimizer(param, 
    #                                     max_constraint_value=np.inf,
    #                                     )
    elif config['name'] == 'CG':
        return CG(
            fn = fn,
            x0 = x0,
            grad = grad
        )
    else:
        raise ValueError(f"Optimizer {config['optimizer']} not supported")
    
class CG:
    def __init__(self, fn, x0, grad):
        self.fn = fn
        self.x0 = x0
        self.grad = grad
    def step(self):
        return fmin_cg(self.fn, self.x0, self.grad, maxiter=1)
