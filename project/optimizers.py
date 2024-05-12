import torch
from kan.LBFGS import LBFGS
from torch.optim.optimizer import Optimizer, required
import numpy as np
from garage.torch.optimizers.conjugate_gradient_optimizer import ConjugateGradientOptimizer
def build_optimizer(config, param):
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
    elif config['name'] == 'cg':
        return ConjugateGradientOptimizer(param, 
                                        max_constraint_value=np.inf,
                                        )
    else:
        raise ValueError(f"Optimizer {config['optimizer']} not supported")