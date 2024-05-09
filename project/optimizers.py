import torch
from kan.LBFGS import LBFGS

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
    else:
        raise ValueError(f"Optimizer {config['optimizer']} not supported")
