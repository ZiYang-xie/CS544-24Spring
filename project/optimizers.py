import torch
from kan.LBFGS import LBFGS
from torch.optim.optimizer import Optimizer, required

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


class ConjugateGradient(Optimizer):
    def __init__(self, params, lr=required, tolerance=1e-10, max_iter=100):
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if tolerance < 0.0:
            raise ValueError("Invalid tolerance: {}".format(tolerance))
        if max_iter < 1:
            raise ValueError("Invalid maximum iterations: {}".format(max_iter))

        defaults = dict(lr=lr, tolerance=tolerance, max_iter=max_iter)
        super(ConjugateGradient, self).__init__(params, defaults)

    def step(self, closure=None):
        """Performs a single optimization step."""
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            tolerance = group['tolerance']
            max_iter = group['max_iter']
            lr = group['lr']

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('ConjugateGradient does not support sparse gradients')

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['iteration'] = 0
                    state['old_grad'] = torch.zeros_like(p.data)
                    state['old_direction'] = torch.zeros_like(p.data)

                old_grad = state['old_grad']
                old_direction = state['old_direction']

                if state['iteration'] > 0:
                    beta = torch.sum(grad * (grad - old_grad)) / torch.sum(old_grad * old_grad)
                    direction = -grad + beta * old_direction
                else:
                    direction = -grad

                # Check tolerance
                grad_norm = torch.norm(grad)
                if grad_norm < tolerance:
                    continue

                # Simple line search: f(x + alpha * d)
                alpha = lr
                p.data.add_(alpha, direction)

                # Update state
                state['old_grad'] = grad
                state['old_direction'] = direction
                state['iteration'] += 1

                if state['iteration'] >= max_iter:
                    break

        return loss

