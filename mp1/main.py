import torch
import torch.nn as nn
import torch.nn.functional as F

import scipy as sp
import numpy as np
import numpy.linalg as la
from time import time

from opt import minimize_with_restart
from dataset import generate_linear_regression_dataset, generate_fitting_dataset
from utils import plot_func2D


class LinearRegression:
    def __init__(self, 
                A=np.array([[2, 1], [1, 2]]), 
                b=np.array([-5, -3]) ) -> None:
        self.A = A
        self.b = b
    
    def func(self, x):
        obj_func = 0.5 * np.dot(x.T, self.A).dot(x) - np.dot(self.b.T, x)
        return obj_func[0]
    
    def grad(self, x):
        grad = self.A @ x - self.b.T
        
        return grad[0]
    
    def run(self, method='CG', init_guess=None):
        if init_guess is None:
            init_guess = np.random.randn(self.A.shape[0],1)
        
        result = minimize_with_restart(self.func, init_guess, method=method, jac=self.grad, tol=1e-1,
                                options={
                                    'gtol': 1e-1,
                                    'disp': True,
                                    'maxiter': self.A.shape[0]+5,
                                    'return_all': True
                                })
        return result['allvecs'][-1]
    
class MLP_fitter:
    def __init__(self, X, Y, input_dim, hidden_dim, output_dim=1) -> None:
        self.X = torch.tensor(X, dtype=torch.float32)
        self.Y = torch.tensor(Y, dtype=torch.float32)
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def func(self, params):
        # Update model weights
        state_dict = {}
        for key in self.model.state_dict().keys():
            state_dict[key] = torch.tensor(params[:self.model.state_dict()[key].numel()]).reshape(self.model.state_dict()[key].shape)
            params = params[self.model.state_dict()[key].numel():]
        self.model.load_state_dict(state_dict)

        # Compute loss
        mse_loss = F.mse_loss(self.model(self.X), self.Y)
        return mse_loss.item()
    
    def grad(self, params):        
        state_dict = {}
        for key in self.model.state_dict().keys():
            state_dict[key] = torch.tensor(params[:self.model.state_dict()[key].numel()]).reshape(self.model.state_dict()[key].shape)
            params = params[self.model.state_dict()[key].numel():]
        self.model.load_state_dict(state_dict)

        # Compute loss
        mse_loss = F.mse_loss(self.model(self.X), self.Y)

        grad = torch.autograd.grad(mse_loss, self.model.parameters(), create_graph=True)
        grad = torch.cat([g.flatten() for g in grad])
        return grad.detach().numpy()
    
    def run(self, method='CG'):
        params_dict = self.model.state_dict()
        init_params = np.concatenate([params_dict[key].detach().numpy().flatten() for key in params_dict.keys()])
        print("Optimize variable size:", init_params.shape[0])
        print(f"Init loss: {self.func(init_params)}")

        print(f"------------{method}------------")
        result = minimize_with_restart(self.func, init_params, method=method, jac=self.grad, tol=1e-3,
                                        options={
                                            'gtol': 1e-2,
                                            'disp': True,
                                            'maxiter': 1000,
                                            'return_all': True
                                        })
        print(f"Final loss: {self.func(result.x)}")
        return result.x

def linear_regression(method='CG'):
    x, y, w = generate_linear_regression_dataset(N=2000)
    lr = LinearRegression(A=np.dot(x, x.T), b=y)
    
    print(f"------------{method}------------")
    start = time()
    alpha = lr.run(method, init_guess=np.random.randn(x.shape[0],1))
    end = time()
    w_hat = np.dot(x.T, alpha)
    print(f"Time Usage: {end-start}, Estimated weights: {w_hat}, True weights: {w}")

def mlp_fitting(method='CG'):
    tgt_func = lambda x: x**3 - 2*x**2 + 3*x - 1
    X, Y = generate_fitting_dataset(N=5000, func=tgt_func)
    mlp = MLP_fitter(X, Y, input_dim=1, hidden_dim=10, output_dim=1)
    
    print("\n"+"="*40)
    start = time()
    param = mlp.run(method)
    end = time()
    print(f"Time Usage: {end-start}")

    # Test the model
    state_dict = {}
    for key in mlp.model.state_dict().keys():
        state_dict[key] = torch.tensor(param[:mlp.model.state_dict()[key].numel()]).reshape(mlp.model.state_dict()[key].shape)
        param = param[mlp.model.state_dict()[key].numel():]
    mlp.model.load_state_dict(state_dict)
    error = F.mse_loss(mlp.model(mlp.X), mlp.Y).item()
    print(f"MSE Error: {error}")
    print("="*40 + "\n")


if __name__ == '__main__':
    # linear_regression()
    mlp_fitting('CG')
    mlp_fitting('BFGS')


    
