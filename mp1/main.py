import torch
import torch.nn as nn
import torch.nn.functional as F

import scipy as sp
import numpy as np
import numpy.linalg as la
from time import time

from opt import minimize_with_restart
from dataset import generate_linear_regression_dataset, generate_fitting_dataset
from utils import plot_func2D, visulize


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
    def __init__(self, X, Y, input_dim, hidden_dim, output_dim=1, layer_num=3) -> None:
        self.X = torch.tensor(X, dtype=torch.float32)
        self.Y = torch.tensor(Y, dtype=torch.float32)
        layers = [nn.Linear(input_dim, hidden_dim), nn.ReLU()] 
        for _ in range(layer_num-1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
        self.model = nn.Sequential(*layers, nn.Linear(hidden_dim, output_dim))


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
        mse_loss = F.mse_loss(self.model(self.X), self.Y, reduction='mean')

        grad = torch.autograd.grad(mse_loss, self.model.parameters(), create_graph=True)
        grad = torch.cat([g.flatten() for g in grad])
        return grad.detach().numpy()
    
    def run(self, method='CG'):
        params_dict = self.model.state_dict()
        init_params = np.concatenate([params_dict[key].detach().numpy().flatten() for key in params_dict.keys()])
        print("Optimize variable size:", init_params.shape[0])
        # init_params = np.random.randn(init_params.shape[0]) * 10 # Highly Non-linear initial func
        print(f"Init loss: {self.func(init_params)}")

        print(f"------------{method}------------")
        result = minimize_with_restart(self.func, init_params, method=method, jac=self.grad, tol=1e-2,
                                        options={
                                            'gtol': 1e-2,
                                            'disp': True,
                                            'maxiter': 1000,
                                            'return_all': True
                                        })
        print(f"Final loss: {self.func(result.x)}")
        return result
import memory_profiler

@memory_profiler.profile
def linear_regression(method='CG'):
    x, y, w = generate_linear_regression_dataset(N=1000, dim=500)
    lr = LinearRegression(A=np.dot(x, x.T), b=y)
    
    print(f"------------{method}------------")
    start = time()
    alpha = lr.run(method, init_guess=np.random.randn(x.shape[0]))
    end = time()
    w_hat = np.dot(x.T, alpha)
    print(f"Time Usage: {end-start}, Estimated weights: {w_hat}, True weights: {w}")

@memory_profiler.profile
def mlp_fitting(method='CG'):
    tgt_func = lambda x: x**3 - 2*x**2 + 3*x - 1 # lambda x: np.sin(x) + np.cos(x) +x**2#
    X, Y = generate_fitting_dataset(N=5000, func=tgt_func)
    layers = [1,3,5]
    for layer in layers:
        result_dict = {
            'error': [],
            'time': [],
            'iterations': [],
        }
        # for _ in range(5):
        mlp = MLP_fitter(X, Y, input_dim=1, hidden_dim=16, output_dim=1, layer_num=layer)

        print("\n"+"="*40)
        start = time()
        results = mlp.run(method)
        end = time()
        print(f"Time Usage: {end-start}")

        # Test the model
        param = results.x
        state_dict = {}
        for key in mlp.model.state_dict().keys():
            state_dict[key] = torch.tensor(param[:mlp.model.state_dict()[key].numel()]).reshape(mlp.model.state_dict()[key].shape)
            param = param[mlp.model.state_dict()[key].numel():]
        mlp.model.load_state_dict(state_dict)
        error = F.mse_loss(mlp.model(mlp.X), mlp.Y, reduction='mean').item()

        result_dict['error'].append(error)
        result_dict['time'].append(end-start)
        result_dict['iterations'].append(results.nit)

        print(f"MSE Error: {error}")
        test_X = np.linspace(-3, 3, 10000).reshape(-1, 1)
        print(f"x: {test_X.shape}")
        visulize(mlp.model, tgt_func, torch.FloatTensor(test_X), f'{method}_layer{layer}')
        print("="*40 + "\n")
        
        with open(f"results/final_res.txt", 'a') as f:
            f.write(f"Layer: {layer}, Method: {method}, Error: {np.mean(result_dict['error'])}, Time: {np.mean(result_dict['time'])}, Iterations: {np.mean(result_dict['iterations'])}\n")

if __name__ == '__main__':
    # linear_regression('CG')
    # linear_regression('BFGS')
    mlp_fitting('CG')
    mlp_fitting('BFGS')


    
