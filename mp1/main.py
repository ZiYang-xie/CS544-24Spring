import scipy as sp
import numpy as np
import numpy.linalg as la

from opt import minimize_with_restart
from dataset import generate_dataset
from utils import plot_func2D


class LinearRegression:
    def __init__(self, 
                 A=np.array([[2, 1], [1, 2]]), 
                 b=np.array([-5, -3]) ) -> None:
        self.A = A
        self.b = b
        self.func = lambda x: 0.5 * x.T @ A @ x + b.T @ x
        self.grad = lambda x: A @ x + b.T
        
    def run(self, method='CG', init_guess=np.array([4,7])):
        result= minimize_with_restart(self.func, init_guess, method=method, jac=self.grad, tol=1e-2,
                            options={
                                'gtol': 1e-2,
                                'disp': True,
                                'maxiter': 5,
                                'return_all': True})
        
        plot_func2D(self.func, result['allvecs'], f'{method}.png')


if __name__ == '__main__':
    lr = LinearRegression()
    lr.run('CG')
    lr.run('BFGS')