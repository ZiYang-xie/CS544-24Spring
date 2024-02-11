import scipy as sp
import numpy as np
import numpy.linalg as la

from opt import minimize_with_restart
from dataset import generate_linear_regression_dataset
from utils import plot_func2D


class LinearRegression:
    def __init__(self, 
                A=np.array([[2, 1], [1, 2]]), 
                b=np.array([-5, -3]) ) -> None:
        self.A = A
        self.b = b
    
    def func(self, x):
        obj_func = 0.5 * x.T @ self.A @ x + self.b.T @ x
        print(f"obj_func: {obj_func}")
        return obj_func[0]
    
    def grad(self, x):
        grad = self.A @ x + self.b.T
        print(f"grad: {grad.shape}")
        return grad
    
    def run(self, method='CG', init_guess=None):
        if init_guess is None:
            init_guess = np.random.randn(self.A.shape[0],1)
            
        result= minimize_with_restart(self.func, init_guess, method=method, jac=None, tol=1e-2,
                            options={
                                'gtol': 1e-2,
                                'disp': True,
                                'maxiter': 5,
                                'return_all': True})
        
        plot_func2D(self.func, result['allvecs'], f'{method}.png')

if __name__ == '__main__':
    x, y, w = generate_linear_regression_dataset(N=100)
    import pdb; pdb.set_trace()
    lr = LinearRegression(A=np.dot(x, x.T), b=y)
    lr.run('CG')
    lr.run('BFGS')
