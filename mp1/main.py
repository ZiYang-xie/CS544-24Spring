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
        obj_func = 0.5 * np.dot(x.T, self.A).dot(x) - np.dot(self.b.T, x)
        # 0.5 * x.T @ self.A @ x - self.b.T @ x
        
        # print(f"obj_func: {obj_func}")
        return obj_func[0]
    
    def grad(self, x):
        grad = self.A @ x - self.b.T
        
        return grad[0]
    
    def run(self, method='CG', init_guess=None):
        
        if init_guess is None:
            init_guess = np.random.randn(self.A.shape[0])
        
        result= minimize_with_restart(self.func, init_guess, method=method, jac=self.grad, tol=1e-1,
                            options={
                                'gtol': 1e-1,
                                'disp': True,
                                'maxiter': self.A.shape[0]+5,
                                'return_all': True})
        
        return result['allvecs'][-1]
        # plot_func2D(self.func, result['allvecs'], f'{method}.png')

if __name__ == '__main__':
    # x: (N, 2)
    # y: (N, 1)
    # alpha: (N, 1)
    x, y, w = generate_linear_regression_dataset(N=2000, dim=10)
    lr = LinearRegression(A=np.dot(x, x.T), b=y)
    print(f"------------CG------------")
    import time
    cg_start = time.time()
    alpha = lr.run('CG', init_guess=np.random.randn(x.shape[0]))
    cg_end = time.time()
    # print(f"alpha: {alpha}")
    w_hat = np.dot(x.T, alpha)
    
    print(f"time:{cg_end-cg_start}, Estimated weights: {w_hat}, True weights: {w}")
    print(f"------------BFGS------------")
    bfgs_start = time.time()
    alpha = lr.run('BFGS')
    bfgs_end = time.time()
    w_hat = np.dot(x.T, alpha)
    print(f"time:{bfgs_end-bfgs_start}, Estimated weights: {w_hat}, True weights: {w}")
