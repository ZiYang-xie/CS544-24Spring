import numpy as np
from scipy import optimize
from scipy.sparse import eye, kron, diags, vstack
from scipy.sparse.linalg import cg
from tqdm import trange
# from .base import BaseSolver

import wandb

class CentralPathSolver():
    def __init__(self, c, A, b, t=1, update_t=1.2, use_wandb=False, vis=True):
        self.c = c
        self.A = A
        self.b = b
        self.t = t
        self.update_t = update_t
        self.values = []
        self.vis = vis
        self.history = {
            'values': [],
        }

    def solve(self, tol=1e-5, max_iter=100):
        def origin_obj_fn(x):
            return self.c @ x
        
        def barrier_fn(x):
            barrier = -np.sum(np.log(x+1e-7))
            assert not np.isnan(barrier)
            return barrier
        
        def obj_fn(x):
            return self.t * origin_obj_fn(x) + barrier_fn(x)
        
        m = self.A.shape[0]
        x = np.ones(self.A.shape[1])
        print(f"Initial x: {x}")
        # Solve the problem
        for i in trange(max_iter):
            results = self.solve_ALM(x, obj_fn)
            x = results['x'] # New initial point
            self.t = self.update_t*self.t
            if m/self.t < tol:
                break
        return results
    
    def gradient_augmented_lagrangian(self, x):
        fx_grad = self.t * self.c
        phi_grad = -1/(x+1e-7)
        lambda_grad = self.A.T @ self.lamb
        mu_grad = self.mu * self.A.T @ (self.A @ x - self.b)
        grad = fx_grad + phi_grad + lambda_grad + mu_grad
        return grad
    
    def solve_ALM(self, x0, obj_fn, max_iter=100, tol=1e-5, verbose=False):
        if verbose:
            print("Starting ALM...")
        
        def augmented_lagrangian(x):
            L = obj_fn(x)
            L += np.sum(self.lamb * (self.A @ x - self.b))
            L += 0.5 * self.mu * np.linalg.norm(self.A @ x - self.b)**2
            return L
        
        last_x = x0
        self.lamb = np.ones(self.A.shape[0])
        self.mu = self.t * x0.shape[0]
        for i in range(max_iter):
            res = optimize.minimize(
                augmented_lagrangian, x0, 
                jac=self.gradient_augmented_lagrangian, 
                method='L-BFGS-B',
                bounds=[(0, np.inf) for _ in range(x0.shape[0])],
                options={'disp': False}
            )
            x0 = res.x
            self.lamb = self.lamb + self.mu * (self.A @ x0 - self.b)
            self.mu = 0.99 * self.mu
            if np.linalg.norm(last_x - x0) < tol:
                if verbose:
                    print(f'Converged at iteration {i}')
                break
            last_x = x0
        
            self.history['values'].append(np.dot(self.c, x0))
        result = {
            'x': x0,
            'value': np.dot(self.c, x0),
            'value_history': self.history['values'],
        }
            
        return result


if __name__ == "__main__":
    def generate_problem(v_num=30, num_eq=10):
        c = np.random.randn(v_num)
        A = np.random.randn(num_eq, v_num)
        b = np.random.randn(num_eq)
        # c = np.array([1,1])
        # A = np.array([[1,-1]])
        # b = np.array([1])
        return c, A, b
    c, A, b = generate_problem(v_num=30, num_eq=15)
    solver = CentralPathSolver(c, A, b)
    results = solver.solve()
    print(results['value'])
    print(results['x'])
    print(np.abs(A @ results['x'] - b).mean())