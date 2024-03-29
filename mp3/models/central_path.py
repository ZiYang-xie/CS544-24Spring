import numpy as np
from scipy import optimize
from scipy.sparse import eye, kron, diags, vstack
from scipy.sparse.linalg import cg
from tqdm import trange
# from .base import BaseSolver

import wandb

class CentralPathSolver():
    def __init__(self, c, A, b, t=10, use_wandb=False, vis=True):
        self.c = c
        self.A = A
        self.b = b
        self.t = t
        self.values = []
        self.vis = vis
        self.history = {
            'values': [],
        }

    def solve(self, tol=1e-2, max_iter=5):
        def origin_obj_fn(x):
            return self.c @ x
        
        def barrier_fn(x):
            return -np.sum(np.log(x))
        
        def obj_fn(x):
            return self.t * origin_obj_fn(x) + barrier_fn(x)
        
        m = self.A.shape[0]
        x = np.ones(self.A.shape[1])
        print(f"Initial x: {x}")
        # Solve the problem
        for i in trange(max_iter):
            results = self.solve_ALM(x, obj_fn)
            x = results['x'] # New initial point
            self.t = 2*self.t
            if m/self.t < tol:
                break
        return results
    
    def gradient_augmented_lagrangian(self, x):
        fx_grad = self.t * self.c
        phi_grad = -1/x
        lambda_grad = -np.sum(self.A * self.lamb[:, None], axis=0)
        mu_grad = self.mu**2 * (self.A @ x - self.b) @ self.A
        grad = fx_grad + phi_grad + lambda_grad + mu_grad
        return grad
    
    def solve_ALM(self, x0, obj_fn, max_iter=10, tol=1e-7, verbose=True):
        if verbose:
            print("Starting ALM...")
        
        def augmented_lagrangian(x):
            L = obj_fn(x)
            L += -np.sum(self.lamb * (self.A @ x - self.b))
            L += 0.5 * self.mu * np.linalg.norm(self.A @ x - self.b)**2
            return L
        
        last_x = x0
        self.lamb = np.ones(self.A.shape[0])
        self.mu = 1 # num_eqs
        for i in trange(max_iter):
            res = optimize.minimize(
                augmented_lagrangian, x0, 
                jac=self.gradient_augmented_lagrangian, 
                method='L-BFGS-B',
                options={'disp': False}
            )
            x0 = res.x
            self.lamb = self.lamb - self.mu * (self.A @ x0 - self.b)
            self.mu = 0.9 * self.mu
            print(f"Value: {np.dot(self.c, x0)}")
            self.history['values'].append(np.dot(self.c, x0))
            if np.linalg.norm(last_x - x0) < tol:
                if verbose:
                    print(f'Converged at iteration {i}')
                break
            last_x = x0

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
        return c, A, b
    c, A, b = generate_problem(v_num=30, num_eq=15)
    solver = CentralPathSolver(c, A, b)
    x = solver.solve()
    print(x)