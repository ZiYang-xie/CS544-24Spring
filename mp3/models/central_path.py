import numpy as np
from scipy import optimize
from scipy.sparse import eye, kron, diags, vstack
from scipy.sparse.linalg import cg
from tqdm import trange
from functools import partial
from .base import BaseSolver

import wandb

class CentralPathSolver(BaseSolver):
    def __init__(self, c, A, b, vis=True):
        self.c = c
        self.A = A
        self.b = b
        self.values = []
        self.vis = vis

    def solve(self, method, tol=1e-6, max_iter=500):
        def barrier_fn(x):
            aug_x = np.concatenate([x, [1]])
            return -np.sum(np.log(-self.A.dot(aug_x)))
        
        def origin_obj_fn(x):
            # import pdb; pdb.set_trace()
            return 0.5*x.dot(self.Q).dot(x) + np.dot(self.B, x)
        
        def obj_fn(x, ):
            return self.t * origin_obj_fn(x) + barrier_fn(x)
        
        self.t = 8
        m = self.A.shape[0]
        x = -np.ones(self.Q.shape[0])
        print(f"Initial x: {x}")
        # Solve the problem
        for i in trange(max_iter):
            if method == 'ALM':
                x = self.solve_ALM(x, obj_fn)
            self.t = 1.2*self.t
            if m/self.t < tol:
                break
        return x
    
    def gradient_augmented_lagrangian(self, x):
        # t*fx gradient
        # import pdb; pdb.set_trace()
        grad = self.t * (self.Q.dot(x) + self.B)
        grad = grad
        # phi(x) gradient
        aug_x = np.concatenate([x, [1]]).astype(np.float64)
        grad += -np.sum(self.A[:, :-1].T / self.A.dot(aug_x))
        # lambda * x gradient
        grad += -self.lamb.dot(self.C[:, -1])
        # mu * C.T * C * x gradient
        tmp_C = self.C[:, :-1]
        const_C = self.C[:, -1]
        grad += self.mu * 2 * (tmp_C.T.dot(tmp_C).dot(x) + tmp_C.T.dot(const_C))
        return grad
    
    def solve_ALM(self, x0, obj_fn, max_iter=10, tol=1e-7, verbose=True):
        if verbose:
            print("Starting ALM...")
        
        def augmented_lagrangian(x, ):
            L = obj_fn(x)
            aug_x = np.concatenate([x, [1]])
            L += -np.sum(self.lamb * np.dot(self.C, aug_x))
            L +=  + 0.5 * self.mu * np.linalg.norm(self.C.dot(aug_x))**2
            return L
        
        x0 = x0
        last_x = x0
        self.lamb = np.ones(self.C.shape[0]) # num_eqs
        self.mu = 1 # num_eqs
        for i in trange(max_iter):
            # print(f"x0: {x0}")
            # print(f"value:{obj_fn(x0,)} {augmented_lagrangian(x0, )}")
            # jacobi = self.gradient_augmented_lagrangian(x0,)
            # print(f"jacobi:{jacobi}")
            res = optimize.minimize(augmented_lagrangian, x0, \
                    jac=self.gradient_augmented_lagrangian, method='L-BFGS-B', options={'disp': False})
            x0 = res.x
            print(f"value:{obj_fn(x0,)} {augmented_lagrangian(x0, )}")
            aug_x = np.concatenate([x0, [1]])
            self.lamb = self.lamb - 0.5 * self.mu * self.C.dot(aug_x)
            self.mu = 0.9 * self.mu
            
            if np.linalg.norm(last_x - x0) < tol:
                if verbose:
                    print(f'Converged at iteration {i}')
                break
            last_x = x0
            
        return x0

    def primal_dual_method(self, x0, obj_fn, max_iter=10, tol=1e-7, verbose=True):
        if verbose:
            print("Starting Primal-Dual Method...")
        x0 = x0
        last_x = x0
        self.lamb = np.ones(self.C.shape[0])

if __name__ == "__main__":
    from utils import generate_problem
    c, A, b = generate_problem(num_eq=10, v_num=30)
    solver = CentralPathSolver(Q, B, A, C)
    x = solver.solve('ALM')
    print(x)