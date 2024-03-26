import numpy as np
from scipy import optimize
from scipy.sparse import eye, kron, diags, vstack
from scipy.sparse.linalg import cg
from tqdm import trange
from functools import partial
from utils import generate_problem
from base import BaseSolver

class CentralPathSolver(BaseSolver):
    def __init__(self, Q, B, A, C):
        self.Q = Q
        self.B = B
        self.A = A
        self.C = C

    def solve(self, method, tol=1e-6):
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
        while True:
            if method == 'ALM':
                x = self.solve_ALM(x, obj_fn)
            self.t = 1.2*self.t
            if m/self.t < tol:
                break
        return x
    
    def gradient_augmented_lagrangian(self, x,):
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

# def generate_problem(n=30, num_ineq=6, num_eq=4):
#     tmp_Q = np.random.randn(n, n)
#     Q = tmp_Q.T @ tmp_Q
#     B = np.random.randn(n)
#     A = np.random.randn(num_ineq, n+1)
#     C = np.random.randn(num_eq, n+1)
#     return Q, B, A, C

if __name__ == "__main__":
    # n = 1, num_ineq = 1, num_eq = 1
    # Generate a random problem of size n
    # f(x) = 0.5*xQx + Bx
    # s.t.  A[x,1] <= 0
    #       C[x,1] = 0
    Q = np.array([[1, 0], [0, 1]])
    B = np.array([0, 0])
    A = np.array([[1, 1, 0]])
    C = np.array([[1, 0, 0]])
    print(f"Q: {Q.shape}, B: {B.shape}, A: {A.shape}, C: {C.shape}")
    # Q, B, A, C = generate_problem()
    # print(f"Q: {Q.shape}, B: {B.shape}, A: {A.shape}, C: {C.shape}")
    # dd
    solver = CentralPathSolver(Q, B, A, C)
    x = solver.solve('ALM')
    print(x)