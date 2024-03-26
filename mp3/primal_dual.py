import numpy as np
from base import BaseSolver
from utils import generate_problem

class PrimalDualSolver(BaseSolver):
    def __init__(self, c, A, b):
        self.c = c
        self.A = A
        self.b = b

    def solve(self, tol=1e-8, max_iter=100):
        m, n = self.A.shape
        x = np.ones(n)
        y = np.zeros(m)
        s = np.ones(n)

        for _ in range(max_iter):
            X = np.diag(x)
            S = np.diag(s)
            mu = np.dot(x, s) / n
            sigma = 0.9

            rp = self.b - self.A @ x
            rd = self.c - self.A.T @ y - s
            rg = -X @ S @ np.ones(n) + sigma * mu * np.ones(n)
            
            # Form and solve the KKT system
            KKT = np.block([
                [np.zeros((n, n)), self.A.T, np.eye(n)],
                [self.A, np.zeros((m, m)), np.zeros((m, n))],
                [S, np.zeros((n, m)), X]
            ])

            rhs = np.concatenate([rd, rp, rg])
            # solve deltas = [dx, dy, ds]
            deltas = np.linalg.solve(KKT, rhs) 
            delta_x = deltas[:n]
            delta_y = deltas[n:n+m]
            delta_s = deltas[n+m:]

            # Line search parameters for step size
            if (delta_x <= 0).any():
                alpha_p = min(1, 0.9 * min(-x[delta_x <= 0] / delta_x[delta_x <= 0]))
            else:
                alpha_p = 1
            if (delta_s <= 0).any():
                alpha_d = min(1, 0.9 * min(-s[delta_s <= 0] / delta_s[delta_s <= 0]))
            else:
                alpha_d = 1
            
            if np.isnan(alpha_p) or np.isinf(alpha_p):
                alpha_p = 1
            if np.isnan(alpha_d) or np.isinf(alpha_d):
                alpha_d = 1

            # Update x, y, s
            x += alpha_p * delta_x
            s += alpha_d * delta_s
            y += alpha_d * delta_y

            # Check for convergence
            rp_norm = np.linalg.norm(rp)
            rd_norm = np.linalg.norm(rd)
            gap = np.dot(x, s)
            if rp_norm < tol and rd_norm < tol and gap < tol:
                break

            print(f"Iteration: {_}, Primal Residual: {rp_norm}, Dual Residual: {rd_norm}, Gap: {gap}")
            print(f"Value: {np.dot(self.c, x)}")

        return x, y, s
    
if __name__ == '__main__':
    c, A, b = generate_problem()
    solver = PrimalDualSolver(c, A, b)
    x, y, s = solver.solve()
    
    