import numpy as np
from .base import BaseSolver

import matplotlib.pyplot as plt
import wandb

class PrimalDualSolver(BaseSolver):
    def __init__(self, c, A, b, sigma=0.95, use_wandb=False, vis=True):
        self.c = c
        self.A = A
        self.b = b
        self.sigma = sigma
        self.history = {
            'values': [],
            'dual_measures': [],
            'rp_norms': [],
            'rd_norms': [],
        }
        self.vis = vis
        self.use_wandb = use_wandb

        if self.use_wandb:
            self.experiment_name = 'primal_dual'
            wandb.init(project='cs544', name=self.experiment_name)
            wandb.run.tags = [self.experiment_name]
            wandb.config.update({
                "num_eqs": A.shape[0],
                "num_vars": A.shape[1],
                "sigma": sigma,
            })

    def solve(self, tol=1e-2, max_iter=500):
        m, n = self.A.shape
        x = np.ones(n)
        y = np.zeros(m)
        s = np.ones(n)

        for _ in range(max_iter):
            X = np.diag(x)
            S = np.diag(s)
            mu = np.dot(x, s) / n

            rp = self.b - self.A @ x
            rd = self.c - self.A.T @ y - s
            rg = -X @ S @ np.ones(n) + self.sigma * mu * np.ones(n)
            
            # Form and solve the KKT system
            KKT = np.block([
                [np.zeros((n, n)), self.A.T, np.eye(n)],
                [self.A, np.zeros((m, m)), np.zeros((m, n))],
                [S, np.zeros((n, m)), X]
            ])
            rhs = np.concatenate([rd, rp, rg])
            # solve deltas = [dx, dy, ds], use pseudo-inverse to avoid singular matrix
            deltas = np.linalg.pinv(KKT) @ rhs
            delta_x = deltas[:n]
            delta_y = deltas[n:n+m]
            delta_s = deltas[n+m:]

            # Line search parameters for step size
            alpha_p = min(1, 0.9 * min(-x[delta_x <= 0] / delta_x[delta_x <= 0])) if np.any(delta_x < 0) else 1
            alpha_d = min(1, 0.9 * min(-s[delta_s <= 0] / delta_s[delta_s <= 0])) if np.any(delta_s < 0) else 1

            # Update x, y, s
            x += alpha_p * delta_x
            s += alpha_d * delta_s
            y += alpha_d * delta_y

            # Check for convergence
            if mu < tol:
                break

            print(f"Iteration: {_}, duality measure: {mu}")
            print(f"Value: {np.dot(self.c, x)}")

            if self.use_wandb:
                wandb.log({
                    'value': np.dot(self.c, x), 
                    'duality_measure': mu,
                    'rp_norm': np.linalg.norm(rp),
                    'rd_norm': np.linalg.norm(rd),
                })
            self.history['values'].append(np.dot(self.c, x))
            self.history['dual_measures'].append(mu)
            self.history['rp_norms'].append(np.linalg.norm(rp))
            self.history['rd_norms'].append(np.linalg.norm(rd))


        result = {
            'x': x,
            's': s,
            'lambda': y,
            'value': np.dot(self.c, x),
            'value_history': self.history['values'],
            'duality_measure': mu,
            'duality_measure_history': self.history['dual_measures'],
        }
        if self.use_wandb:
            wandb.finish()
        return result
    
if __name__ == '__main__':
    from utils import generate_problem
    c, A, b = generate_problem(num_eq=10, v_num=30)
    solver = PrimalDualSolver(c, A, b)
    x, y, s = solver.solve()
    
    