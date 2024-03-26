import numpy as np
from base import BaseSolver
from utils import generate_problem

import matplotlib.pyplot as plt

class CorrectorPrimalDualSolver(BaseSolver):
    def __init__(self, c, A, b, vis=True, correct=True):
        self.c = c
        self.A = A
        self.b = b
        self.values = []
        self.vis = vis
        self.corrector_step = correct

    def solve(self, tol=1e-2, max_iter=100):
        m, n = self.A.shape
        x = np.ones(n)
        y = np.zeros(m)
        s = np.ones(n)

        for _ in range(max_iter):
            X = np.diag(x)
            S = np.diag(s)
            mu = np.dot(x, s) / n
            sigma = 0.98

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
            # solve deltas = [dx, dy, ds], use pseudo-inverse to avoid singular matrix
            deltas = np.linalg.pinv(KKT) @ rhs
            delta_x = deltas[:n]
            delta_y = deltas[n:n+m]
            delta_s = deltas[n+m:]
            
            if self.corrector_step:
                # Corrector step
                Delta_X = np.diag(delta_x)
                Delta_S = np.diag(delta_s)
                
                abs_Delta = np.abs(Delta_X @ Delta_S @ np.ones(n))
                abs_rg = np.abs(rg)
                ratio = abs_rg / abs_Delta
                
                print(f"Delta_X: {np.max(np.abs(Delta_X))}, Delta_S: {np.max(np.abs(Delta_S))}, rg: {np.max(np.abs(rg))}, ratio: {np.max(ratio)}")
                c_rg = -0.1*ratio*Delta_X @ Delta_S @ np.ones(n) + rg
                c_rhs = np.concatenate([rd, rp, c_rg])
                c_deltas = np.linalg.pinv(KKT) @ c_rhs
                delta_x = c_deltas[:n]
                delta_y = c_deltas[n:n+m]
                delta_s = c_deltas[n+m:]
                
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

            self.values.append(np.dot(self.c, x))

        # Generate a plot to show the decrease in the objective value
        if self.vis:
            plt.plot(self.values)
            plt.xlabel("Iteration")
            plt.ylabel("Objective Value")
            plt.title(f"{self.corrector_step}_Primal Dual Solver")
            plt.show()
            plt.savefig(f'{self.corrector_step}_corrector_primal_dual.png')
            plt.cla()

        return x, y, s
    
if __name__ == '__main__':
    c, A, b = generate_problem(num_eq=10, v_num=30)
    solver = CorrectorPrimalDualSolver(c, A, b, correct=True)
    x1, y1, s1 = solver.solve()
    # print(f"{(x1>=0).all()}, {A @ x1 - b}")
    solver = CorrectorPrimalDualSolver(c, A, b, correct=False)
    x, y, s = solver.solve()
    
    print(f"{(x1>=0).all()}, {A @ x1 - b}")
    print(f"{(x>=0).all()}, {A @ x - b}")
    
    