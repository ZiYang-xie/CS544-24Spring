import numpy as np
import z3

def generate_problem(v_num=30, num_eq=10):
    c = np.abs(np.random.randn(v_num))
    A = np.random.randn(num_eq, v_num)
    b = np.random.randn(num_eq)
    return c, A, b

def verify_constraints(A, b):
    x = z3.RealVector('x', A.shape[1])
    constraints = [z3.Sum([A[i,j] * x[j] for j in range(A.shape[1])]) == b[i] for i in range(A.shape[0])]
    s = z3.Solver()
    s.add(constraints)
    return True if s.check() == z3.sat else False

if __name__ == "__main__":
    c, A, b = generate_problem()
    print(verify_constraints(A, b))