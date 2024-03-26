import numpy as np


# Generate a random problem of size n
# f(x) = cx
# s.t.  Ax = b, x >= 0
def generate_problem(v_num=30, num_eq=10):
    # c = np.random.randn(v_num)
    # A = np.random.randn(num_eq, v_num)
    # b = np.random.randn(num_eq)
    c = np.array([1,1])
    A = np.array([[1,-1]])
    b = np.array([0])
    return c, A, b