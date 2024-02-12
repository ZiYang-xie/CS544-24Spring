import numpy as np

def generate_linear_regression_dataset(N, dim=2):
    w = np.random.randn(1,dim)
    x = np.random.randn(dim, N)
    y = np.dot(w, x) + np.random.randn(1, N) * 0.01

    # linear regression
    w_hat, residuals, rank, s = np.linalg.lstsq(x.T, y.T, rcond=None)
    print(f"Estimated weights: {w_hat.T}")
    print(f"True weights: {w}")
    return x.T, y.T, w


def generate_fitting_dataset(N, func):
    x = np.random.randn(N, 1)
    y = func(x) #+ np.random.randn(1, N) * 0.01
    return x, y