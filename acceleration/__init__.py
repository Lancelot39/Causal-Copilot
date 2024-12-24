import cupy as cp

def rbf_kernel(X, gamma):
    pairwise_sq_dists = cp.sum((X[:, None, :] - X[None, :, :]) ** 2, axis=-1)
    return cp.exp(-gamma * pairwise_sq_dists)
