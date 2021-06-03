import numpy as np
import numpy.linalg as LA


def gen_wfsm(N, A):
    W = np.exp(2 * np.random.randn(A, N, N))
    lbound = np.max(np.real(LA.eig(W)[0]) + 1)
    W = W / lbound
    start = np.exp(2 * np.random.randn(N))
    end = np.exp(2 * np.random.randn(N))
    return W, start, end
