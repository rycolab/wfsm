import numpy as np
import numpy.linalg as LA
from numba import jit


@jit(nopython=True)
def kleene(W):
    """
    Compute the Kleene closure of a matrix

    W must have a spectral radius less than 1
    """
    N = W.shape[0]
    return LA.inv(np.eye(N) - W)


@jit(nopython=True)
def Z(W, start, end):
    """
    Compute normalization constant of a WFSM
    """
    W_sum = np.sum(W, 0)
    W_star = kleene(W_sum)
    return start @ W_star @ end


@jit(nopython=True, parallel=True)
def jacZ(W, start, end):
    """
    Evaluate Jacobian of the normalization constant of a WFSM
    """
    A, _, N = W.shape
    W_sum = np.sum(W, 0)
    W_star = kleene(W_sum)
    s = start @ W_star
    e = W_star @ end
    J = np.zeros((A, N, N))
    for i in range(N):
        for j in range(N):
            J[:, i, j] = s[i] * e[j]
    return J


@jit(nopython=True, parallel=True)
def hessZ(W, start, end):
    """
    Evaluate Hessian of the normalization constant of a WFSM
    """
    A, _, N = W.shape
    W_sum = np.sum(W, 0)
    W_star = kleene(W_sum)
    s = start @ W_star
    e = W_star @ end
    H = np.zeros((A, N, N, A, N, N))
    for i in range(N):
        for j in range(N):
            for k in range(N):
                for l in range(N):
                    H[:, i, j, :, k, l] = (
                            s[i] * W_star[j, k] * e[l] +
                            s[k] * W_star[l, i] * e[j]
                    )
    return H
