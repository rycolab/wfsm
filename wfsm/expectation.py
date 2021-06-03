import numpy as np
from numba import jit

from wfsa.wfsm import kleene


@jit(nopython=True, parallel=True)
def second_order(W, start, end, r, t):
    """
    Compute the second-order expectation over a WFSM

    r and t are decomposed additively-decomposable functions
    """
    A, _, N = W.shape
    R = r.shape[-1]
    T = t.shape[-1]
    W_sum = np.sum(W, 0)
    W_star = kleene(W_sum)
    s = start @ W_star
    e = W_star @ end
    z = start @ W_star @ end
    r_s, r_e = np.zeros((N, R)), np.zeros((N, R))
    t_s, t_e = np.zeros((N, T)), np.zeros((N, T))
    for a in range(A):
        for i in range(N):
            for j in range(N):
                r_s[j] += W[a, i, j] * s[i] * r[a, i, j]
                t_s[j] += W[a, i, j] * s[i] * t[a, i, j]
                t_e[i] += W[a, i, j] * e[j] * t[a, i, j]
                r_e[i] += W[a, i, j] * e[j] * r[a, i, j]
    exp = np.zeros((R, T))
    for i in range(N):
        for j in range(N):
            exp += W_star[i, j] * (np.outer(r_s[i], t_e[j]) + np.outer(r_e[j], t_s[i]))
            for a in range(A):
                exp += s[i] * e[j] * W[a, i, j] * np.outer(r[a, i, j], t[a, i, j])
    exp /= z
    return exp


@jit(nopython=True, parallel=True)
def second_order_slow(W, start, end, r, t):
    A, _, N = W.shape
    R = r.shape[-1]
    T = t.shape[-1]
    W_sum = np.sum(W, 0)
    W_star = kleene(W_sum)
    s = start @ W_star
    e = W_star @ end
    z = start @ W_star @ end
    exp = np.zeros((R, T))
    for a1 in range(A):
        for i1 in range(N):
            for j1 in range(N):
                exp += s[i1] * e[j1] * W[a1, i1, j1] * np.outer(r[a1, i1, j1], t[a1, i1, j1])
                for a2 in range(A):
                    for i2 in range(N):
                        for j2 in range(N):
                            exp += (
                                (s[i1] * W_star[j1, i2] * e[j2] + s[i2] * W_star[j2, i1] * e[j1])
                                * W[a1, i1, j1] * W[a2, i2, j2] * np.outer(r[a1, i1, j1], t[a2, i2, j2])
                            )
    exp /= z
    return exp