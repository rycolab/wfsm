import torch


def kleene(W):
    N = W.shape[0]
    return torch.inverse(torch.eye(N) - W)


def Z(W, start, end):
    W_sum = W.sum(0)
    W_star = kleene(W_sum)
    return start @ W_star @ end


def jacZ(W, start, end):
    z = Z(W, start, end)
    jac = torch.autograd.grad(z, [W])[0]
    return jac


def hessZ(W, start, end):
    A, _, N = W.shape
    z = Z(W, start, end)
    jac = torch.autograd.grad(z, [W], create_graph=True, retain_graph=True)[0]
    hess = torch.zeros((A, N, N, A, N, N)).float()
    for a in range(A):
        for i in range(N):
            for j in range(N):
                hess[a, i, j] = torch.autograd.grad(jac[a, i, j], [W], retain_graph=True)[0]
    return hess
