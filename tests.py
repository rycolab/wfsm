import numpy as np
import torch

from wfsm.ad import jacZ as jacZad, hessZ as hessZad
from wfsm.derivatives import jacZ, hessZ
from wfsm.expectation import second_order, second_order_slow
from wfsm.util import gen_wfsm


def test_jacZ(W, start, end):
    W_ad = torch.from_numpy(W).requires_grad_(True)
    start_ad = torch.from_numpy(start)
    end_ad = torch.from_numpy(end)
    truth, pred = jacZad(W_ad, start_ad, end_ad), jacZ(W, start, end)
    truth = truth.detach().numpy()
    assert np.allclose(truth, pred, rtol=1e-5)


def test_hessZ(W, start, end):
    W_ad = torch.from_numpy(W).requires_grad_(True)
    start_ad = torch.from_numpy(start)
    end_ad = torch.from_numpy(end)
    truth, pred = hessZad(W_ad, start_ad, end_ad), hessZ(W, start, end)
    truth = truth.detach().numpy()
    assert np.allclose(truth, pred, rtol=1e-5)


def test_grads(N, A):
    W, start, end = gen_wfsm(N, A)
    test_jacZ(W, start, end)
    test_hessZ(W, start, end)
    # test_order_m(W)


def test_expectation(N, A, R, T):
    W, start, end = gen_wfsm(N, A)
    r = np.exp(np.random.rand(A, N, N, R))
    t = np.exp(np.random.rand(A, N, N, T))
    x1 = second_order(W, start, end, r, t)
    x2 = second_order_slow(W, start, end, r, t)
    assert np.allclose(x1, x2)


if __name__ == '__main__':
    for A in range(1, 5):
        for N in range(3, 10):
            for _ in range(3):
                test_grads(N, A)
            for R in range(5):
                for T in range(5):
                    for _ in range(3):
                        test_expectation(N, A, R, T)