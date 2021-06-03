# Inference Algorithms for Weighted Finite-State Machines
This library contains an implementation for finding derivatives for Weighted
Finite-State Machines (WFSMs).
A detalied description of these algorithms and their runtimes, including
proofs of correctness can be found in
["Higher-order Derivatives of Weighted Finite-state Machines"](https://arxiv.org/abs/2106.00749)

## Citation

This code is for the papers _Please Mind the Root: Decoding Arborescences for Dependency Parsing_ and
_On Finding the K-best Non-projective Dependency Trees_ featured in EMNLP 2020 and ACL 2021 respectively.
Please cite as:

```bibtex
@inproceedings{zmigrod-etal-2021-wfsm,
    title = "Higher-order Derivatives of Weighted Finite-state Machines",
    author = "Zmigrod, Ran  and
      Vieira, Tim  and
      Cotterell, Ryan",
    year = "2021",
    url = "https://arxiv.org/abs/2106.00749",
}
```

## Requirements and Installation

* Python version >= 3.6

Installation:
```bash
git clone https://github.com/rycolab/wfsm
cd wfsm
pip install -e .
```


## Example
We take a purely linear algebraic approach to WFSM.
We characterize a WFSM with `A` symbols (including the empty string) and `N` states
by a `A x N x N` matrix for the transition weights.
We represent the start and end weights using `start` and `end`.
To find `Z`, its Jacobian, and its Hessian can be done with the following
```python
import numpy as np
from wfsm.util import gen_wfsm
from wfsm.derivatives import Z, jacZ, hessZ

A = 3
N = 5

W, start, end = gen_wfsm(A, N)
z = Z(W, start, end)
jac = jacZ(W, start, end)
hess = hessZ(W, start, end)
```
We will upload code for mth order derivatives soon.

This library also contains code for finding second-order expectations.
```python
from wfsm.expectation import second_order

# Randomly generated additively-decomposable functions
r = np.exp(np.random.rand(A, N, N, R))
t = np.exp(np.random.rand(A, N, N, T))
e = second_order(W, start, end, r, t)
```