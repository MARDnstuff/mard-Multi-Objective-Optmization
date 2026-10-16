
from scipy.optimize import minimize
import numpy as np
import logging


logger = logging.getLogger(__name__)


def foo(x: np.ndarray) -> float:
    return x[0]**2 + x[1]**2

if __name__ == "__main__":
    x0 = np.array([100, 100])
    x1 = np.array([-1, 1])

    methods = [
        'Nelder-Mead',
        'Powell',
        'CG',
        'BFGS',
        #'Newton-CG',
        'L-BFGS-B',
        'TNC',
        'COBYLA',
        'COBYQA',
        'SLSQP',
        'trust-constr',
        #'dogleg',
        #'trust-ncg',
        # 'trust-krylov',
        # 'trust-exact'
    ]

    # XO
    for method in methods:
        t = minimize(foo, x0, method=method)
        print(f"\n ======= [X0] {method} ======= \n")
        print(t)


    # X1
    for method in methods:
        t = minimize(foo, x1, method=method)
        print(f"\n ======= [X1] {method} ======= \n")
        print(t)
    