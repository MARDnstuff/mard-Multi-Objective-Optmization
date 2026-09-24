import numpy as np
from scipy.optimize import minimize
from .baseFunction import Function
import logging


logger = logging.getLogger(__name__)

class ZDT1(Function):
    """
    ZDT1 

    Definitions
        F = (f1(x), f2(x, g)), where

        f1(x) = x1
        f2(x) = g(x) * (1 - sqrt(f1/g(x)))

        g(x) = 1 + 9/(n - 1) * SUM(x_i, i=2 to n)

    Constraints
        n = 30
        x_i in [0, 1]

    """
    def __init__(self) -> None:
        """
        Constructor
        """
        domain = (0, 1)
        self.n = 30
        super().__init__(domain)

    def labels_var(self) -> list:
        labels = [f"x{i}" for i in range(1, self.n)]
        return labels


    def _g_helper(self, x: np.ndarray) -> float:
        sum_g = 1 + (9/(self.n - 1) * x[1:self.n + 1].sum())
        return sum_g


    def f1(self, x: np.ndarray) -> float:
        """
        Función objetivo 1 a minimizar

        :param x: valor flotante
        :return: Valor escalar de la función evaluada en x.
        """
        return x[0]
    

    def f2(self, x: np.ndarray) -> float:
        """
        Función objetivo 2 a minimizar

        :param x: valor flotante
        :return: Valor escalar de la función evaluada en x.
        """

        return self._g_helper(x) * (1 - np.sqrt(self.f1(x)/self._g_helper(x)))


    def get_samples_domain(self, n_samples: int) -> np.ndarray:
        return np.random.uniform(self.domain[0], self.domain[1], size=(n_samples, self.n))


    def get_samples_range(self, sample_domain: np.ndarray) -> np.ndarray:
        return np.array([[self.f1(x), self.f2(x)] for x in sample_domain])
