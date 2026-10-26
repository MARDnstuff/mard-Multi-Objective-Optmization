import numpy as np
import math
from scipy.optimize import minimize
from .baseFunction import Function
import logging


logger = logging.getLogger(__name__)

class ZDT4(Function):
    """
    ZDT4 

    Definitions

        f1(x) = x1
        f2(x, g) = g(x) * (1 - sqrt(f1/g(x)))

        g(x) = 1 + 10*(n - 1) * SUM(x_i**2 - 10*cos-(4pix_i), i=2 to n)

    Constraints
        n = 10
        x_i in [0, 1]
        x_2,...,n  in [-5, 5]

    """
    def __init__(self) -> None:
        """
        Constructor
        """
        domain = (-5, 5)
        self.x_domain = (0, 1)
        self.n = 10
        self.f1_invok = 0
        self.f2_invok = 0
        super().__init__(domain, False)

    def labels_var(self) -> list:
        labels = [f"x{i}" for i in range(1, self.n)]
        return labels


    def _g_helper(self, x: np.ndarray) -> float:
        mysum = 0
        for i in range(1, self.n):
            mysum += x[i]**2 - 10*math.cos(4*math.pi*x[i])

        sum_g = 1 + (10*(self.n - 1) + mysum)
        return sum_g


    def f1(self, x: np.ndarray) -> float:
        """
        Función objetivo 1 a minimizar

        :param x: valor flotante
        :return: Valor escalar de la función evaluada en x.
        """
        self.f1_invok += 1
        return x[0]
    

    def f2(self, x: np.ndarray) -> float:
        """
        Función objetivo 2 a minimizar

        :param x: valor flotante
        :return: Valor escalar de la función evaluada en x.
        """
        self.f2_invok += 1
        return self._g_helper(x) * (1 - np.sqrt(self.f1(x)/self._g_helper(x)))


    def get_samples_domain(self, n_samples: int) -> list[np.ndarray]:
        """
        Retorna n muestra dentro del rango factible de cada variables
        """
        x1 = np.random.uniform(self.x_domain[0], self.x_domain[1], size=(n_samples, 1))
        x2_n = np.random.uniform(self.domain[0], self.domain[1], size=(n_samples, self.n - 1))

        samples = []
        for xi, xj in zip(x1, x2_n):
            samples.append(np.concatenate((xi, xj)))
        return np.array(samples)

    def get_samples_range(self, sample_domain: np.ndarray) -> np.ndarray:
        return np.array([[self.f1(x), self.f2(x)] for x in sample_domain])
