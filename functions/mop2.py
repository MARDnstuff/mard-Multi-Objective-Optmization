import numpy as np
import math
from .baseFunction import Function
import logging


logger = logging.getLogger(__name__)

class MOP2(Function):
    """
    MOP1 

    Definitions
        F = (f1(x), f2(x)), where

        f1(x) = 1 - e(- sum from i = 1 to n of (xi - 1/sqrt(n))^2
        f2(x) = 1 - e(- sum from i + 1 to n of (xi + 1/sqrt(n))^2

    Constraints
        -4 <= x_i <= 4; i = 1,2,3

    """
    def __init__(self) -> None:
        """
        Constructor
        """
        domain = (-1, 1)
        self.n = 3
        super().__init__(domain)

    def f1(self, x: np.ndarray) -> float:
        """
        Función objetivo 1 a minimizar

        :param x: valor flotante
        :return: Valor escalar de la función evaluada en x.
        """
        # Aqui espero algo como x = [x1, x2, x3]
        n = len(x)
        res = 0
        for i in range(0, len(x)):
            op = (x[i] - 1/(math.sqrt(n)))**2
            res += op
        fx = 1 - math.exp(-1*res)
        return fx

    def f2(self, x: np.ndarray) -> float:
        """
        Función objetivo 2 a minimizar

        :param x: valor flotante
        :return: Valor escalar de la función evaluada en x.
        """
        # Aqui espero algo como x = [x1, x2, x3]
        n = len(x)
        res = 0
        for i in range(0, len(x)):
            op = (x[i] + 1/(math.sqrt(n)))**2
            res += op
        fx = 1 - math.exp(-1*res)
        return fx


    def get_samples_domain(self, n_samples: int) -> np.ndarray:
        """
        Genera n cantidad de muestras aleatorias restringido al dominio
        de la función
        
        :param n_samples: número de muestra requeridas
        :return: arreglo de m dimensiones con n_samples
        """
        # El dominio solo es de una dimension 
        return np.random.uniform(*self.domain, size=(n_samples, 3))

    
    def get_samples_range(self, sample_domain: np.ndarray) -> np.ndarray:
        """
        Se calcula la funciones objetivos dado la muestra del dominio

        :param sample_domain: muestra del dominio
        :return: arreglo de m dimensiones con n_samples
        """

        # Dado que tenemos dos funciones, la dimension de los elementos en el arreglo es de 2
        return np.array([[self.f1(x), self.f2(x)] for x in sample_domain])
