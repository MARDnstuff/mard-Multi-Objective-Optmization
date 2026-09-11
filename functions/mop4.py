import numpy as np
import math
from .baseFunction import Function
import logging


logger = logging.getLogger(__name__)

class MOP4(Function):
    """
    MOP4

    Definitions
        F = (f1(x), f2(x)), where

        f1(x) = sum from i = 1 to n - 1  of  -10e^(-0.2*sqrt(x_i^2 + x_i+1^2)
        f2(x) = sum from i = 1 to n of abs(x_i)^a + 5*sin(xi)^b

    Constraints
        -5 <= x_i <= 5; i = 1,2,3
        a = 0.8
        b = 3

    """
    def __init__(self) -> None:
        """
        Constructor
        """
        domain = (-1.5, 1.5)
        super().__init__(domain)

    def f1(self, x: np.ndarray) -> float:
        """
        Función objetivo 1 a minimizar

        :param x: valor flotante
        :return: Valor escalar de la función evaluada en x.
        """
        # Aqui espero algo como x = [x1, x2, x3]
        res = 0
        for i in range(0, len(x) - 1):
            pwr = -0.2*math.sqrt(x[i]**2 + x[i+1]**2)
            fx = -10*math.exp(pwr)
            res += fx
        return res

    def f2(self, x: np.ndarray) -> float:
        """
        Función objetivo 2 a minimizar

        :param x: valor flotante
        :return: Valor escalar de la función evaluada en x.
        """
        a = 0.8
        b = 3
        res = 0
        for i in range(0, len(x)):
            fx = abs(x[i])**a + (5*math.sin(x[i]))**b
            res += fx
        return res


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
