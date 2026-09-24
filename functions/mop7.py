import numpy as np
import math
from .baseFunction import Function
import logging


logger = logging.getLogger(__name__)

class MOP7(Function):
    """
    MOP7

    Definitions
        F = (f1(x,y), f2(x,y), f3(x,y)), where

        f1(x,y) = (x - 2)^2 / 2 + (y + 1)^2 / 13 + 3
        f2(x,y) = (x + y - 3)^2 / 36 + (-x + y + 2)^2 / 8 -17
        f3(x,y) = (x + 2y - 1)^2 / 175 + (2y - x)^2 / 17 - 13

    Constraints
        -400 <= x,y <= 400

    """
    def __init__(self) -> None:
        """
        Constructor
        """
        domain = (-2, 2)
        self.n = 2
        super().__init__(domain)

    def f1(self, x: np.ndarray) -> float:
        """
        Función objetivo 1 a minimizar

        :param x: valor flotante
        :return: Valor escalar de la función evaluada en x.
        """
        # Aqui espero algo como x = [x,y]
        return ((x[0] - 2)**2)/2 + ((x[1] + 1)**2/13) + 3

    def f2(self, x: np.ndarray) -> float:
        """
        Función objetivo 2 a minimizar

        :param x: valor flotante
        :return: Valor escalar de la función evaluada en x.
        """
        return ((x[0] + x[1] - 3)**2)/36 + ((-x[0] + x[1] + 2)**2)/8 - 17

    def f3(self, x: np.ndarray) -> float:
        """
        Función objetivo 3 a minimizar

        :param x: valor flotante
        :return: Valor escalar de la función evaluada en x.
        """
        return ((x[0] + 2*x[1] - 1)**2 / 175) + ((2*x[1] - x[0])**2)/17 - 13


    def get_samples_domain(self, n_samples: int) -> np.ndarray:
        """
        Genera n cantidad de muestras aleatorias restringido al dominio
        de la función
        
        :param n_samples: número de muestra requeridas
        :return: arreglo de m dimensiones con n_samples
        """
        # El dominio solo es de una dimension 
        return np.random.uniform(*self.domain, size=(n_samples, 2))

    
    def get_samples_range(self, sample_domain: np.ndarray) -> np.ndarray:
        """
        Se calcula la funciones objetivos dado la muestra del dominio

        :param sample_domain: muestra del dominio
        :return: arreglo de m dimensiones con n_samples
        """

        # Dado que tenemos dos funciones, la dimension de los elementos en el arreglo es de 2
        return np.array([[self.f1(x), self.f2(x), self.f3(x)] for x in sample_domain])
