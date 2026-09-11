import numpy as np
from .baseFunction import Function
import logging


logger = logging.getLogger(__name__)

class MOP1(Function):
    """
    MOP1 

    Definitions
        F = (f1(x), f2(x)), where

        f1(x) = x^2
        f2(x) = (x - 2)^2

    Constraints
        -10^5 <= x <= 10^5

    """
    def __init__(self) -> None:
        """
        Constructor
        """
        domain = (-10**5, 10**5)
        super().__init__(domain)

    def f1(self, x: float) -> float:
        """
        Función objetivo 1 a minimizar

        :param x: valor flotante
        :return: Valor escalar de la función evaluada en x.
        """
        if not (x >= self.domain[0] and x <= self.domain[1]):
            raise ValueError("x value is out of the valid range")
        
        return x**2

    def f2(self, x: float) -> float:
        """
        Función objetivo 2 a minimizar

        :param x: valor flotante
        :return: Valor escalar de la función evaluada en x.
        """
        if not (x >= self.domain[0] and x <= self.domain[1]):
            raise ValueError("x value is out of the valid range")
        return (x - 2)**2


    def get_samples_domain(self, n_samples: int) -> np.ndarray:
        """
        Genera n cantidad de muestras aleatorias restringido al dominio
        de la función
        
        :param n_samples: número de muestra requeridas
        :return: arreglo de m dimensiones con n_samples
        """
        # El dominio solo es de una dimension 
        return np.random.uniform(-5, 5, size=(n_samples, 1))

    
    def get_samples_range(self, sample_domain: np.ndarray) -> np.ndarray:
        """
        Se calcula la funciones objetivos dado la muestra del dominio

        :param sample_domain: muestra del dominio
        :return: arreglo de m dimensiones con n_samples
        """

        # Dado que tenemos dos funciones, la dimension de los elementos en el arreglo es de 2
        return np.array([[self.f1(x[0]), self.f2(x[0])] for x in sample_domain])
