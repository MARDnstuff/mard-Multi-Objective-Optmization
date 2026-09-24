import numpy as np
import math
from .baseFunction import Function
import logging


logger = logging.getLogger(__name__)

class MOP3(Function):
    """
    MOP3

    Definitions
        F = (f1(x,y), f2(x,y)), where

        f1(x, y) = - [1 + (A1 - B1)^2 + (A2 - B2)^2] 
        f2(x, y) = - [(x + 3)^2 + (y + 1)^2]

    Constraints
        -3.1416 <= x,y <= 3.1416
        A1 = 0.5 sin(1) - 2cos(1) + sin(2) - 1.5cos(2)
        A2 = 1.5 sin(1) - cos(1) + 2sin(2) - 0.5cos(2)
        B1 = 0.5 sin(x) - 2cos(x) + sin(y) - 1.5cos(y)
        B2 = 1.5 sin(x) - cos(x) + 2sin(y) - 0.5cos(y)

    """
    def __init__(self) -> None:
        """
        Constructor
        """
        domain = (-3.1416, 3.1416)
        self.n = 2
        super().__init__(domain)

    def f1(self, x: np.ndarray) -> float:
        """
        Función objetivo 1 a minimizar

        :param x: valor flotante
        :return: Valor escalar de la función evaluada en x.
        """
        # Aqui espero algo como [x, y]
        A1 = 0.5*math.sin(1) - 2*math.cos(1) + math.sin(2) - 1.5*math.cos(2)
        A2 = 1.5*math.sin(1) - math.cos(1) + 2*math.sin(2) - 0.5*math.cos(2)
        B1 = 0.5*math.sin(x[0]) - 2*math.cos(x[0]) + math.sin(x[1]) - 1.5*math.cos(x[1])
        B2 = 1.5*math.sin(x[0]) - math.cos(x[0]) + 2*math.sin(x[1]) - 0.5*math.cos(x[1])
        return -1*(1 + (A1 - B1)**2 + (A2 - B2)**2)

    def f2(self, x: np.ndarray) -> float:
        """
        Función objetivo 2 a minimizar

        :param x: valor flotante
        :return: Valor escalar de la función evaluada en x.
        """
        # Aqui espero algo como [x, y]
        return -1*((x[0] + 3)**2 + (x[1] + 1)**2)


    def get_samples_domain(self, n_samples: int) -> np.ndarray:
        """
        Genera n cantidad de muestras aleatorias restringido al dominio
        de la función
        
        :param n_samples: número de muestra requeridas
        :return: arreglo de m dimensiones con n_samples
        """
        return np.random.uniform(*self.domain, size=(n_samples, 2))

    
    def get_samples_range(self, sample_domain: np.ndarray) -> np.ndarray:
        """
        Se calcula la funciones objetivos dado la muestra del dominio

        :param sample_domain: muestra del dominio
        :return: arreglo de m dimensiones con n_samples
        """
        # Dado que tenemos dos funciones, la dimension de los elementos en el arreglo es de 2
        return np.array([[self.f1(x), self.f2(x)] for x in sample_domain])
