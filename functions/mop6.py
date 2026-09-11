import numpy as np
import math
from .baseFunction import Function
import logging


logger = logging.getLogger(__name__)

class MOP6(Function):
    """
    MOP6

    Definitions
        F = (f1(x,y), f2(x,y)), where

        f1(x) = x
        f2(x) = (1 + 10*y)*(1 - (x/(1 + 10*y))**alpha - (x/(1 + 10*y))*sin(2piqx))

    Constraints
        -5 <= x_i <= 5; i = 1,2,3
        a = 0.8
        b = 3

    """
    def __init__(self) -> None:
        """
        Constructor
        """
        domain = (0, 1)
        super().__init__(domain)

    def f1(self, x: np.ndarray) -> float:
        """
        Función objetivo 1 a minimizar

        :param x: valor flotante
        :return: Valor escalar de la función evaluada en x.
        """
        # Aqui espero algo como [x,y]
        return x[0]

    def f2(self, x: np.ndarray) -> float:
        """
        Función objetivo 2 a minimizar

        :param x: valor flotante
        :return: Valor escalar de la función evaluada en x.
        """
        q = 4
        alpha = 2
        return (1 + 10*x[1])*(1 - (x[0]/(1 + 10*x[1]))**alpha - (x[0]/(1 + 10*x[1]))*math.sin(2*math.pi*q*x[0]))


    def get_samples_domain(self, n_samples: int) -> np.ndarray:
        """
        Genera n cantidad de muestras aleatorias restringido al dominio
        de la función
        
        :param n_samples: número de muestra requeridas
        :return: arreglo de m dimensiones con n_samples
        """
        # El dominio solo es de una dimension 
        return np.random.uniform(0,1, size=(n_samples, 2))

    
    def get_samples_range(self, sample_domain: np.ndarray) -> np.ndarray:
        """
        Se calcula la funciones objetivos dado la muestra del dominio

        :param sample_domain: muestra del dominio
        :return: arreglo de m dimensiones con n_samples
        """

        # Dado que tenemos dos funciones, la dimension de los elementos en el arreglo es de 2
        return np.array([[self.f1(x), self.f2(x)] for x in sample_domain])
