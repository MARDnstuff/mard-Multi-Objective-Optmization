from abc import ABC, abstractmethod
import numpy as np

class OpAlgorithm(ABC):
    """
    Clase abstracta base para clases tipo Algoritmo de Optimización
    """

    def __init__(self, objective_space: np.ndarray) -> None:
        """
        Constructor
        """
        self.objective_space: np.ndarray = objective_space
        self.f_count_comparison: int = 0

    def dominates(self, fx: np.ndarray, fy: np.ndarray) -> bool:
        """
        Se tiene que cumplir que fxi <= fyi, 

        todos tiene que cumplir con que sean menor o igual, y al menos uno debe ser estrictamente
        menor que uno de los elementos para considerar que x domina a y.
        
        """
        self.f_count_comparison += 1
        all_less_or_equal = all(fxi <= fyi for fxi, fyi in zip(fx, fy))
        at_least_one = any(fxi < fyi for fxi, fyi in zip(fx, fy))
        return all_less_or_equal and at_least_one
