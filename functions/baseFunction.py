from abc import ABC, abstractmethod
import numpy as np

class Function(ABC):
    """
    Clase abstracta base para clases tipo Función
    """

    def __init__(self, domain: tuple[float, float]) -> None:
        """
        Constructor
        
        """
        self.domain = domain
