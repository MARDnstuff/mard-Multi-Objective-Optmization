from .opAlgorithm import OpAlgorithm
import numpy as np


class KungEfficientMethod(OpAlgorithm):

    def __init__(self, objective_space: np.ndarray) -> None:
        """
        Constructor
        """
        super().__init__(objective_space)


    def run(self) -> np.ndarray:
        """
        TBD
        """
        pass