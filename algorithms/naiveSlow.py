from .opAlgorithm import OpAlgorithm
import numpy as np


class NaiveSlow(OpAlgorithm):

    def __init__(self, objective_space: np.ndarray) -> None:
        """
        Constructor
        """
        super().__init__(objective_space)


    def run(self) -> np.ndarray:
        """
        Encuentra el conjunto de puntos no dominados (frente de Pareto)
        dentro de un conjunto de puntos en el espacio de objetivos.

        Para cada punto i, se compara contra TODOS los demás puntos j del conjunto
        (no basta con encontrar un solo j que no lo domine). Un punto i se considera
        no dominado únicamente si, tras revisar todos los puntos j, ninguno de ellos domina a i. 
        
        Si se encuentra al menos un j que domina a i, el punto se descarta inmediatamente y se pasa al siguiente i.
        """
        P_non_dominated_idx = []

        for i in range(0, len(self.objective_space)):
            dominated = False
            for j in range(0, len(self.objective_space)):
                if i != j and self.dominates(self.objective_space[j], self.objective_space[i]):
                    dominated = True
                    break
            if not dominated:
                P_non_dominated_idx.append(i)

        return np.array(P_non_dominated_idx)

    