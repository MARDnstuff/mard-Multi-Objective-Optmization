from .opAlgorithm import OpAlgorithm
import numpy as np


class ContinuouslyUpdated(OpAlgorithm):

    def __init__(self, objective_space: np.ndarray) -> None:
        """
        Constructor
        """
        super().__init__(objective_space)


    def run(self) -> np.ndarray:
        """
        Identifica el conjunto no dominado (Enfoque 2), comparando cada
        nueva solución i contra los miembros ACTUALES de P' (que va
        creciendo y encogiéndose dinámicamente), en vez de comparar
        contra todo el conjunto original desde el inicio.
        """

        # Paso 1: Inicializar a P' con la primera solución
        P_non_dominated_idx = [0]

        # Paso 5: recorremos i = 0, ..., n-1
        for i in range(0, len(self.objective_space)):
            vi = self.objective_space[i]
            j = 0
            i_dominated = False

            # Paso 2-4: comparar i contra lso miembros actuales de P'
            while(j < len(P_non_dominated_idx)):
                pj = self.objective_space[P_non_dominated_idx[j]]

                if self.dominates(vi, pj):
                    # i domina a j -> eliminar a j de P'
                    # swap and pop O(1)
                    P_non_dominated_idx[j] = P_non_dominated_idx[-1]
                    P_non_dominated_idx.pop()
                elif self.dominates(pj, vi):
                    # J domina a i -> i queda descartado , pasamos al siguiente i
                    i_dominated = True
                    break
                else:
                    # Ninguno domina al otro -> segimos omparando con el siguiente j (miembro de P')
                    j += 1

            # Paso 5: Si i sobrevive a todos las comparaciones, se inserta en Pi
            if not i_dominated:
                P_non_dominated_idx.append(i)

        return np.array(P_non_dominated_idx)