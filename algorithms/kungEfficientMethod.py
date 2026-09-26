from .opAlgorithm import OpAlgorithm
import numpy as np


class KungEfficientMethod(OpAlgorithm):

    def __init__(self, objective_space: np.ndarray) -> None:
        """
        Constructor
        """
        # ordena por f1 ascendente
        self.sorted_objective_space_idx: np.ndarray = np.argsort(objective_space[:, 0])
        super().__init__(objective_space)


    def front(self, idx: np.ndarray) -> np.ndarray:
        """
        Calcula el frente de Pareto (conjunto de índices no dominados) 
        sobre el subconjunto de soluciones indicado por ``idx``, usando 
        una estrategia recursiva de divide y vencerás (variante del 
        Kung Efficient Method).

        El algoritmo divide el conjunto de índices en dos mitades, 
        calcula recursivamente el frente de Pareto de cada mitad y 
        luego fusiona ambos resultados descartando las soluciones de la 
        mitad inferior que son dominadas por alguna solución de la mitad 
        superior. Las soluciones de la mitad superior ya están libres de 
        dominancia interna por construcción recursiva, por lo que no es 
        necesario re-evaluarlas contra la mitad inferior.
        """
        if len(idx) == 1:
            return idx

        mid = len(idx) // 2
        T_idx, B_idx = idx[:mid], idx[mid:]

        T_p = self.front(T_idx)
        B_p = self.front(B_idx)

        myMerge = []
        for i in B_p:
            dominated = False
            for j in T_p:
                if self.dominates(self.objective_space[j], self.objective_space[i]):
                    dominated = True
                    break
            if not dominated:
                myMerge.append(i)

        if len(myMerge) == 0:
            return T_p

        return np.concatenate((T_p, np.array(myMerge)))


    def run(self) -> np.ndarray:
        return self.front(self.sorted_objective_space_idx)
    