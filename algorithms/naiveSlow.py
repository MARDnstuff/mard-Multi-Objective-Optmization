import numpy as np

COUNT_COMPARISON = 0

def dominates(fx: np.ndarray, fy: np.ndarray) -> bool:
    """
    Se tiene que cumplir que fxi <= fyi, 

    todos tiene que cumplir con que sean menor o igual, y al menos uno debe ser estrictamente
    menor que uno de los elementos para considerar que x domina a y.
    
    """
    global COUNT_COMPARISON
    all_less_or_equal = all(fxi <= fyi for fxi, fyi in zip(fx, fy))
    at_least_one = any(fxi < fyi for fxi, fyi in zip(fx, fy))
    COUNT_COMPARISON += 1
    return all_less_or_equal and at_least_one


def naive_slow_optimization(objective_space: np.ndarray) -> np.ndarray:
    """
    Encuentra el conjunto de puntos no dominados (frente de Pareto)
    dentro de un conjunto de puntos en el espacio de objetivos.

    Para cada punto i, se compara contra TODOS los demás puntos j del conjunto
    (no basta con encontrar un solo j que no lo domine). Un punto i se considera
    no dominado únicamente si, tras revisar todos los puntos j, ninguno de ellos domina a i. 
    
    Si se encuentra al menos un j que domina a i, el punto se descarta inmediatamente y se pasa al siguiente i.
    """
    P_non_dominated_idx = []

    for i in range(0, len(objective_space)):
        dominated = False
        for j in range(0, len(objective_space)):
            if i != j and dominates(objective_space[j], objective_space[i]):
                dominated = True
                break
        if not dominated:
            P_non_dominated_idx.append(i)

    return np.array(P_non_dominated_idx)

    