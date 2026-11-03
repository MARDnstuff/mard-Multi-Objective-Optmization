import numpy as np
from pymoo.core.problem import Problem

class RastriginFunction(Problem):
    """
    Rastringin Function
    source: https://www.sfu.ca/~ssurjano/rastr.html
    """
    def __init__(self, n_var: int =2, A: float=10.0) -> None:
        super().__init__(n_var=n_var, n_obj=1, xl=-5.12, xu=5.12, vtype=float)
        self.A = A
        self.foo_name = 'Rastrigin'
        self.f_opt = 0.0

    def _evaluate(self, x: np.ndarray, out: dict, *args, **kwargs) -> float:
        """
        :param x: matriz de forma (n_puntos, n_var); cada fila es un punto.
        :param out: diccionario donde se escribe el resultado en out["F"],
                    de forma (n_puntos,) o (n_puntos, 1).
        """
        d = x.shape[1]
        out["F"] = self.A * d + np.sum(x**2 - self.A * np.cos(2 * np.pi * x), axis=1)
    