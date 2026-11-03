import numpy as np
from pymoo.core.problem import Problem

class RoosArnold63(Problem):
    """
    Roos & Arnold (1963) Function
    source: https://www.sfu.ca/~ssurjano/roosarn63.html
    
    f(x) = prod_i |4 x_i - 2|,  x_i in [0, 1], any dimension d.
    Minimum f = 0 wherever some x_i = 0.5.
    
    """
    def __init__(self, n_var: int=5) -> None:
        super().__init__(n_var=n_var, n_obj=1, xl=0.0, xu=1.0)
        self.foo_name = 'RoosArnold63'
        self.f_opt = 0.0

    def _evaluate(self, x: np.ndarray, out: dict, *args, **kwargs) -> float:
        """
        :param x: matriz de forma (n_puntos, n_var); cada fila es un punto.
        :param out: diccionario donde se escribe el resultado en out["F"],
                    de forma (n_puntos,) o (n_puntos, 1).
        """
        out["F"] = np.prod(np.abs(4 * x - 2), axis=1)