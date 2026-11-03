import numpy as np
from pymoo.core.problem import Problem

class Park91b(Problem):
    """
    Park (1991) Function 2
    source: http://sfu.ca/~ssurjano/park91b.html

    High fidelity:  f(x) = (2/3) exp(x1 + x2) - x4 sin(x3) + x3
    Low fidelity (Xiong et al. 2013):  f_L(x) = 1.2 f(x) - 1
    x_i in [0, 1).
    
    """

    def __init__(self, low_fidelity: bool=False) -> None:
        super().__init__(n_var=4, n_obj=1, xl=0.0, xu=1.0)
        self.low_fidelity = low_fidelity
        self.foo_name = 'Park91b'
        self.f_opt = -0.2 if self.low_fidelity else 2 / 3

    def _evaluate(self, x:np.ndarray, out: dict, *args, **kwargs) -> float:
        """
        :param x: matriz de forma (n_puntos, n_var); cada fila es un punto.
        :param out: diccionario donde se escribe el resultado en out["F"],
                    de forma (n_puntos,) o (n_puntos, 1).
        """
        x1, x2, x3, x4 = x[:, 0], x[:, 1], x[:, 2], x[:, 3]
        y = (2 / 3) * np.exp(x1 + x2) - x4 * np.sin(x3) + x3
        out["F"] = 1.2 * y - 1 if self.low_fidelity else y