import numpy as np
from pymoo.core.problem import Problem

class Park91a(Problem):
    """
    Park (1991) Function 1
    source: https://www.sfu.ca/~ssurjano/park91a.html
    
    High fidelity:
        f(x) = x1/2 * [sqrt(1 + (x2 + x3^2) x4 / x1^2) - 1] + (x1 + 3 x4) exp(1 + sin x3)
    Low fidelity (Xiong et al. 2013):
        f_L(x) = (1 + sin(x1)/10) f(x) - 2 x1 + x2^2 + x3^2 + 0.5 x_i in [0, 1).

    Note: the first term is evaluated in the algebraically equivalent form
    0.5 * (sqrt(x1^2 + (x2 + x3^2) x4) - x1), which is identical for x1 > 0
    and stays finite at x1 = 0 (the original form divides by x1^2).
    
    """

    def __init__(self, low_fidelity: bool=False) -> None:
        super().__init__(n_var=4, n_obj=1, xl=0.0, xu=1.0)
        self.low_fidelity = low_fidelity
        self.foo_name = 'Park91a'
        self.f_opt = 0.5 if self.low_fidelity else 0.0

    @staticmethod
    def high(x: np.ndarray) -> np.ndarray:
        x1, x2, x3, x4 = x[:, 0], x[:, 1], x[:, 2], x[:, 3]
        term1 = 0.5 * (np.sqrt(x1**2 + (x2 + x3**2) * x4) - x1)
        term2 = (x1 + 3 * x4) * np.exp(1 + np.sin(x3))
        return term1 + term2

    def _evaluate(self, x: np.ndarray, out: dict, *args, **kwargs) -> float:
        """
        :param x: matriz de forma (n_puntos, n_var); cada fila es un punto.
        :param out: diccionario donde se escribe el resultado en out["F"],
                    de forma (n_puntos,) o (n_puntos, 1).
        """
        yh = self.high(x)
        # Alternate form
        if self.low_fidelity:
            x1, x2, x3 = x[:, 0], x[:, 1], x[:, 2]
            yh = (1 + np.sin(x1) / 10) * yh - 2 * x1 + x2**2 + x3**2 + 0.5
        out["F"] = yh