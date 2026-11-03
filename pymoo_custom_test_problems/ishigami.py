import numpy as np
from pymoo.core.problem import Problem

class Ishigami(Problem):
    """
    Ishigami
    source: https://www.sfu.ca/~ssurjano/ishigami.html
    """

    def __init__(self, a: float=7.0, b: float=0.1) -> None:
        super().__init__(n_var=3, n_obj=1, xl=-np.pi, xu=np.pi)
        self.a, self.b = a, b
        self.foo_name = 'Ishigami'
        self.f_opt = -1 - self.b * np.pi**4

    def _evaluate(self, x, out, *args, **kwargs) -> float:
        """
        :param x: matriz de forma (n_puntos, n_var); cada fila es un punto.
        :param out: diccionario donde se escribe el resultado en out["F"],
                    de forma (n_puntos,) o (n_puntos, 1).
        """
        x1, x2, x3 = x[:, 0], x[:, 1], x[:, 2]
        out["F"] = (np.sin(x1) + self.a * np.sin(x2) ** 2 + self.b * x3**4 * np.sin(x1))