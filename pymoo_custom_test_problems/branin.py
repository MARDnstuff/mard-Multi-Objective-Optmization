import numpy as np
from pymoo.core.problem import Problem

class Branin(Problem):
    """
    Branin Function
    source: https://www.sfu.ca/~ssurjano/branin.html
    """

    def __init__(self, a:float = 1.0, b:float = 5.1 / (4 * np.pi**2), c:float = 5 / np.pi, r:float = 6.0, s:float = 10.0, t:float = 1 / (8 * np.pi)) -> None:
        super().__init__(n_var=2, n_obj=1, xl=[-5, 0], xu=[10, 15])
        self.a, self.b, self.c, self.r, self.s, self.t = a, b, c, r, s, t
        self.foo_name = 'Branin'
        self.f_opt = 5 / (4 * np.pi)

    def _evaluate(self, x: np.ndarray, out: dict, *args, **kwargs) -> float:
        """
        :param x: matriz de forma (n_puntos, n_var); cada fila es un punto.
        :param out: diccionario donde se escribe el resultado en out["F"],
                    de forma (n_puntos,) o (n_puntos, 1).
        """
        x1, x2 = x[:, 0], x[:, 1]
        term1 = self.a * (x2 - self.b * x1**2 + self.c * x1 - self.r) ** 2
        term2 = self.s * (1 - self.t) * np.cos(x1)
        out["F"] = term1 + term2 + self.s