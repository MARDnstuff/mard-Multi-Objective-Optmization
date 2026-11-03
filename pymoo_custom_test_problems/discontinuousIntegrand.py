import numpy as np
from pymoo.core.problem import Problem

class DiscontinuousIntegrand(Problem):
    """
    Discontinuous Integrand Family
    source: https://www.sfu.ca/~ssurjano/disc.html
    
    f(x) = 0            if x1 > u1 or x2 > u2
    exp(sum_i a_i x_i)     otherwise
    
    x_i in [0, 1]. Defaults u_i = 0.5, a_i = 5 (as in the SFU code).
    As in the SFU definition, the discontinuity involves only x1 and x2.
    Intended as an integration test; as an optimization problem, minimize -f
    to look for its maximum (set maximize=True).
    
    """

    def __init__(self, n_var:int = 2, u: np.ndarray =None, a: np.ndarray =None, maximize: bool=False) -> None:
        super().__init__(n_var=n_var, n_obj=1, xl=0.0, xu=1.0)
        self.u = np.full(n_var, 0.5) if u is None else np.asarray(u, float)
        self.a = np.full(n_var, 5.0) if a is None else np.asarray(a, float)
        self.maximize = maximize
        self.foo_name = 'Discontinuous_Integrand'
        self.f_opt = -np.exp(5) if self.maximize else 0.0

    def _evaluate(self, x: np.ndarray, out: dict, *args, **kwargs) -> float:
        """
        :param x: matriz de forma (n_puntos, n_var); cada fila es un punto.
        :param out: diccionario donde se escribe el resultado en out["F"],
                    de forma (n_puntos,) o (n_puntos, 1).
        """
        outside = (x[:, 0] > self.u[0]) | (x[:, 1] > self.u[1])
        y = np.where(outside, 0.0, np.exp(x @ self.a))
        out["F"] = -y if self.maximize else y