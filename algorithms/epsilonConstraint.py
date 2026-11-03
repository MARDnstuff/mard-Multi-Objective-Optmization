from typing import Callable, Optional

import numpy as np
from scipy.optimize import minimize


class EpsilonConstraint():

    def __init__(self, fl: Callable[[np.ndarray], float], fj: list[Callable[[np.ndarray], float]],
                bounds: list[tuple[float, float]], eps_range: tuple[float, float], n: int,
                n_eps_points: int = 100, n_starts: int = 2, x0: Optional[float] = None, seed: Optional[int] = None) -> None:
        """
        Constructor

        :param fl: función objetivo seleccionada
        :param fj: resto de funciones como restricciones (por ahora solo se usa fj[0])
        :param bounds: rango de valores posibles por cada variable [(-5, 5), (2, 10), ...]
        :param eps_range: rango de valores epsilon (inicio, fin)
        :param n: número de variables
        :param n_eps_points: número de puntos distribuidos en el rango de epsilon
        :param n_starts: puntos iniciales aleatorios adicionales por cada epsilon
                        (además del arranque en caliente con la solución anterior)
        :param seed: semilla para reproducibilidad
        :param x0: punto de inicio
        """
        self.fl = fl
        self.fj = fj
        self.func_bounds = bounds
        self.eps_range = eps_range
        self.n = n
        self.n_eps_points = n_eps_points
        self.n_starts = n_starts
        self.rng = np.random.default_rng(seed)
        self.x0 = x0

    @staticmethod
    def non_dominated(F: np.ndarray) -> np.ndarray:
        """
        Filtra los puntos dominados (2 objetivos, minimización).
        Ordena por f1 y conserva solo los puntos que mejoran el mejor f2 visto.
        """
        F = np.unique(np.round(F, 6), axis=0)
        order = np.argsort(F[:, 0])
        best, keep = np.inf, []
        for i in order:
            if F[i, 1] < best:
                keep.append(i)
                best = F[i, 1]
        return F[keep]

    def _solve(self, eps: float, starts: list[np.ndarray]) -> Optional[tuple]:
        """
        Resuelve  min fl(x)  s.a.  fj(x) <= eps  desde varios puntos iniciales
        y regresa la mejor solución factible: (x, fl(x), fj(x)), o None.
        """
        best = None
        for x0 in starts:
            r = minimize(self.fl, x0, bounds=self.func_bounds, method="SLSQP", constraints=[
                # SciPy "ineq" significa fun(x) >= 0  ->  eps - fj(x) >= 0  ->  fj(x) <= eps
                {"type": "ineq", "fun": lambda x, e=eps: e - self.fj[0](x)}
            ])
            if not r.success:
                continue
            fj_val = self.fj[0](r.x)                # una sola evaluación
            if fj_val <= eps + 1e-6 and (best is None or r.fun < best[1]):
                best = (r.x, float(r.fun), float(fj_val))
        return best

    def run(self) -> np.ndarray:
        """
        Calcula un punto (débil) de Pareto por cada epsilon y regresa
        solo los puntos no dominados.

        Se selecciona una función objetivo (fl) y las demás funciones (fj)
        fungen como restricciones, donde fj <= eps.
        """
        lb, ub = np.array(self.func_bounds, dtype=float).T
        front = []
        x_prev = self.x0
        
        for eps in np.linspace(self.eps_range[0], self.eps_range[1], self.n_eps_points):
            # Arranque en caliente con la solución anterior + arranques aleatorios
            k = self.n_starts if x_prev is not None else max(self.n_starts, 1)
            starts = list(self.rng.uniform(lb, ub, size=(k, self.n)))
            if x_prev is not None:
                starts.insert(0, x_prev)

            best = self._solve(eps, starts)
            if best is not None:            # descarta infactibles / fallos
                x_prev = best[0]
                front.append(best[1:])

        if not front:
            return np.empty((0, 2))
        return self.non_dominated(np.array(front))