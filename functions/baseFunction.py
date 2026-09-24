from abc import ABC
import inspect
import logging
import numpy as np
from scipy.optimize import minimize

logger = logging.getLogger(__name__)


class Function(ABC):
    """
    Clase abstracta base para problemas multiobjetivo.

    Las subclases deben definir:
        - self.n: número de variables de decisión.
        - self.domain: tupla (min, max) común a todas las variables.
        - Métodos f1, f2, ..., fk: objetivos a minimizar.
    """

    def __init__(self, domain: tuple[float, float]) -> None:
        if not hasattr(self, "n"):
            raise AttributeError(
                "La subclase debe definir self.n antes de llamar a super().__init__()."
            )
        self.domain = domain
        self.z_ideal: np.ndarray | None = None
        self.z_nadir: np.ndarray | None = None
        self.ideal_vector()

    def _get_objective_methods(self) -> list:
        objectives = []
        i = 1
        while True:
            method = getattr(self, f"f{i}", None)
            if method is None or not callable(method):
                break
            objectives.append(method)
            i += 1

        if not objectives:
            raise ValueError("No se encontraron métodos f1, f2, ... en esta instancia.")
        return objectives

    def ideal_vector(self, method: str = "L-BFGS-B", n_restarts: int = 1) -> None:
        """
        Calcula z_ideal y z_nadir usando la tabla de pagos.

        :param method: método de scipy.optimize.minimize (debe soportar bounds).
        :param n_restarts: número de reinicios por objetivo (útil en multimodales).
        """
        objectives = self._get_objective_methods()
        M = len(objectives)
        bounds = [self.domain] * self.n

        payoff = np.zeros((M, M))

        for k, f_k in enumerate(objectives):
            best_val = np.inf
            best_x = None

            for _ in range(n_restarts):
                x0 = np.random.uniform(self.domain[0], self.domain[1], self.n)
                res = minimize(f_k, x0, bounds=bounds, method=method)

                if not res.success:
                    logger.warning(f"f{k+1} no convergió: {res.message}")

                if res.fun < best_val:
                    best_val = res.fun
                    best_x = res.x

            for i, f_i in enumerate(objectives):
                payoff[k, i] = f_i(best_x)

        if not np.isfinite(payoff).all():
            logger.warning("La tabla de pagos contiene valores no finitos.")

        self.z_ideal = np.diag(payoff)
        self.z_nadir = payoff.max(axis=0)
        logger.debug(f"z_ideal={self.z_ideal}, z_nadir={self.z_nadir}")