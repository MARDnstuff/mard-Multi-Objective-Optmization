from typing import Callable, Optional
from scipy.optimize import minimize
import numpy as np
import logging


logger = logging.getLogger(__name__)

class Weighting():

    def __init__(self, fx: list[Callable[[np.ndarray], float]],  bounds: list[tuple[float, float]], n_weights: int = 100 ) -> None:
        """
        Constructor

        :param fx: lista con las dos funciones objetivo [f1, f2]
        :param bounds: rango de valores posibles por cada variable [(-5, 5), (2, 10), ...]
        :param n_weights: número de pesos distribuidos uniformemente en [0, 1]
        """
        # Por ahora solo se soportan dos objetivos: w = [w1, 1 - w1]
        if len(fx) != 2:
            raise ValueError(f"Weighting solo soporta 2 objetivos, se recibieron {len(fx)}")

        # Funciones objetivos
        self.fx = fx

        # Dominio
        self.func_bounds = bounds
        self.n_weights = n_weights

    def run(self) -> np.ndarray:
        res = []
        x0 = np.random.rand(1,len(self.func_bounds))[0]

        for w1 in np.linspace(0.0, 1.0, self.n_weights):
            w = np.array([w1, 1.0 - w1])
            x = self.getPoint(w, x0)

            # Si el solver no convergió, se descarta este peso y se conserva
            # el x0 anterior como arranque para el siguiente
            if x is None:
                continue

            res.append([float(f(x)) for f in self.fx])
            x0 = x

        if not res:
            logger.warning("Ningún peso convergió; se regresa un arreglo vacío")
            return np.empty((0, len(self.fx)))

        return np.array(res)


    def getPoint(self, obj_relevance: np.ndarray, x0: np.ndarray) -> Optional[np.ndarray]:
        """
        La suma del producto de las funciones objetivo con sus pesos

        con x dentro de la area factible

        :return: x óptimo para los pesos dados, o None si SLSQP no convergió
        """

        def sumOfWiFx(x: np.ndarray) -> float:
            sumup = 0
            for fi, wi in zip(self.fx, obj_relevance):
                sumup += fi(x)*wi
            return sumup


        r = minimize(
            sumOfWiFx,
            x0,
            bounds=self.func_bounds,
            method="SLSQP",
        )

        if not r.success:
            logger.warning(f"SLSQP no convergió con pesos {obj_relevance}: {r.message}")
            return None

        return r.x