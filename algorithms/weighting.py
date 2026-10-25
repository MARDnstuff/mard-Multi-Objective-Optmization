from typing import Callable
from scipy.optimize import minimize
import numpy as np


class Weighting():

    def __init__(self, fx: list[Callable[[np.ndarray], float]],  bounds: list[tuple[float, float]] ) -> None:
        """
        Constructor
        """
        # Funciones objetivos
        self.fx = fx

        # Dominio
        self.func_bounds = bounds

    def run(self) -> np.ndarray:
        res = []
        x0 = np.random.rand(1,len(self.func_bounds))[0]
        # x0 = np.zeros(len(self.func_bounds))
        for w1 in np.linspace(0.0, 1.0, 2000):
            w = np.array([w1, 1.0 - w1])
            x = self.getPoint(w, x0)
            res.append([float(f(x)) for f in self.fx])
            x0 = x

        return np.array(res)
            

    def getPoint(self, obj_relevance: np.ndarray, x0: np.ndarray) -> np.ndarray:
        """
        La suma del producto de las funciones objetivo con sus pesos

        con x dentro de la area factible
        """

        def sumOfWiFx(x: np.ndarray) -> float:
            sumup = 0
            for fi, wi in zip(self.fx, obj_relevance):
                sumup += fi(x)*wi
            return sumup

        x0 = np.zeros(len(self.func_bounds))

        r = minimize(
            sumOfWiFx,
            x0,
            bounds=self.func_bounds,
            method="SLSQP",
        )

        if not r.success:
            raise RuntimeError(f"SLSQP no convergió: {r.message}")
        
        return r.x

        
        
        