from typing import Callable
import random
from scipy.optimize import minimize
import numpy as np


class EpsilonConstraint():

    def __init__(self, fl: Callable[[np.ndarray], float], fj: list[Callable[[np.ndarray], float]], bounds: list[tuple[float, float]]) -> None:
        """
        Constructor
        """
        # Función objetivo
        self.fl = fl
        # Funciones como restricciones
        self.fj = fj
        # Dominio
        self.func_bounds = bounds


    def run(self) -> np.ndarray:
        """
        Ejemplo para MOP1

        Calcula un punto débil de pareto por cada iteración sobre
        epsilon. 

        Se selecciona una función objetivo (fl) y las demás funciones (fj)
        funjen como restricciones, donde fj <= ej

        Cada función restriccion debe tener su propio epsilon

        """

        # Frente de pareto
        front = []

        # Punto fijo - requisito de scipy
        x0 = np.random.rand(1,10)[0]

        # Recorremos epsilon uniformement desde 0 hasta 6 (31 puntos)
        for eps in np.linspace(0, 10, 100):     # rango elegido a ojo

            r = minimize(self.fl, x0, bounds=self.func_bounds, method="SLSQP", constraints=[
                # fj <= ej  ->. fj - ej <= 0
                {"type": "ineq", "fun": lambda x, e=eps: e - self.fj[0](x)}

            ])

            # Descarta infactibles / fallos -> que talvez el solver no convergio
            if r.success and self.fj[0](r.x) <= eps + 1e-6:   
                front.append((float(self.fl(r.x)), float(self.fj[0](r.x))))

        # quedarse sin duplicados
        return np.unique(np.round(front, 6), axis=0)
        
        