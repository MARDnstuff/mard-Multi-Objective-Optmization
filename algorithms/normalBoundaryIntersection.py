from typing import Callable
import random
from scipy.optimize import dual_annealing, OptimizeResult, minimize
import numpy as np
import logging


logger = logging.getLogger(__name__)


class NBI():
    """
    Normal-Boundary Intersection
    """

    def __init__(self, obj_functions: list[Callable[[np.ndarray], float]], test_point: np.ndarray , bounds: list[tuple[float, float]], p: int = 10) -> None:
        """
        Constructor
        """
        # Funciones objetivos
        self.obj_functions = obj_functions

        # Número de objetivos
        self.n = len(self.obj_functions)

        # Restricciones del problema
        # bounds = [ (-l1 , +l1), (-l2 , +l2), ..., (-ln , +ln) ]
        self.func_bounds = bounds

        # Puntos inicial
        self.test_point = test_point

        self.p = p


    def evaluateFunctions(self, functions: list[Callable[[np.ndarray], float]], global_mimina: np.ndarray) -> np.ndarray:
        """
        Evalua cada función con el minimo global obtenido
        """
        curr_anchor = []
        for f in functions:
            curr_anchor.append(f(global_mimina))
        return np.array(curr_anchor)
    

    def inidividualMinima(self, curr_functions: list[Callable[[np.ndarray], float]] , curr_functions_bounds: list[tuple[float, float]], lex_tol: float = 1e-12) -> None: 
        """
        Se minimiza cada función objetivo 
        para obtener las anclas y el punto ideal 
        de cada función

        Paso 1: min f_i  -> f_i* (minimo global)
        Paso 2 (desempate): si f_i tiene varios minimizadores (por ejemplo, f1 = x1 en
        ZDT4 no depende de x2..xn), entre ellos se elige el que minimiza las demas
        funciones:

            min  sum_{j != i} f_j(x) + M * max(0, f_i(x) - f_i* - lex_tol)

        Asi el ancla es un punto de Pareto y no uno cualquiera del conjunto de empates.
        """

        try:
            # vector ideal o utopian pint
            self.shadow_minima = []

            # Anclas
            self.anchors = []

            # Minimizadores x_i*
            self.anchor_x = []

            for i, fi in enumerate(curr_functions):
                # Paso 1: minimo global de la función objetivo actual
                res1: OptimizeResult = dual_annealing(
                    func=fi, # función objetivo actual
                    bounds=curr_functions_bounds, # Limites de TODAS las variables
                    seed=42, # Semilla para poder reproducir el experimento
                )
                fi_star = res1.fun
                x_star = res1.x

                # Paso 2: desempate con las demás funciones
                others = [f for j, f in enumerate(curr_functions) if j != i]
                M = 1e4

                def lexicographic(x, fi=fi, fi_star=fi_star, others=others):
                    gap = max(0.0, fi(x) - fi_star - lex_tol)
                    return sum(f(x) for f in others) + M * gap

                res2: OptimizeResult = dual_annealing(
                    func=lexicographic,
                    bounds=curr_functions_bounds,
                    x0=x_star,
                    seed=42,
                )

                # Aceptamos el desempate solo si sigue siendo (casi) minimo de f_i
                if fi(res2.x) <= fi_star + 10 * lex_tol:
                    x_star = res2.x
                else:
                    logger.info(f"Ancla {i+1}: sin desempate (minimizador unico o desempate no valido); se usa el minimo de f{i+1}")

                # Construimos el vector ideal
                self.shadow_minima.append(fi(x_star))

                # Construimos su ancla
                self.anchor_x.append(x_star)
                self.anchors.append(self.evaluateFunctions(self.obj_functions, x_star))

            # Convertimos a np.ndarray por conveniencia
            self.shadow_minima = np.array(self.shadow_minima)
            self.anchors = np.array(self.anchors)
            self.anchor_x = np.array(self.anchor_x)
        
        except Exception as e:
            logger.error(f"Error al obtener Minima: {e}", stack_info=True)
    

    def buildPayOffTable(self) -> None:
        """
        Se construye la pay-off matrix de nxn
        al restarle las anclas el vector ideal
        """
        try:
            # Restamos a las anclas, el vector ideal
            self.payoff_matrix = (self.anchors - self.shadow_minima).T

            # Verificamos la diagonal
            diag = np.diag(self.payoff_matrix) 
            tol = 1e-8

            es_cero      = np.all(np.abs(diag) < tol)      # ~ 0
            es_positivo  = np.all(diag > -tol)             # >= 0 (tolera -1e-17)
            

            if es_cero and es_positivo:
                logger.info("Diagonal ~ 0 y no negativa")
            else:
                raise ValueError(f"Falla -> cero={es_cero}, positivo={es_positivo}, diag={diag}")

        except ValueError as e:
            logger.error(f"Error en payOff matrix: {e}", stack_info=True)


    def buildDirectionRays(self) -> None:
        """
        Se construye el CHIM (parte 1) -
        Convex Hull of Individual Minima

        Aqui se obtiene la dirección hacia el vector ideal
        """

        # Se suma todas la columnas de la matrix de pagos
        phi = np.sum(self.payoff_matrix, axis=1)

        # Cambiamos el signo
        self.direction_rays = -1*phi

        logger.info(f"Direccion de los rays = {self.direction_rays}")
    

    def buildPins(self) -> np.ndarray:
        """
        Se genran los pins desde donde se lanzan los rays
        DFS, generamos todas las combinaciones posibles que la suma nos de 1
        """
        out = []

        def descend(level, prefix, remaining):
            if level == self.n - 1:   # El ultimo componente tomar lo que resta
                out.append(prefix + [remaining]) 
                return
            for m in range(remaining + 1): # m = 0, 1, ..., remaining
                descend(level + 1, prefix + [m], remaining - m)

        descend(0, [], self.p)
        logger.info(f"out = \n {out}")
        return np.array(out) / self.p


    def solveSubproblem(self, beta, z0):
        pin = self.payoff_matrix @ beta

        def on_ray(z):
            x, t = z[:-1], z[-1]
            return pin + t * self.direction_rays - (self.evaluateFunctions(self.obj_functions, x) - self.shadow_minima)

        res = minimize(
            lambda z: -z[-1], z0,
            bounds=list(self.func_bounds) + [(None, None)],
            constraints=[{"type": "eq", "fun": on_ray}],
            method="SLSQP", options={"ftol": 1e-12}
        )

        return res.x


    def run(self) -> np.ndarray:
        """
        Normal-Boundary Intersection
        """

        # Individual minima
        self.inidividualMinima(
            curr_functions=self.obj_functions,
            curr_functions_bounds=self.func_bounds,
        )

        # PayOff Matrix
        self.buildPayOffTable()

        logger.info(f"F*] Vector ideal = {self.shadow_minima}")
        logger.info(f"F[x*] Anclas = {self.anchors}")
        logger.info(f"Pay-Off matrix = \n {self.payoff_matrix}")


        self.buildDirectionRays()

        # Pins y dirección
        pins = self.buildPins()
        logger.info(f"Pins = \n {pins}")

        # Arrancamos desde el ancla del primer pin (beta = e_n): ya es solucion con t = 0.
        # test_point se conserva en el constructor por compatibilidad.
        z = np.append(self.anchor_x[-1], 0.0)
        solutions = []
        for beta in pins:
            z = self.solveSubproblem(beta, z)
            solutions.append(z)
        solutions = np.array(solutions)

        # puntos NBI = F(x*)
        X = solutions[:, :-1]
        F_points = np.array([self.evaluateFunctions(self.obj_functions, x) for x in X])

        return pins, X, F_points