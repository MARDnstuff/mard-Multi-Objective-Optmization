import numpy as np
from .baseFunction import Function
import logging


logger = logging.getLogger(__name__)

class MeetingPoint(Function):
    """
    Punto de encuentro entre dos ciudades

    Definitions
        F = (f1(x), f2(x)), where x = (u, v)

        f1(x) = (u - a1)^2 + (v - a2)^2
        f2(x) = (u - b1)^2 + (v - b2)^2

    Constraints
        -10 <= u, v <= 10
        a = (0, 0)
        b = (3, 4)

    Pareto set
        x = a + t(b - a),  t in [0, 1]

    Pareto front
        f2 = (L - sqrt(f1))^2,  f1 in [0, L^2],  L = ||b - a||

    """
    def __init__(self, a: tuple[float, float] = (0.0, 0.0), b: tuple[float, float] = (3.0, 4.0)) -> None:
        """
        Constructor

        :param a: ubicación de la ciudad A
        :param b: ubicación de la ciudad B
        """
        domain = (-10, 10)
        self.n = 2
        self.a = np.asarray(a, dtype=float)
        self.b = np.asarray(b, dtype=float)
        # Distancia entre las ciudades
        self.L = float(np.linalg.norm(self.b - self.a))
        self.f1_invok = 0
        self.f2_invok = 0
        super().__init__(domain)

    def _check_domain(self, x: np.ndarray) -> None:
        """
        Verifica que cada variable de decisión esté dentro del dominio
        """
        if np.any(x < self.domain[0]) or np.any(x > self.domain[1]):
            raise ValueError("x value is out of the valid range")

    def f1(self, x: np.ndarray) -> float:
        """
        Función objetivo 1 a minimizar: distancia al cuadrado desde la ciudad A

        :param x: arreglo (vector) de variables de decisión [u, v]
        :return: Valor escalar de la función evaluada en x.
        """
        self._check_domain(x)
        self.f1_invok += 1
        return float((x[0] - self.a[0])**2 + (x[1] - self.a[1])**2)

    def f2(self, x: np.ndarray) -> float:
        """
        Función objetivo 2 a minimizar: distancia al cuadrado desde la ciudad B

        :param x: arreglo (vector) de variables de decisión [u, v]
        :return: Valor escalar de la función evaluada en x.
        """
        self._check_domain(x)
        self.f2_invok += 1
        return float((x[0] - self.b[0])**2 + (x[1] - self.b[1])**2)

    def get_samples_domain(self, n_samples: int, seed: int = None) -> np.ndarray:
        """
        Genera n cantidad de muestras aleatorias restringido al dominio
        de la función

        :param n_samples: número de muestra requeridas
        :param seed: semilla para reproducibilidad
        :return: arreglo de forma (n_samples, 2)
        """
        rng = np.random.default_rng(seed)
        return rng.uniform(self.domain[0], self.domain[1], size=(n_samples, self.n))

    def get_samples_range(self, sample_domain: np.ndarray) -> np.ndarray:
        """
        Se calcula la funciones objetivos dado la muestra del dominio

        :param sample_domain: muestra del dominio
        :return: arreglo de forma (n_samples, 2) con [f1, f2]
        """
        return np.array([[self.f1(x), self.f2(x)] for x in sample_domain])

    def get_true_front(self, n_points: int = 100) -> np.ndarray:
        """
        Frente de Pareto teórico: f2 = (L - sqrt(f1))^2, f1 en [0, L^2]
        Útil como referencia para validar los métodos.

        :param n_points: número de puntos del frente
        :return: arreglo de forma (n_points, 2) con [f1, f2]
        """
        t = np.linspace(0, 1, n_points)
        return np.column_stack([(t * self.L)**2, ((1 - t) * self.L)**2])

    def get_true_set(self, n_points: int = 100) -> np.ndarray:
        """
        Conjunto de Pareto teórico: segmento entre las ciudades, x = a + t(b - a)

        :param n_points: número de puntos del conjunto
        :return: arreglo de forma (n_points, 2) con [u, v]
        """
        t = np.linspace(0, 1, n_points)[:, None]
        return self.a + t * (self.b - self.a)