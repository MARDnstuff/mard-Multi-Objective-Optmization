from config.logging import setUpLogging
from functions.mop1 import MOP1
from functions.mop4 import MOP4
from functions.mop2 import MOP2
from functions.mop3 import MOP3
from functions.mop5 import MOP5
from functions.mop6 import MOP6
from functions.mop7 import MOP7
from utils.graphs import plot_points
import algorithms.naiveSlow as alg
import algorithms.continuouslyUpdated as alg2
import algorithms.kungEfficientMethod as alg3
import time
import numpy as np
import logging


# Logging
setUpLogging()
logger = logging.getLogger(__name__)

def naive_slow_example() -> None:
    """
    Ejemplo de optimización por fuerza fruta
    """
    mop1 = MOP1()
    domain = mop1.get_samples_domain(1000)
    image = mop1.get_samples_range(domain)

    plot_points(domain, names=["x"], title="Dominio")
    plot_points(image, names=["f1", "f2"], title="Imagen")

    naiveSlow = alg.NaiveSlow(image)
    inicio = time.perf_counter()
    idx = naiveSlow.run()
    fin = time.perf_counter()
    
    frente_pareto = image[idx]
    conjunto_pareto = domain[idx]

    plot_points(frente_pareto, names=["f1", "f2"], title="Frente de Pareto")
    plot_points(conjunto_pareto, names=["x"], title="Conjunto de Pareto")


    # mop2 = MOP2()
    # domain = mop2.get_samples_domain(1000)
    # image = mop2.get_samples_range(domain)

    # plot_points(domain, names=["x1", "x2", "x3"], title="Dominio")
    # plot_points(image, names=["f1", "f2"], title="Imagen")

    # naiveSlow = alg.NaiveSlow(image)
    # inicio = time.perf_counter()
    # idx = naiveSlow.run()
    # fin = time.perf_counter()
    
    # frente_pareto = image[idx]
    # conjunto_pareto = domain[idx]

    # plot_points(conjunto_pareto, names=["x1", "x2", "x3"], title="Conjunto de Pareto")
    # plot_points(frente_pareto, names=["f1", "f2"], title="Frente de Pareto")

    # mop3 = MOP3()
    # domain = mop3.get_samples_domain(1000)
    # image = mop3.get_samples_range(domain)

    # plot_points(domain, names=["x", "y"], title="Dominio")
    # plot_points(image, names=["f1", "f2"], title="Imagen")

    # naiveSlow = alg.NaiveSlow(image)
    # inicio = time.perf_counter()
    # idx = naiveSlow.run()
    # fin = time.perf_counter()
    
    # frente_pareto = image[idx]
    # conjunto_pareto = domain[idx]

    # plot_points(conjunto_pareto, names=["x", "y"], title="Conjunto de Pareto")
    # plot_points(frente_pareto, names=["f1", "f2"], title="Frente de Pareto")


    # mop4 = MOP4()
    # domain = mop4.get_samples_domain(1000)
    # image = mop4.get_samples_range(domain)

    # plot_points(domain, names=["x1", "x2", "x3"], title="Dominio")
    # plot_points(image, names=["f1", "f2"], title="Imagen")

    # naiveSlow = alg.NaiveSlow(image)
    # inicio = time.perf_counter()
    # idx = naiveSlow.run()
    # fin = time.perf_counter()
    
    # frente_pareto = image[idx]
    # conjunto_pareto = domain[idx]

    # plot_points(frente_pareto, names=["f1", "f2"], title="Frente de Pareto")
    # plot_points(conjunto_pareto, names=["x1", "x2", "x3"], title="Conjunto de Pareto")

    # mop5 = MOP5()
    # domain = mop5.get_samples_domain(1000)
    # image = mop5.get_samples_range(domain)

    # plot_points(domain, names=["x", "y"], title="Dominio")
    # plot_points(image, names=["f1", "f2", "f3"], title="Imagen")

    # naiveSlow = alg.NaiveSlow(image)
    # inicio = time.perf_counter()
    # idx = naiveSlow.run()
    # fin = time.perf_counter()
    
    # frente_pareto = image[idx]
    # conjunto_pareto = domain[idx]

    # plot_points(frente_pareto, names=["f1", "f2", "f3"], title="Frente de Pareto")
    # plot_points(conjunto_pareto, names=["x", "y"], title="Conjunto de Pareto")


    # mop6 = MOP6()
    # domain = mop6.get_samples_domain(1000)
    # image = mop6.get_samples_range(domain)

    # plot_points(domain, names=["x", "y"], title="Dominio")
    # plot_points(image, names=["f1", "f2"], title="Imagen")

    # naiveSlow = alg.NaiveSlow(image)
    # inicio = time.perf_counter()
    # idx = naiveSlow.run()
    # fin = time.perf_counter()
    
    # frente_pareto = image[idx]
    # conjunto_pareto = domain[idx]

    # plot_points(frente_pareto, names=["f1", "f2"], title="Frente de Pareto")
    # plot_points(conjunto_pareto, names=["x", "y"], title="Conjunto de Pareto")

    # mop7 = MOP7()
    # domain = mop7.get_samples_domain(1000)
    # image = mop7.get_samples_range(domain)

    # plot_points(domain, names=["x", "y"], title="Dominio")
    # plot_points(image, names=["f1", "f2", "f3"], title="Imagen")

    # naiveSlow = alg.NaiveSlow(image)
    # inicio = time.perf_counter()
    # idx = naiveSlow.run()
    # fin = time.perf_counter()

    # frente_pareto = image[idx]
    # conjunto_pareto = domain[idx]

    # plot_points(frente_pareto, names=["f1", "f2", "f3"], title="Frente de Pareto")
    # plot_points(conjunto_pareto, names=["x", "y"], title="Conjunto de Pareto")

    logger.info(f" Número de comparaciones: {naiveSlow.f_count_comparison} , Tiempo: {fin - inicio:.4f} segundos")


def continuously_updated() -> None:
    mop1 = MOP1()
    domain = mop1.get_samples_domain(1000)
    image = mop1.get_samples_range(domain)

    plot_points(domain, names=["x"], title="Dominio")
    plot_points(image, names=["f1", "f2"], title="Imagen")

    naiveSlow = alg2.ContinuouslyUpdated(image)
    inicio = time.perf_counter()
    idx = naiveSlow.run()
    fin = time.perf_counter()
    
    frente_pareto = image[idx]
    conjunto_pareto = domain[idx]

    plot_points(frente_pareto, names=["f1", "f2"], title="Frente de Pareto")
    plot_points(conjunto_pareto, names=["x"], title="Conjunto de Pareto")


    logger.info(f" Número de comparaciones: {naiveSlow.f_count_comparison} , Tiempo: {fin - inicio:.4f} segundos")

    naiveSlow = alg.NaiveSlow(image)
    inicio = time.perf_counter()
    idx = naiveSlow.run()
    fin = time.perf_counter()
    
    frente_pareto = image[idx]
    conjunto_pareto = domain[idx]

    plot_points(frente_pareto, names=["f1", "f2"], title="Frente de Pareto 2")
    plot_points(conjunto_pareto, names=["x"], title="Conjunto de Pareto 2")

    logger.info(f" Número de comparaciones: {naiveSlow.f_count_comparison} , Tiempo: {fin - inicio:.4f} segundos")
    

if __name__ == "__main__":
    logger.info("===== WELCOME MARD =====")
    # naive_slow_example()
    # continuously_updated()

    mop2 = MOP2()
    domain = mop2.get_samples_domain(1000)
    image = mop2.get_samples_range(domain)

    kung = alg3.KungEfficientMethod(image)
    idx = kung.run()

    plot_points(domain, names=["x"], title="Dominio")
    plot_points(image, names=["f1", "f2"], title="Imagen")


    frente_pareto = image[idx]
    conjunto_pareto = domain[idx]

    plot_points(frente_pareto, names=["f1", "f2"], title="Frente de Pareto")
    plot_points(conjunto_pareto, names=["x"], title="Conjunto de Pareto")