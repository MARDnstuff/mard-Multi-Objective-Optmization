from config.logging import setUpLogging
from functions.mop1 import MOP1
from functions.mop4 import MOP4
from functions.mop2 import MOP2
from functions.mop3 import MOP3
from functions.mop5 import MOP5
from functions.mop6 import MOP6
from functions.mop7 import MOP7
from functions.zdt1 import ZDT1
from utils.graphs import plot_points
import algorithms.naiveSlow as alg
import algorithms.continuouslyUpdated as alg2
import algorithms.kungEfficientMethod as alg3
import time
import numpy as np
import json
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

def kung() -> None:
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

if __name__ == "__main__":

    mop1 = MOP1()
    domain = mop1.get_samples_domain(10)
    image = mop1.get_samples_range(domain)

    naiveSlow = alg.NaiveSlow(image)
    idx = naiveSlow.run()
    
    frente_pareto = image[idx]
    conjunto_pareto = domain[idx]

    print(frente_pareto)
    print(conjunto_pareto)

    # problem_name = "MOP5"
    # problem = MOP5()
    # # myvar = problem.labels_var()
    # myvar = ["x", "y"]
    # myfoo = ["f1", "f2", "f3"]

    # res = {
    #     problem_name: [],
    # }

    # logger.info(f"===== {problem_name} =====")

    # domain = problem.get_samples_domain(1000)
    # image = problem.get_samples_range(domain)

    # naiveSlow1 = alg.NaiveSlow(image)
    # inicio = time.perf_counter()
    # idx = naiveSlow1.run()
    # fin = time.perf_counter()

    # frente_pareto = image[idx]
    # conjunto_pareto = domain[idx]

    # plot_points(domain, names=myvar, title="Dominio", save_path=f"IMG/Tarea02/{problem_name}_Dominio.png", show=False)
    # plot_points(image, names=myfoo, title="Imagen", save_path=f"IMG/Tarea02/{problem_name}_Imagen.png", show=False)
    # plot_points(frente_pareto, names=myfoo, title="Frente de Pareto", save_path=f"IMG/Tarea02/{problem_name}_FrenteDePareto_NaiveAndSlow.png", show=False, z_ideal=problem.z_ideal, z_nadir=problem.z_nadir)
    # plot_points(conjunto_pareto, names=myvar, title="Conjunto de Pareto", save_path=f"IMG/Tarea02/{problem_name}_ConjuntoDePareto_NaiveAndSlow.png", show=False)

    # res[problem_name].append({
    #     "Algoritmo": "Naive and Slow", 
    #     "Tiempo": fin - inicio, 
    #     "No. Comparaciones": naiveSlow1.f_count_comparison,
    #     "Cardinalidad Frente": len(frente_pareto)
    # }) 

    # contUpdated1 = alg2.ContinuouslyUpdated(image)
    # inicio = time.perf_counter()
    # idx = contUpdated1.run()
    # fin = time.perf_counter()

    # frente_pareto = image[idx]
    # conjunto_pareto = domain[idx]

    # plot_points(frente_pareto, names=myfoo, title="Frente de Pareto", save_path=f"IMG/Tarea02/{problem_name}_FrenteDePareto_ContinuouslyUpdated.png", show=False, z_ideal=problem.z_ideal, z_nadir=problem.z_nadir)
    # plot_points(conjunto_pareto, names=myvar, title="Conjunto de Pareto", save_path=f"IMG/Tarea02/{problem_name}_ConjuntoDePareto_ContinuouslyUpdated.png", show=False)


    # res[problem_name].append({
    #     "Algoritmo": "Continuously Updated", 
    #     "Tiempo": fin - inicio, 
    #     "No. Comparaciones": contUpdated1.f_count_comparison,
    #     "Cardinalidad Frente": len(frente_pareto)
    # })


    # kungAlgo = alg3.KungEfficientMethod(image)
    # inicio = time.perf_counter()
    # idx = kungAlgo.run()
    # fin = time.perf_counter()

    # frente_pareto = image[idx]
    # conjunto_pareto = domain[idx]

    # plot_points(frente_pareto, names=myfoo, title="Frente de Pareto", save_path=f"IMG/Tarea02/{problem_name}_FrenteDePareto_KungEfficientMethod.png", show=False, z_ideal=problem.z_ideal, z_nadir=problem.z_nadir)
    # plot_points(conjunto_pareto, names=myvar, title="Conjunto de Pareto", save_path=f"IMG/Tarea02/{problem_name}_ConjuntoDePareto_KungEfficientMethod.png", show=False)


    # res[problem_name].append({
    #     "Algoritmo": "Kung Efficient Method", 
    #     "Tiempo": fin - inicio, 
    #     "No. Comparaciones": kungAlgo.f_count_comparison,
    #     "Cardinalidad Frente": len(frente_pareto)
    # }) 

    # res["z_ideal"] = list(problem.z_ideal)
    # res["z_nadir"] = list(problem.z_nadir)

    # logger.info(res)

    # with open(f"results/{problem_name}_resultados.json", "w") as f:
    #     json.dump(res, f, indent=4)
    

    

