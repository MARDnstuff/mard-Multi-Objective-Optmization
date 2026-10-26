from config.logging import setUpLogging
from functions.mop1 import MOP1
from functions.mop4 import MOP4
from functions.mop2 import MOP2
from functions.mop3 import MOP3
from functions.mop5 import MOP5
from functions.mop6 import MOP6
from functions.mop7 import MOP7
from functions.zdt1 import ZDT1
from functions.zdt4 import ZDT4
from utils.graphs import plot_points, plot_nbi
import algorithms.naiveSlow as alg
import algorithms.continuouslyUpdated as alg2
import algorithms.kungEfficientMethod as alg3
from algorithms.epsilonConstraint import EpsilonConstraint
from algorithms.weighting import Weighting
from algorithms.normalBoundaryIntersection import NBI
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

def e_constraint_zdt4(path: str, show: bool, res: dict) -> None:
    zdt4 = ZDT4()
    zdt4_bounds = [zdt4.domain] + [zdt4.x_domain] * 9
    epsilonConst2= EpsilonConstraint(fl=zdt4.f2, fj=[zdt4.f1], bounds=zdt4_bounds, n=zdt4.n)
    inicio = time.perf_counter()
    F_points = epsilonConst2.run()
    fin = time.perf_counter()
    print(F_points)
    plot_points(F_points, names=["f2", "f1"], title="ZDT4: Frente de Pareto", save_path=path, show=show)
    res["ZDT4"].append({
        "Algoritmo": "EpsilonConstraint", 
        "Tiempo": fin - inicio, 
        "Invocaciones f1": zdt4.f1_invok,
        "Invocaciones f2": zdt4.f2_invok,
        "Cardinalidad Frente": len(F_points)
    })

def e_constraint_MOP4(path: str, show: bool, res: dict) -> None:
    mop = MOP4()
    mop_bounds = [mop.domain] * mop.n
    epsilonConst= EpsilonConstraint(fl=mop.f2, fj=[mop.f1], bounds=mop_bounds, n=mop.n)
    inicio = time.perf_counter()
    F_points = epsilonConst.run()
    fin = time.perf_counter()
    print(F_points)
    plot_points(F_points, names=["f2", "f1"], title="MOP4: Frente de Pareto", save_path=path, show=show)
    res["MOP4"].append({
        "Algoritmo": "EpsilonConstraint", 
        "Tiempo": fin - inicio, 
        "Invocaciones f1": mop.f1_invok,
        "Invocaciones f2": mop.f2_invok,
        "Cardinalidad Frente": len(F_points)
    })

def e_constraint_MOP1(path: str, show: bool, res: dict) -> None:
    mop = MOP1()
    mop_bounds = [mop.domain] * mop.n
    epsilonConst= EpsilonConstraint(fl=mop.f2, fj=[mop.f1], bounds=mop_bounds, n=mop.n)
    inicio = time.perf_counter()
    F_points = epsilonConst.run()
    fin = time.perf_counter()
    print(F_points)
    plot_points(F_points, names=["f2", "f1"], title="MOP1: Frente de Pareto", save_path=path, show=show)
    
    res["MOP1"].append({
        "Algoritmo": "EpsilonConstraint", 
        "Tiempo": fin - inicio, 
        "Invocaciones f1": mop.f1_invok,
        "Invocaciones f2": mop.f2_invok,
        "Cardinalidad Frente": len(F_points)
    })

def weighting_zdt4() -> None:
    zdt4 = ZDT4()
    zdt4_bounds = [zdt4.x_domain] + [zdt4.domain] * 9
    weight = Weighting(np.array([zdt4.f1, zdt4.f2]), zdt4_bounds)
    inicio = time.perf_counter()
    pts = weight.run()
    fin = time.perf_counter()
    print(len(pts))
    plot_points(pts, names=["f1", "f2"], title="Frente de Pareto")

def NBI_example_MOP1(path: str, show: bool, res: dict) -> None:
    mop = MOP1()

    nbi = NBI(
        obj_functions=[mop.f1, mop.f2],
        test_point=np.array([5.0]),      # un punto, N = 1 variable
        bounds=[mop.domain] * mop.n,   # un (min, max) por variable
    )

    inicio = time.perf_counter()
    pins, X, F_points = nbi.run()
    fin = time.perf_counter()

    logger.info(f"Pins = \n{pins}")
    logger.info(f"F_points = \n{F_points}")
    logger.info(f"X = \n {X}")

    plot_nbi(
        F_points,
        anchors=nbi.anchors,
        shadow_minima=nbi.shadow_minima,
        direction_rays=nbi.direction_rays,
        payoff_matrix=nbi.payoff_matrix,
        betas=pins,
        front=mop.get_samples_range(np.linspace(0, 2, 200).reshape(-1, 1)),
        title="NBI - MOP1",
        save_path=path + "_with_rays",
        show=show
    )

    plot_points(F_points, names=["f1","f2"], title="NBI - Frente de Pareto (MOP1)", save_path=path, show=show)

    res["MOP1"].append({
        "Algoritmo": "NBI", 
        "Tiempo": fin - inicio, 
        "Invocaciones f1": mop.f1_invok,
        "Invocaciones f2": mop.f2_invok,
        "Cardinalidad Frente": len(F_points)
    })

def NBI_example_MOP4(path: str, show: bool, res: dict) -> None:
    mop = MOP4()

    nbi = NBI(
        obj_functions=[mop.f1, mop.f2],
        test_point=np.array([5.0, 5.0 , 5.0]),      # un punto, N = 1 variable
        bounds=[mop.domain] * mop.n,   # un (min, max) por variable
        p=30
    )

    inicio = time.perf_counter()
    pins, X, F_points = nbi.run()
    fin = time.perf_counter()

    logger.info(f"Pins = \n{pins}")
    logger.info(f"F_points = \n{F_points}")
    logger.info(f"X = \n {X}")

    plot_nbi(
        F_points,
        anchors=nbi.anchors,
        shadow_minima=nbi.shadow_minima,
        direction_rays=nbi.direction_rays,
        payoff_matrix=nbi.payoff_matrix,
        betas=pins,
        front=mop.get_samples_range(np.linspace(0, 2, 200).reshape(-1, 1)),
        title="NBI - MOP4",
        save_path=path + "_with_rays",
        show=show
    )

    plot_points(F_points, names=["f1","f2"], title="NBI - Frente de Pareto (MOP4)", save_path=path, show=show)

    res["MOP4"].append({
        "Algoritmo": "NBI", 
        "Tiempo": fin - inicio, 
        "Invocaciones f1": mop.f1_invok,
        "Invocaciones f2": mop.f2_invok,
        "Cardinalidad Frente": len(F_points)
    })

def NBI_example_ZDT4(path: str, show: bool, res: dict) -> None:
    mop = ZDT4()

    zdt4_bounds = [mop.x_domain] + [mop.domain] * 9

    nbi = NBI(
        obj_functions=[mop.f1, mop.f2],
        test_point=np.array([5.0]*10),      # un punto, N = 1 variable
        bounds=zdt4_bounds,   # un (min, max) por variable
        p=30
    )

    inicio = time.perf_counter()
    pins, X, F_points = nbi.run()
    fin = time.perf_counter()


    logger.info(f"Pins = \n{pins}")
    logger.info(f"F_points = \n{F_points}")
    logger.info(f"X = \n {X}")


    dom = mop.get_samples_domain(200)
    img = mop.get_samples_range(dom)

    plot_nbi(
        F_points,
        anchors=nbi.anchors,
        shadow_minima=nbi.shadow_minima,
        direction_rays=nbi.direction_rays,
        payoff_matrix=nbi.payoff_matrix,
        betas=pins,
        front=img,
        title="NBI - ZDT4",
        save_path=path + "_with_rays",
        show=show
    )

    plot_points(F_points, names=["f1","f2"], title="NBI - Frente de Pareto (ZDT4)", save_path=path, show=show)

    res["ZDT4"].append({
        "Algoritmo": "NBI", 
        "Tiempo": fin - inicio, 
        "Invocaciones f1": mop.f1_invok,
        "Invocaciones f2": mop.f2_invok,
        "Cardinalidad Frente": len(F_points)
    })


if __name__ == "__main__":
    res = {
        "MOP1": [],
        "MOP4": [],
        "ZDT4": [],
    }
    show = False
    prefix = "IMG/Tarea04/"
    problem = ["MOP1", "MOP4", "ZDT4"]
    type = ["NBI", "EpsilonConstraint"]
    NBI_example_MOP1(path=f"{prefix}{type[0]}_{problem[0]}", show=show, res=res)
    NBI_example_MOP4(path=f"{prefix}{type[0]}_{problem[1]}", show=show, res=res)
    NBI_example_ZDT4(path=f"{prefix}{type[0]}_{problem[2]}", show=show, res=res)
    e_constraint_MOP1(path=f"{prefix}{type[1]}_{problem[0]}", show=show, res=res)
    e_constraint_MOP4(path=f"{prefix}{type[1]}_{problem[1]}", show=show, res=res)
    e_constraint_zdt4(path=f"{prefix}{type[1]}_{problem[2]}", show=show, res=res)

    with open(f"results/NBI_and_EpsilonConstraint_resultados.json", "w") as f:
        json.dump(res, f, indent=4)
    
    
    # weighting_zdt4()
    # mop1 = MOP1()

    # epsilonConstring = EpsilonConstraint(fl=mop1.f1, fj=[mop1.f2], bounds=mop1.domain)
    # pts = epsilonConstring.run()

    # print(pts)
    # plot_points(pts, names=["f1", "f2"], title="Frente de Pareto")


    # zdt1 = ZDT1()

    # epsilonConst2= EpsilonConstraint(fl=zdt1.f1, fj=[zdt1.f2], bounds=zdt1.domain)
    # pts = epsilonConst2.run()

    # print(pts)
    # plot_points(pts, names=["f1", "f2"], title="Frente de Pareto")
    
    # frente_pareto = image[idx]
    # conjunto_pareto = domain[idx]

    # print(frente_pareto)
    # print(conjunto_pareto)

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
    
