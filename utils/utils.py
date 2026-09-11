# Generación de puntos aleatorios
import random
import numpy as np
import pandas as pd

def getRandomValuesWithRange(n: int , a: float, b: float) -> np.array:
    """
    Genera n valores aletorios dado un rango [a, b] (inclusivo)

    :param n: número de valores aletorios
    :param a: rango inferior
    :param b: rando superior
    :return: lista con n valores aletorios
    """
    res = set()
    i = 0
    while i < n:
        curr_random_pt = random.uniform(a, b)
        if curr_random_pt not in res:
            res.add(curr_random_pt)
            i += 1
    return np.array(list(res))

def getTestValues(n: int, x: int, a: float, b: float) -> np.array:
    """
    Genera n arreglos con x variables, cumpliendo con el rango [a, b]

    :param n: Numeros de pruebas
    :param x: Numero de variables
    :param a: rango inferior
    :param b: rango superior
    """
    test_values = []
    for _ in range(n):
        curr_val = getRandomValuesWithRange(x, a, b) 
        test_values.append(curr_val)
    return np.array(test_values)

def mergeLists(l1: list, l2: list) -> np.array[tuple]:
    """
    Combina dos listas de listas que tienen el mismo tamaño y sus elementos igual
    """
    if len(l1) != len(l2):
        raise ValueError("Las listas deben ser del mismo tamaño")
    
    res = []
    for l1_a, l2_b  in zip(l1, l2):
        if len(l1_a) != len(l2_b):
            raise ValueError("Las elementos deben ser del mismo tamaño")
        for a, b in zip(l1_a, l2_b):
            curr_point = (a, b)
            res.append(curr_point)
    return np.array(res)


def crear_tabla_comparativa(resultados_por_metodo: dict, nombre_funcion: str) -> pd.DataFrame:
    """
    Crea una tabla comparativa de resultados de métodos de optimización.
    
    :param resultados_por_metodo: dict con nombre del método como key y dict de resultados como value Ej: {"MDG_Wolfe": resultados_rastrigin_MDG_Wolfe, ...}
    :param nombre_funcion: nombre de la función para el título
    :return: DataFrame con la tabla comparativa
    """
    filas = []

    for metodo, resultados in resultados_por_metodo.items():
        for i, datos in resultados.items():
            filas.append({
                "Punto":    i + 1,
                "Método":   metodo,
                "x_optimo": np.round(datos["x_optimo"], 4),
                "llego_optimo": datos["llego_optimo"],
                "dist": datos["dist"],
                "n_iter":   datos["n_iter"],
                "f_invok":  datos["f_invok"],
                "Df_invok": datos["Df_invok"],
                "H_invok":  datos["H_invok"],
            })

    df = pd.DataFrame(filas).sort_values(["Punto", "Método"]).reset_index(drop=True)

    print(f"\n📊 Tabla comparativa — {nombre_funcion}")
    return df

def resumen_tabla(df: pd.DataFrame) -> pd.DataFrame:
    """
    Genera un resumen estadístico por método a partir de la tabla comparativa.

    Muestra para cada método:
    - Cuántas veces llegó al óptimo.
    - Mejor aproximación (menor distancia al óptimo entre los que llegaron).
    - Promedio de invocaciones de f, Df y H.

    :param df: DataFrame generado por crear_tabla_comparativa.
    :return:   DataFrame con el resumen por método.
    """
    filas = []

    for metodo, grupo in df.groupby("Método"):

        # Cuántas veces llegó al óptimo
        n_llego = grupo["llego_optimo"].sum()

        # Mejor aproximación: menor distancia entre los que llegaron al óptimo
        llego = grupo[grupo["llego_optimo"] == True]
        if not llego.empty:
            mejor_idx  = llego["dist"].idxmin()
            mejor_dist = llego.loc[mejor_idx, "dist"]
            mejor_x    = llego.loc[mejor_idx, "x_optimo"]
        else:
            mejor_dist = None
            mejor_x    = None

        filas.append({
            "Método"          : metodo,
            "Llegó al óptimo" : f"{n_llego}/{len(grupo)}",
            "Mejor dist"      : round(mejor_dist, 6) if mejor_dist is not None else "—",
            "Mejor x_optimo"  : mejor_x,
            "Avg f_invok"     : round(grupo["f_invok"].mean(),  2),
            "Avg Df_invok"    : round(grupo["Df_invok"].mean(), 2),
            "Avg H_invok"     : round(grupo["H_invok"].mean(),  2),
        })

    resumen = pd.DataFrame(filas).reset_index(drop=True)

    print("📋 Resumen por método")
    return resumen