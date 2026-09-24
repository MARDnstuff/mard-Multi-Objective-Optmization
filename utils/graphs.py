import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


def _draw_ideal_nadir_2d(ax, dim_x: int, dim_y: int, z_ideal, z_nadir):
    """
    Dibuja, sobre un eje 2D, los puntos z_ideal/z_nadir (usando las
    dimensiones dim_x, dim_y) y el rectángulo delimitador entre ambos.
    """
    ix, iy = z_ideal[dim_x], z_ideal[dim_y]
    nx, ny = z_nadir[dim_x], z_nadir[dim_y]

    ax.scatter([ix], [iy], marker="*", s=140, color="green", edgecolor="black", zorder=5, label="Ideal")
    ax.scatter([nx], [ny], marker="X", s=100, color="red", edgecolor="black", zorder=5, label="Nadir")

    ancho = nx - ix
    alto = ny - iy
    rect = patches.Rectangle(
        (ix, iy), ancho, alto,
        linewidth=1.2, linestyle="--", edgecolor="gray",
        facecolor="none", zorder=4,
    )
    ax.add_patch(rect)


def _draw_ideal_nadir_3d(ax, z_ideal, z_nadir):
    """
    Dibuja, sobre un eje 3D, los puntos z_ideal/z_nadir y el hipercubo
    (cuboide) delimitador, trazando sus 12 aristas.
    """
    ax.scatter(*z_ideal, marker="*", s=140, color="green", edgecolor="black", zorder=5, label="Ideal")
    ax.scatter(*z_nadir, marker="X", s=100, color="red", edgecolor="black", zorder=5, label="Nadir")

    xs = [z_ideal[0], z_nadir[0]]
    ys = [z_ideal[1], z_nadir[1]]
    zs = [z_ideal[2], z_nadir[2]]

    # Los 8 vértices del cuboide (todas las combinaciones min/max por eje)
    vertices = [(x, y, z) for x in xs for y in ys for z in zs]

    # Une dos vértices si difieren en exactamente una coordenada (arista)
    for a in vertices:
        for b in vertices:
            if sum(abs(np.array(a) - np.array(b)) > 1e-12) == 1:
                ax.plot(*zip(a, b), color="gray", linestyle="--", linewidth=1)


def _draw_ideal_nadir_1d(ax, z_ideal, z_nadir):
    """Dibuja z_ideal/z_nadir como líneas verticales sobre un eje 1D."""
    ax.axvline(z_ideal[0], color="green", linestyle="--", linewidth=1.5, label="Ideal")
    ax.axvline(z_nadir[0], color="red", linestyle="--", linewidth=1.5, label="Nadir")


def plot_points(
    data: np.ndarray,
    names=None,
    title="",
    save_path=None,
    dpi=150,
    show=True,
    z_ideal: np.ndarray = None,
    z_nadir: np.ndarray = None,
):
    """
    Grafica un conjunto de puntos de n dimensiones y opcionalmente guarda
    la figura. Si se proporcionan z_ideal y z_nadir (arreglos de tamaño
    n_dim), también dibuja esos puntos y el rectángulo/hipercubo que
    delimitan (las líneas del vector ideal al punto de Nadir).

    :param data: array-like de forma (n_puntos, n_dim)
    :param names: lista de nombres para cada dimensión (opcional)
    :param title: título de la gráfica
    :param save_path: ruta donde guardar la figura (opcional)
    :param dpi: resolución de la imagen guardada
    :param show: si True, muestra la gráfica con plt.show()
    :param z_ideal: arreglo (n_dim,) con el vector ideal (opcional)
    :param z_nadir: arreglo (n_dim,) con el punto de Nadir (opcional)
    """
    data = np.asarray(data)
    n_puntos, n_dim = data.shape
    logger.debug(f"Data shape dimensions: {n_dim}")

    if names is None or len(names) < n_dim:
        names = [f"x{i+1}" for i in range(n_dim)]

    tiene_referencia = z_ideal is not None and z_nadir is not None
    if tiene_referencia:
        z_ideal = np.asarray(z_ideal)
        z_nadir = np.asarray(z_nadir)
        if len(z_ideal) != n_dim or len(z_nadir) != n_dim:
            raise ValueError(
                f"z_ideal/z_nadir deben tener tamaño {n_dim}, "
                f"igual al número de dimensiones de data."
            )

    if n_dim == 1:
        fig, ax = plt.subplots(figsize=(6, 2))
        ax.scatter(data[:, 0], np.zeros(n_puntos), s=8, alpha=0.5)
        ax.set_yticks([])
        ax.set_xlabel(names[0])
        if tiene_referencia:
            _draw_ideal_nadir_1d(ax, z_ideal, z_nadir)
            ax.legend(fontsize=8)

    elif n_dim == 2:
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.scatter(data[:, 0], data[:, 1], s=10, alpha=0.6, zorder=3)
        ax.set_xlabel(names[0])
        ax.set_ylabel(names[1])
        if tiene_referencia:
            _draw_ideal_nadir_2d(ax, 0, 1, z_ideal, z_nadir)
            ax.legend(fontsize=8)

    elif n_dim == 3:
        fig = plt.figure(figsize=(6, 5))
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(data[:, 0], data[:, 1], data[:, 2], s=10, alpha=0.6)
        ax.set_xlabel(names[0])
        ax.set_ylabel(names[1])
        ax.set_zlabel(names[2])
        if tiene_referencia:
            _draw_ideal_nadir_3d(ax, z_ideal, z_nadir)
            ax.legend(fontsize=8)

    else:
        # más de 3 dimensiones -> matriz de dispersión por pares
        fig, axes = plt.subplots(n_dim, n_dim, figsize=(2.2 * n_dim, 2.2 * n_dim))
        for i in range(n_dim):
            for j in range(n_dim):
                ax = axes[i, j]
                if i == j:
                    ax.hist(data[:, i], bins=25, alpha=0.7)
                    if tiene_referencia:
                        ax.axvline(z_ideal[i], color="green", linestyle="--", linewidth=1)
                        ax.axvline(z_nadir[i], color="red", linestyle="--", linewidth=1)
                else:
                    ax.scatter(data[:, j], data[:, i], s=8, alpha=0.5, zorder=3)
                    if tiene_referencia:
                        # eje x = dimensión j, eje y = dimensión i
                        _draw_ideal_nadir_2d(ax, j, i, z_ideal, z_nadir)
                if i == n_dim - 1:
                    ax.set_xlabel(names[j], fontsize=9)
                if j == 0:
                    ax.set_ylabel(names[i], fontsize=9)
                ax.tick_params(labelsize=7)

    fig.suptitle(title)
    plt.tight_layout()

    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        logger.debug(f"Figura guardada en: {save_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig


# ------------------------------------------------------------------
# Ejemplo de uso
# ------------------------------------------------------------------
if __name__ == "__main__":
    dominio = np.random.uniform(-5, 5, size=(500, 1))
    imagen = np.column_stack([dominio[:, 0]**2, (dominio[:, 0] - 2)**2])

    z_ideal = np.array([0.0, 0.0])   # min de f1=x^2 en x=0, min de f2=(x-2)^2 en x=2
    z_nadir = np.array([4.0, 4.0])   # f1 en x=2 -> 4, f2 en x=0 -> 4

    plot_points(
        imagen,
        names=["f1", "f2"],
        title="Frente de Pareto con vector ideal y Nadir",
        z_ideal=z_ideal,
        z_nadir=z_nadir,
    )