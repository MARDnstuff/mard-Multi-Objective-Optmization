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


def plot_nbi(
    F_points: np.ndarray,
    anchors: np.ndarray,
    shadow_minima: np.ndarray,
    direction_rays: np.ndarray = None,
    payoff_matrix: np.ndarray = None,
    betas: np.ndarray = None,
    X: np.ndarray = None,
    front: np.ndarray = None,
    names=("f1", "f2"),
    title="Normal-Boundary Intersection",
    save_path=None,
    dpi=150,
    show=True,
):
    """
    Grafica el resultado de NBI para 2 objetivos: el CHIM (línea entre las
    anclas), los rayos quasi-normales y los puntos NBI encontrados.

    :param F_points: (n_puntos, 2) puntos NBI en el espacio de objetivos
    :param anchors: (2, 2) anclas, la fila i es F(x_i*)
    :param shadow_minima: (2,) vector ideal F*
    :param direction_rays: (2,) dirección quasi-normal n_hat (opcional, dibuja los rayos)
    :param payoff_matrix: (2, 2) matriz de pagos Phi (necesaria junto con betas)
    :param betas: (n_puntos, 2) betas usados (necesarios para ubicar los pines)
    :param front: (m, 2) puntos densos del frente de Pareto real, como referencia (opcional)
    :param X: (n_puntos, N) soluciones en el espacio de decisión (opcional);
              si se da, se agrega un segundo panel con el conjunto de Pareto
    """
    F_points = np.asarray(F_points)
    anchors = np.asarray(anchors)
    shadow_minima = np.asarray(shadow_minima)

    if F_points.shape[1] != 2:
        raise ValueError("plot_nbi solo soporta 2 objetivos; usa plot_points para n >= 3.")

    if X is None:
        fig, ax = plt.subplots(figsize=(6.5, 5.5))
        ax_x = None
    else:
        fig, (ax, ax_x) = plt.subplots(1, 2, figsize=(12, 5.5))

    # CHIM: segmento que une las anclas
    ax.plot(anchors[:, 0], anchors[:, 1], color="gray", linestyle="--", linewidth=1.5,
            label="CHIM", zorder=2)

    # Pines (Phi·beta + F*) y rayos hacia los puntos NBI
    if direction_rays is not None and payoff_matrix is not None and betas is not None:
        pins = (np.asarray(payoff_matrix) @ np.asarray(betas).T).T + shadow_minima
        ax.scatter(pins[:, 0], pins[:, 1], s=18, color="tab:orange", zorder=3, label="Pines")
        for pin, fp in zip(pins, F_points):
            ax.annotate("", xy=fp, xytext=pin,
                        arrowprops=dict(arrowstyle="->", color="tab:orange", alpha=0.6, lw=0.9),
                        zorder=2)

    # Frente de Pareto real (referencia)
    if front is not None:
        front = np.asarray(front)
        front = front[np.argsort(front[:, 0])]
        ax.plot(front[:, 0], front[:, 1], color="black", linewidth=1.2, alpha=0.6,
                label="Frente de Pareto real", zorder=1)

    # Puntos NBI
    ax.scatter(F_points[:, 0], F_points[:, 1], s=28, color="tab:blue", edgecolor="black",
            zorder=4, label="Frente de Pareto (NBI)")

    # Anclas e ideal
    ax.scatter(anchors[:, 0], anchors[:, 1], marker="s", s=60, color="tab:red",
            edgecolor="black", zorder=5, label="Anclas")
    ax.scatter(*shadow_minima, marker="*", s=160, color="green", edgecolor="black",
            zorder=5, label="Ideal F*")

    ax.set_xlabel(names[0])
    ax.set_ylabel(names[1])
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    ax.set_title("Frente de Pareto")

    # Panel 2: conjunto de Pareto (espacio de decisión)
    if ax_x is not None:
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        colores = np.linspace(0, 1, len(X))   # mismo orden que los betas
        if X.shape[1] == 1:
            ax_x.scatter(X[:, 0], np.zeros(len(X)), c=colores, cmap="viridis", s=35, edgecolor="black", zorder=3)
            ax_x.set_yticks([])
            ax_x.set_xlabel("x")
        else:
            ax_x.scatter(X[:, 0], X[:, 1], c=colores, cmap="viridis", s=35, edgecolor="black", zorder=3)
            ax_x.set_xlabel("x1")
            ax_x.set_ylabel("x2")
        ax_x.set_title("Conjunto de Pareto (espacio de decisión)")
        ax_x.grid(alpha=0.3)

    fig.suptitle(title)
    plt.tight_layout()

    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")

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