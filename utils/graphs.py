import numpy as np
import matplotlib.pyplot as plt
import logging

logger = logging.getLogger(__name__)

def plot_points(data: np.ndarray, names=None, title=""):
    """
    Grafica un conjunto de puntos de n dimensiones.

    :param data: array-like de forma (n_puntos, n_dim)
    :param names: lista de nombres para cada dimensión (opcional)
    :param title: título de la gráfica
    """
    data = np.asarray(data)
    n_puntos, n_dim = data.shape
    logger.debug(f"Data shape dimensions: {n_dim}")
    
    if names is None or len(names) < n_dim:
        names = [f"x{i+1}" for i in range(n_dim)]

    if n_dim == 1:
        fig, ax = plt.subplots(figsize=(6, 1.5))
        ax.scatter(data[:, 0], np.zeros(n_puntos), s=8, alpha=0.5)
        ax.set_yticks([])
        ax.set_xlabel(names[0])

    elif n_dim == 2:
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.scatter(data[:, 0], data[:, 1], s=10, alpha=0.6)
        ax.set_xlabel(names[0])
        ax.set_ylabel(names[1])

    elif n_dim == 3:
        fig = plt.figure(figsize=(6, 5))
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(data[:, 0], data[:, 1], data[:, 2], s=10, alpha=0.6)
        ax.set_xlabel(names[0])
        ax.set_ylabel(names[1])
        ax.set_zlabel(names[2])

    else:
        # más de 3 dimensiones -> matriz de dispersión por pares
        fig, axes = plt.subplots(n_dim, n_dim, figsize=(2.2 * n_dim, 2.2 * n_dim))
        for i in range(n_dim):
            for j in range(n_dim):
                ax = axes[i, j]
                if i == j:
                    ax.hist(data[:, i], bins=25, alpha=0.7)
                else:
                    ax.scatter(data[:, j], data[:, i], s=8, alpha=0.5)
                if i == n_dim - 1:
                    ax.set_xlabel(names[j], fontsize=9)
                if j == 0:
                    ax.set_ylabel(names[i], fontsize=9)
                ax.tick_params(labelsize=7)

    fig.suptitle(title)
    plt.tight_layout()
    plt.show()


# Ejemplo de uso
if __name__ == "__main__":
    dominio = np.random.uniform(-5, 5, size=(500, 2))
    imagen = np.column_stack([dominio[:, 0]**2, (dominio[:, 0] - 2)**2])

    plot_points(dominio, names=["x1", "x2"], title="Dominio")
    plot_points(imagen, names=["f1", "f2"], title="Imagen")