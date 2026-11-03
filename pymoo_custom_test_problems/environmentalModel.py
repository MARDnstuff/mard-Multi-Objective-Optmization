import numpy as np
from pymoo.core.problem import Problem

def environ_output(x:np.ndarray, s=None, t=None):
    """
    Raw functional output of the environmental model.

    x = (M, D, L, tau), shape (n_points, 4).
    Returns shape (n_points, len(s) * len(t)), ordered by location, then time.
    C(s,t) = M/sqrt(4 pi D t) exp(-s^2/(4 D t)) + I(tau < t) M/sqrt(4 pi D (t - tau)) exp(-(s - L)^2/(4 D (t - tau)))
    Output Y = sqrt(4 pi) C(s,t).
    Default grids follow the SFU code: s = 0.5,...,2.5 ; t = 0.3, 0.6, ..., 60.
    """
    x = np.atleast_2d(x)
    s = np.array([0.5, 1.0, 1.5, 2.0, 2.5]) if s is None else np.asarray(s, float)
    t = np.arange(1, 201) * 0.3 if t is None else np.asarray(t, float)

    M, D, L, tau = (x[:, i][:, None, None] for i in range(4))
    S, T = s[None, :, None], t[None, None, :]

    term1 = M / np.sqrt(4 * np.pi * D * T) * np.exp(-S**2 / (4 * D * T))
    dt = T - tau
    active = dt > 0
    dt_safe = np.where(active, dt, 1.0)            # avoid sqrt/div of <= 0
    term2 = np.where(active, M / np.sqrt(4 * np.pi * D * dt_safe) * np.exp(-(S - L) ** 2 / (4 * D * dt_safe)), 0.0)
    Y = np.sqrt(4 * np.pi) * (term1 + term2)
    return Y.reshape(x.shape[0], -1)


class EnvironmentalModel(Problem):
    """
    Environmental Model Function
    source: https://www.sfu.ca/~ssurjano/environ.html

    Calibration problem: find (M, D, L, tau) whose output matches observed data.

    Objective: sum of squared errors between environ_output(x) and `observed`.
    If `observed` is None, it is generated at the reference values used on the
    SFU page (M=10, D=0.07, L=1.505, tau=30.1525), so the optimum is f = 0 there.
    Domain: M in [7, 13], D in [0.02, 0.12], L in [0.01, 3], tau in [30.01, 30.295].

    The SSE objective is one common way to turn the functional output into a
    single objective; the SFU page itself does not define a misfit.
    """

    TRUE_X = np.array([10.0, 0.07, 1.505, 30.1525])

    def __init__(self, observed=None, s=None, t=None) -> None:
        super().__init__(n_var=4, n_obj=1, xl=[7, 0.02, 0.01, 30.01], xu=[13, 0.12, 3, 30.295])
        self.s, self.t = s, t
        self.observed = (environ_output(self.TRUE_X, s, t)[0] if observed is None else np.asarray(observed, float))
        self.foo_name = 'Environmental_Model'
        self.f_opt = 0.0

    def _evaluate(self, x: np.ndarray, out: dict, *args, **kwargs) -> float:
        Y = environ_output(x, self.s, self.t)
        out["F"] = np.sum((Y - self.observed) ** 2, axis=1)