import numpy as np
from scipy.optimize import minimize
from typing import Callable


class HeatEquationSolver:
    """Solves the 1D heat equation using finite differences."""

    def __init__(self, L: float, T: float, nx: int, nt: int):
        """
        Parameters:
        -----------
        L : float
            Length of spatial domain
        T : float
            Final time
        nx : int
            Number of spatial grid points
        nt : int
            Number of time steps
        """
        self.L = L
        self.T = T
        self.nx = nx
        self.nt = nt

        self.x = np.linspace(0, L, nx)
        self.dx = L / (nx - 1)
        self.t = np.linspace(0, T, nt)
        self.dt = T / (nt - 1)

    def solve_direct(
        self,
        k: float,
        u_initial: Callable[[np.ndarray], np.ndarray],
        u_left: Callable[[np.ndarray], np.ndarray],
        u_right: Callable[[np.ndarray], np.ndarray],
    ) -> np.ndarray:
        """
        Solve the direct heat equation problem.

        Parameters:
        -----------
        k : float
            Thermal diffusivity
        u_initial : callable
            Initial condition u(x, 0) = u_initial(x)
        u_left : callable
            Left boundary condition u(0, t) = u_left(t)
        u_right : callable
            Right boundary condition u(L, t) = u_right(t)

        Returns:
        --------
        u : np.ndarray
            Solution u(x, t) of shape (nt, nx)
        """
        r = k * self.dt / (self.dx**2)
        if r > 0.5:
            print(f"Warning: Stability parameter r = {r:.4f} > 0.5")
            print(f"Consider reducing dt or increasing dx")

        u = np.zeros((self.nt, self.nx))

        u[0, :] = u_initial(self.x)
        for n in range(0, self.nt - 1):
            for i in range(1, self.nx - 1):
                u[n + 1, i] = u[n, i] + r * (u[n, i + 1] - 2 * u[n, i] + u[n, i - 1])
            u[n + 1, 0] = u_left(self.t[n + 1])
            u[n + 1, -1] = u_right(self.t[n + 1])

        return u

    def extract_measurements(
        self, u: np.ndarray, measurement_points: np.ndarray
    ) -> np.ndarray:
        """
        Extract temperature measurements at specific spatial points.

        Parameters:
        -----------
        u : np.ndarray
            Full solution u(x, t) of shape (nt, nx)
        measurement_points : np.ndarray
            Indices of measurement points (including boundaries)

        Returns:
        --------
        u_measured : np.ndarray
            Measurements at specified points, shape (nt, n_measurements)
        """
        return u[:, measurement_points]


class InverseProblemSolver:
    """Solves the inverse problem to estimate thermal diffusivity k."""

    def __init__(
        self,
        solver: HeatEquationSolver,
        measurement_points: np.ndarray,
        u_measured: np.ndarray,
        u_initial: Callable,
        u_left: Callable,
        u_right: Callable,
    ):
        """
        Parameters:
        -----------
        solver : HeatEquationSolver
            Heat equation solver instance
        measurement_points : np.ndarray
            Indices of measurement points
        u_measured : np.ndarray
            Measured temperature data, shape (nt, n_measurements)
        u_initial : callable
            Initial condition
        u_left : callable
            Left boundary condition
        u_right : callable
            Right boundary condition
        """
        self.solver = solver
        self.measurement_points = measurement_points
        self.u_measured = u_measured
        self.u_initial = u_initial
        self.u_left = u_left
        self.u_right = u_right
        self.history = []

    def objective_function(self, k: float) -> float:
        """
        Compute the objective function (least squares error).

        Parameters:
        -----------
        k : float or array-like
            Thermal diffusivity parameter

        Returns:
        --------
        error : float
            Sum of squared errors between computed and measured data
        """
        k_val = float(k[0]) if hasattr(k, "__iter__") else float(k)  # type: ignore

        if k_val <= 0:
            return 1e10

        u_computed = self.solver.solve_direct(
            k_val, self.u_initial, self.u_left, self.u_right
        )
        u_at_points = u_computed[:, self.measurement_points]
        error = np.sum((u_at_points - self.u_measured) ** 2)
        self.history.append((k_val, error))

        return error  # type: ignore

    def solve(self, k_initial_guess: float = 0.01, method: str = "Nelder-Mead") -> dict:
        """
        Solve the inverse problem to find optimal k.

        Parameters:
        -----------
        k_initial_guess : float
            Initial guess for k
        method : str
            Optimization method (see scipy.optimize.minimize)

        Returns:
        --------
        result : dict
            Dictionary containing optimization results
        """
        print(f"Starting inverse problem solution...")
        print(f"Initial guess: k = {k_initial_guess}")
        print(f"Optimization method: {method}")

        self.history = []
        result = minimize(
            self.objective_function,
            x0=k_initial_guess,
            method=method,
            options={"disp": True, "maxiter": 1000},
        )

        k_estimated = result.x[0] if hasattr(result.x, "__iter__") else result.x

        print(f"\nOptimization complete!")
        print(f"Estimated k = {k_estimated:.6f}")
        print(f"Final objective value = {result.fun:.6e}")
        print(f"Number of iterations = {len(self.history)}")

        return {
            "k_estimated": k_estimated,
            "objective_value": result.fun,
            "success": result.success,
            "message": result.message,
            "history": self.history,
            "result": result,
        }
