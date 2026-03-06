import numpy as np
from scipy.optimize import minimize
from typing import Callable, Union


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
        k: Union[float, np.ndarray],
        u_initial: Callable[[np.ndarray], np.ndarray],
        u_left: Callable[[np.ndarray], np.ndarray],
        u_right: Callable[[np.ndarray], np.ndarray],
    ) -> np.ndarray:
        """
        Solve the direct heat equation problem u_t = k(x) * u_xx.

        Parameters:
        -----------
        k : float or np.ndarray of shape (nx,)
            Thermal diffusivity — scalar for constant, array for spatially varying.
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
        k_arr = np.full(self.nx, float(k)) if np.isscalar(k) else np.asarray(k, dtype=float)
        r = k_arr * self.dt / self.dx**2

        if np.any(r > 0.5):
            print(f"Warning: max stability parameter r = {np.max(r):.4f} > 0.5")
            print(f"Consider reducing dt or increasing dx")

        u = np.zeros((self.nt, self.nx))
        u[0, :] = u_initial(self.x)

        for n in range(self.nt - 1):
            u[n + 1, 1:-1] = u[n, 1:-1] + r[1:-1] * (u[n, 2:] - 2 * u[n, 1:-1] + u[n, :-2])
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
    """Solves the inverse problem to estimate spatially-varying k(x)."""

    def __init__(
        self,
        solver: HeatEquationSolver,
        measurement_points: np.ndarray,
        u_measured: np.ndarray,
        u_initial: Callable,
        u_left: Callable,
        u_right: Callable,
        degree: int = 4,
        alpha: float = 0.0,
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
        degree : int
            Degree of the polynomial used to parameterize k(x).
            k is represented as k(x) = c_0 + c_1*(x/L) + ... + c_d*(x/L)^d,
            where x is normalized by L for numerical stability.
        alpha : float
            Tikhonov regularization weight (penalizes large coefficients).
        """
        self.solver = solver
        self.measurement_points = measurement_points
        self.u_measured = u_measured
        self.u_initial = u_initial
        self.u_left = u_left
        self.u_right = u_right
        self.degree = degree
        self.alpha = alpha
        self.history = []

        # Vandermonde matrix: rows = spatial grid points, cols = polynomial terms
        x_norm = solver.x / solver.L  # normalize to [0, 1]
        self.V = np.vander(x_norm, degree + 1, increasing=True)  # shape (nx, degree+1)

    def _coeffs_to_k_array(self, coeffs: np.ndarray) -> np.ndarray:
        """Evaluate the polynomial k(x) = V @ coeffs on the full spatial grid."""
        return self.V @ coeffs

    def objective_function(self, coeffs: np.ndarray) -> float:
        """
        Compute the objective function (least-squares error + regularization).

        Parameters:
        -----------
        coeffs : np.ndarray of shape (degree+1,)
            Polynomial coefficients [c_0, c_1, ..., c_degree]

        Returns:
        --------
        error : float
        """
        k_arr = self._coeffs_to_k_array(coeffs)
        if np.any(k_arr <= 0):
            return 1e10

        u_computed = self.solver.solve_direct(k_arr, self.u_initial, self.u_left, self.u_right)
        u_at_points = u_computed[:, self.measurement_points]
        error = float(np.sum((u_at_points - self.u_measured) ** 2))

        if self.alpha > 0:
            error += self.alpha * float(np.sum(coeffs[1:] ** 2))

        self.history.append((coeffs.copy(), error))
        return error

    def solve(
        self,
        k_initial_guess: Union[float, np.ndarray] = 0.01,
        method: str = "L-BFGS-B",
    ) -> dict:
        """
        Solve the inverse problem to find optimal k(x).

        Parameters:
        -----------
        k_initial_guess : float or np.ndarray
            Initial guess for k. A scalar sets the constant term c_0 with
            all higher-order coefficients initialized to zero. An array of
            length (degree+1) sets the full coefficient vector directly.
        method : str
            Optimization method (see scipy.optimize.minimize)

        Returns:
        --------
        result : dict
            Dictionary containing optimization results
        """
        if np.isscalar(k_initial_guess):
            x0 = np.zeros(self.degree + 1)
            x0[0] = float(k_initial_guess)
        else:
            x0 = np.asarray(k_initial_guess, dtype=float)

        print(f"Starting inverse problem solution...")
        print(f"Initial coefficients: {x0}")
        print(f"Optimization method: {method}")
        print(f"Polynomial degree: {self.degree}")

        self.history = []
        result = minimize(
            self.objective_function,
            x0=x0,
            method=method,
            options={"disp": True, "maxiter": 2000},
        )

        coeffs_estimated = result.x
        k_estimated_arr = self._coeffs_to_k_array(coeffs_estimated)

        print(f"\nOptimization complete!")
        print(f"Estimated polynomial coefficients = {coeffs_estimated}")
        print(f"Final objective value = {result.fun:.6e}")
        print(f"Number of iterations = {len(self.history)}")

        return {
            "coeffs_estimated": coeffs_estimated,
            "k_estimated_arr": k_estimated_arr,
            "objective_value": result.fun,
            "success": result.success,
            "message": result.message,
            "history": self.history,
            "result": result,
        }
