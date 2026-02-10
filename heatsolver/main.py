"""
Inverse Problem for Heat Equation
==================================
Solves the 1D heat equation inverse problem to estimate thermal diffusivity k.

Heat equation: u_t = k * u_xx
Domain: x ∈ [0, L], t ∈ [0, T]
Boundary conditions: u(0,t) = u_0(t), u(L,t) = u_L(t)
Initial condition: u(x,0) = f(x)

Direct problem: Given k, solve for u(x,t)
Inverse problem: Given measurements u(x_i, t), estimate k
"""

import numpy as np
import matplotlib.pyplot as plt
from solvers import HeatEquationSolver, InverseProblemSolver
from plots import plot_results


def main():
    """Main function to demonstrate the inverse problem solution."""

    print("=" * 70)
    print("INVERSE PROBLEM FOR HEAT EQUATION")
    print("=" * 70)

    # Problem parameters
    L = 1.0  # Length of domain
    T = 0.5  # Final time
    nx = 51  # Number of spatial points
    nt = 501  # Number of time steps
    k_true = 0.01  # True thermal diffusivity (to be estimated)

    print(f"\nProblem setup:")
    print(f"  Domain: x ∈ [0, {L}], t ∈ [0, {T}]")
    print(f"  Grid: {nx} spatial points, {nt} time steps")
    print(f"  True k = {k_true}")

    def u_initial(x):
        """Initial condition: u(x, 0) = sin(πx)"""
        return np.sin(np.pi * x)

    def u_left(t):
        """Left boundary: u(0, t) = 0"""
        return 0.0

    def u_right(t):
        """Right boundary: u(L, t) = 0"""
        return 0.0

    solver = HeatEquationSolver(L, T, nx, nt)

    print(f"\n{'='*70}")
    print("STEP 1: Solving direct problem with true k")
    print(f"{'='*70}")

    u_true = solver.solve_direct(k_true, u_initial, u_left, u_right)  # type: ignore
    print(f"✓ Direct problem solved")
    print(f"  Solution shape: {u_true.shape}")
    print(f"  u range: [{np.min(u_true):.6f}, {np.max(u_true):.6f}]")

    print(f"\n{'='*70}")
    print("STEP 2: Extracting measurements")
    print(f"{'='*70}")

    n_interior_points = 5
    interior_indices = np.linspace(1, nx - 2, n_interior_points, dtype=int)
    measurement_points = np.concatenate([[0], interior_indices, [nx - 1]])

    print(f"Measurement points:")
    for i, idx in enumerate(measurement_points):
        print(f"  x_{i} = {solver.x[idx]:.4f} (index {idx})")

    u_measured = solver.extract_measurements(u_true, measurement_points)
    print(f"✓ Measurements extracted")
    print(f"  Measurement array shape: {u_measured.shape}")

    noise_level = 0.0
    if noise_level > 0:
        noise = np.random.normal(0, noise_level * np.std(u_measured), u_measured.shape)
        u_measured += noise
        print(f"  Added {noise_level*100}% noise to measurements")

    print(f"\n{'='*70}")
    print("STEP 3: Solving inverse problem")
    print(f"{'='*70}")

    inverse_solver = InverseProblemSolver(
        solver, measurement_points, u_measured, u_initial, u_left, u_right
    )

    k_initial_guess = 0.02
    result = inverse_solver.solve(k_initial_guess, method="Nelder-Mead")

    k_estimated = result["k_estimated"]

    print(f"\n{'='*70}")
    print("STEP 4: Verification")
    print(f"{'='*70}")

    u_reconstructed = solver.solve_direct(k_estimated, u_initial, u_left, u_right)  # type: ignore

    relative_error_k = abs(k_estimated - k_true) / k_true * 100
    max_error_u = np.max(np.abs(u_true - u_reconstructed))
    mean_error_u = np.mean(np.abs(u_true - u_reconstructed))

    print(f"\nResults:")
    print(f"  True k              = {k_true}")
    print(f"  Estimated k         = {k_estimated:.6f}")
    print(f"  Relative error in k = {relative_error_k:.4f}%")
    print(f"  Max error in u      = {max_error_u:.6e}")
    print(f"  Mean error in u     = {mean_error_u:.6e}")

    print(f"\n{'='*70}")
    print("STEP 5: Generating plots")
    print(f"{'='*70}")

    fig = plot_results(
        solver,
        u_true,
        u_reconstructed,
        measurement_points,
        k_true,
        k_estimated,
        result["history"],
    )

    plt.savefig("results/inverse_heat_results.png", dpi=150, bbox_inches="tight")
    print(f"✓ Results saved to 'results/inverse_heat_results.png'")

    plt.show()

    print(f"\n{'='*70}")
    print("INVERSE PROBLEM SOLUTION COMPLETE")
    print(f"{'='*70}")

    return result, u_true, u_reconstructed


if __name__ == "__main__":
    result, u_true, u_reconstructed = main()
