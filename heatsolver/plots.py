import numpy as np
import matplotlib.pyplot as plt
from solvers import HeatEquationSolver


def plot_results(
    solver: HeatEquationSolver,
    u_true: np.ndarray,
    u_reconstructed: np.ndarray,
    measurement_points: np.ndarray,
    k_true: float,
    k_estimated: float,
    history: list,
):
    """Plot comparison of true and reconstructed solutions."""

    fig = plt.figure(figsize=(16, 10))

    ax1 = fig.add_subplot(2, 3, 1, projection="3d")
    X, T = np.meshgrid(solver.x, solver.t)
    ax1.plot_surface(X, T, u_true, cmap="viridis", alpha=0.8)
    ax1.set_xlabel("x")
    ax1.set_ylabel("t")
    ax1.set_zlabel("u(x,t)")
    ax1.set_title(f"True Solution (k = {k_true})")

    ax2 = fig.add_subplot(2, 3, 2, projection="3d")
    ax2.plot_surface(X, T, u_reconstructed, cmap="plasma", alpha=0.8)
    ax2.set_xlabel("x")
    ax2.set_ylabel("t")
    ax2.set_zlabel("u(x,t)")
    ax2.set_title(f"Reconstructed Solution (k = {k_estimated:.6f})")

    ax3 = fig.add_subplot(2, 3, 3, projection="3d")
    error = np.abs(u_true - u_reconstructed)
    ax3.plot_surface(X, T, error, cmap="hot", alpha=0.8)
    ax3.set_xlabel("x")
    ax3.set_ylabel("t")
    ax3.set_zlabel("|error|")
    ax3.set_title(f"Absolute Error\nMax error = {np.max(error):.6f}")

    ax4 = fig.add_subplot(2, 3, 4)
    time_indices = [
        0,
        len(solver.t) // 4,
        len(solver.t) // 2,
        3 * len(solver.t) // 4,
        -1,
    ]
    for idx in time_indices:
        ax4.plot(
            solver.x,
            u_true[idx, :],
            "--",
            alpha=0.6,
            label=f"t={solver.t[idx]:.3f} (true)",
        )
        ax4.plot(solver.x, u_reconstructed[idx, :], "-", alpha=0.8)
    ax4.scatter(
        solver.x[measurement_points],
        np.zeros(len(measurement_points)),
        c="red",
        marker="x",  # type: ignore
        s=100,
        zorder=5,
        label="Measurement points",
    )
    ax4.set_xlabel("x")
    ax4.set_ylabel("u(x,t)")
    ax4.set_title("Solution Snapshots at Different Times")
    ax4.legend(fontsize=8)
    ax4.grid(True, alpha=0.3)

    ax5 = fig.add_subplot(2, 3, 5)
    for i, point_idx in enumerate(measurement_points):
        ax5.plot(
            solver.t,
            u_true[:, point_idx],
            "--",
            alpha=0.6,
            label=f"x={solver.x[point_idx]:.2f} (true)",
        )
        ax5.plot(solver.t, u_reconstructed[:, point_idx], "-", alpha=0.8)
    ax5.set_xlabel("t")
    ax5.set_ylabel("u(x,t)")
    ax5.set_title("Temperature at Measurement Points")
    ax5.legend(fontsize=8)
    ax5.grid(True, alpha=0.3)

    ax6 = fig.add_subplot(2, 3, 6)
    k_values = [h[0] for h in history]
    obj_values = [h[1] for h in history]
    ax6.semilogy(k_values, obj_values, "b.-", alpha=0.7)
    ax6.axvline(
        k_true, color="g", linestyle="--", linewidth=2, label=f"True k = {k_true}"
    )
    ax6.axvline(
        k_estimated,
        color="r",
        linestyle="--",
        linewidth=2,
        label=f"Estimated k = {k_estimated:.6f}",
    )
    ax6.set_xlabel("k")
    ax6.set_ylabel("Objective Function (log scale)")
    ax6.set_title("Optimization History")
    ax6.legend()
    ax6.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig
