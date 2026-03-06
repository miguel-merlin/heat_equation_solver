import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from solvers import HeatEquationSolver


def plot_results(
    solver: HeatEquationSolver,
    u_true: np.ndarray,
    u_reconstructed: np.ndarray,
    measurement_points: np.ndarray,
    k_true_arr: np.ndarray,
    k_estimated_arr: np.ndarray,
    history: list,
):
    """Plot comparison of true and reconstructed solutions and k(x)."""

    fig = plt.figure(figsize=(16, 10))

    ax1 = fig.add_subplot(2, 3, 1, projection="3d")
    X, T = np.meshgrid(solver.x, solver.t)
    ax1.plot_surface(X, T, u_true, cmap="viridis", alpha=0.8)
    ax1.set_xlabel("x")
    ax1.set_ylabel("t")
    ax1.set_zlabel("u(x,t)")
    ax1.set_title("True Solution")

    ax2 = fig.add_subplot(2, 3, 2, projection="3d")
    ax2.plot_surface(X, T, u_reconstructed, cmap="plasma", alpha=0.8)
    ax2.set_xlabel("x")
    ax2.set_ylabel("t")
    ax2.set_zlabel("u(x,t)")
    ax2.set_title("Reconstructed Solution")

    ax3 = fig.add_subplot(2, 3, 3, projection="3d")
    error = np.abs(u_true - u_reconstructed) / (np.abs(u_true) + 1e-12)
    ax3.plot_surface(X, T, error, cmap="hot", alpha=0.8)
    ax3.set_xlabel("x")
    ax3.set_ylabel("t")
    ax3.set_zlabel("relative error")
    ax3.set_title(f"Relative Error\nMax = {np.max(error):.2e}")

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    ax4 = fig.add_subplot(2, 3, 4)
    time_indices = [
        0,
        len(solver.t) // 4,
        len(solver.t) // 2,
        3 * len(solver.t) // 4,
        -1,
    ]
    for i, idx in enumerate(time_indices):
        color = colors[i % len(colors)]
        ax4.plot(solver.x, u_true[idx, :], "--", color=color, alpha=0.7, label=f"t={solver.t[idx]:.3f}")
        ax4.plot(solver.x, u_reconstructed[idx, :], "-", color=color, alpha=0.9)
    ax4.scatter(
        solver.x[measurement_points],
        np.zeros(len(measurement_points)),
        c="red",
        marker="x",  # type: ignore
        s=100,
        zorder=5,
    )
    ax4_legend = (
        [Line2D([0], [0], linestyle="--", color="k", label="True"),
         Line2D([0], [0], linestyle="-",  color="k", label="Reconstructed"),
         Line2D([0], [0], linestyle="", marker="x", color="red", markersize=8, label="Measurement points")]
        + [Line2D([0], [0], color=colors[i % len(colors)], label=f"t={solver.t[idx]:.3f}")
           for i, idx in enumerate(time_indices)]
    )
    ax4.legend(handles=ax4_legend, fontsize=8)
    ax4.set_xlabel("x")
    ax4.set_ylabel("u(x,t)")
    ax4.set_title("Solution Snapshots at Different Times")
    ax4.grid(True, alpha=0.3)

    ax5 = fig.add_subplot(2, 3, 5)
    for i, point_idx in enumerate(measurement_points):
        color = colors[i % len(colors)]
        ax5.plot(solver.t, u_true[:, point_idx], "--", color=color, alpha=0.7, label=f"x={solver.x[point_idx]:.2f}")
        ax5.plot(solver.t, u_reconstructed[:, point_idx], "-", color=color, alpha=0.9)
    ax5_legend = (
        [Line2D([0], [0], linestyle="--", color="k", label="True"),
         Line2D([0], [0], linestyle="-",  color="k", label="Reconstructed")]
        + [Line2D([0], [0], color=colors[i % len(colors)], label=f"x={solver.x[pt]:.2f}")
           for i, pt in enumerate(measurement_points)]
    )
    ax5.legend(handles=ax5_legend, fontsize=8)
    ax5.set_xlabel("t")
    ax5.set_ylabel("u(x,t)")
    ax5.set_title("Temperature at Measurement Points")
    ax5.grid(True, alpha=0.3)

    ax6 = fig.add_subplot(2, 3, 6)
    ax6.plot(solver.x, k_true_arr, "g-", linewidth=2, label="True k(x)")
    ax6.plot(solver.x, k_estimated_arr, "r--", linewidth=2, label="Estimated k(x)")
    ax6.set_xlabel("x")
    ax6.set_ylabel("k(x)")
    ax6.set_title("True vs Estimated k(x)")
    ax6.legend()
    ax6.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig
