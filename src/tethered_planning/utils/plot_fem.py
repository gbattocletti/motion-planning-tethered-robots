import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from tethered_planning.env import env_2d
from tethered_planning.utils import plot
from tethered_planning.utils.colors import PlotColors

# Matplotlib settings
labels_font_size = 8
tick_labels_font_size = 8
mpl.rcParams.update(
    {
        "pgf.texsystem": "xelatex",  # or any other engine you want to use
        "text.usetex": True,  # use TeX for all texts
        "font.family": "serif",
        "font.size": labels_font_size,
        "axes.labelsize": labels_font_size,
        "legend.fontsize": labels_font_size,
        "xtick.labelsize": tick_labels_font_size,
        "ytick.labelsize": tick_labels_font_size,
        "pgf.rcfonts": False,
        "pgf.preamble": "\\usepackage[T1]{fontenc}",  # extra preamble for LaTeX
    }
)


def plot_fem(
    env: env_2d.Env2D,
    tether_init: np.ndarray,
    tether_final: np.ndarray | None = None,
    trajectory: np.ndarray | None = None,
    tether_snapshots: list[np.ndarray] | None = None,
    cmap: list[str] | None = None,
    show_plot: bool = False,
    figsize: np.ndarray = np.array([6, 6]),
) -> tuple[plt.Figure, plt.Axes]:
    """
    Plots a FEM simulation, including the tether initial and final configurations, and
    the trajectory followed by the free endpoint (i.e., by the robot).

    Args:
        env (env_2d.Env2D): environment object with obstacles
        tether_init (np.ndarray): initial tether configuration or state
        tether_final (np.ndarray): final tether configuration or state
        trajectory (np.ndarray): trajectory of the robot
        tether_snapshots (list[np.ndarray], optional): list of intermediate
            tether configurations between initial and final configurations
        cmap (list[str], optional): colormap as list of color with same length as
            the tether_snapshots list
        show_plot (bool, optional): wether to show the plot
        figsize (np.ndarray, optional): figure dimensions in cm

    Returns:
        (plt.Figure, plt.Axes): Figure and Axes objects
    """
    # Validate inputs
    if tether_final is None:
        tether_final = tether_init

    # Plotting settings
    linewidth_trajectory = 1.1
    linewidth_initial_final = 1.0
    linewidth_intermediate = 0.9
    markersize_initial_final = 1
    markersize_intermediate = 0.75

    # Select colormap
    if tether_snapshots is not None and (
        cmap is None or len(cmap < len(tether_snapshots))
    ):
        cmap_cont = mpl.colormaps["viridis"]
        cmap = cmap_cont(np.linspace(0, 1, len(tether_snapshots)))

    fig, ax = plot.plot_env(
        env,
        show_anchor=False,
        show_robot=False,
        show_tether=False,
        show_goal=False,
        show_legend=False,
        show_generators=False,
        show_generators_labels=False,
        show_robot_anchor_labels=False,
        show_axes_labels=False,
        show_obstacles_labels=False,
        figsize=figsize,
    )

    # Plot initial tether configuration
    ax.plot(
        tether_init[:, 0],
        tether_init[:, 1],
        "o-",
        color="#3A3A3A",
        linewidth=linewidth_initial_final,
        markersize=markersize_initial_final,
        zorder=6,
    )

    # Plot final tether configuration
    ax.plot(
        tether_final[:, 0],
        tether_final[:, 1],
        "o-",
        color="#000000",
        linewidth=linewidth_initial_final,
        markersize=markersize_initial_final,
        zorder=7,
    )

    # Plot trajectory
    if trajectory is not None:
        ax.plot(
            trajectory[:, 0],
            trajectory[:, 1],
            color="#0086DF",
            linewidth=linewidth_trajectory,
            zorder=8,
        )

    # Plot robot (initial and final)
    ax.plot(
        tether_init[-1, 0],
        tether_init[-1, 1],
        "o",
        color="#0086DF",
        markersize=2,
        zorder=10,
    )
    ax.plot(
        tether_final[-1, 0],
        tether_final[-1, 1],
        "D",
        color="#0086DF",
        markersize=2,
        zorder=10,
    )

    # Plot intermediate tether configurations
    if tether_snapshots is not None:
        for idx, tether in enumerate(tether_snapshots):
            ax.plot(
                tether[:, 0],
                tether[:, 1],
                "o-",
                color=cmap[idx],
                linewidth=linewidth_intermediate,
                markersize=markersize_intermediate,
                zorder=5,
            )

    # Plot anchor and goal
    # NOTE: the anchor and goal in the env object are assumed to be updated and correct
    ax.plot(
        env.anchor_point[0],
        env.anchor_point[1],
        marker="o",
        markersize=2,
        markerfacecolor=PlotColors.anchor_color,
        alpha=1,
        markeredgecolor=PlotColors.anchor_edge_color,
        markeredgewidth=1,
        zorder=9,
    )
    ax.plot(
        env.goal_vertices[0],
        env.goal_vertices[1],
        marker="o",
        markersize=2,
        markerfacecolor=PlotColors.goal_color,
        alpha=1,
        markeredgecolor=PlotColors.goal_color,
        markeredgewidth=1,
        zorder=9,
    )
    ax.plot(
        env.goal_vertices[0],
        env.goal_vertices[1],
        marker="o",
        markersize=10,
        markerfacecolor=PlotColors.goal_color,
        alpha=0.3,
        markeredgecolor=PlotColors.goal_color,
        markeredgewidth=1,
        zorder=4,
    )

    # Set aspect ratio, axes limits, and labels
    # CHECKME: may be redundant since it is already covered in plot_env
    ax.set_aspect("equal", "box")
    ax.set_xlim([0, env.size[0]])
    ax.set_ylim([0, env.size[1]])
    ax.grid(True, which="major", linestyle=":", color="gray", linewidth=0.5, zorder=1)
    ax.grid(True, which="minor", linestyle=":", color="gray", linewidth=0.3, zorder=1)
    ax.minorticks_on()
    ax.set_xlabel("")
    ax.set_xlabel("")
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    # Save and show plot
    if show_plot is True:
        plt.show()

    return fig, ax
