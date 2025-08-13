from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib as mpl
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from shapely.geometry import LineString, MultiLineString
from shapely.plotting import plot_line, plot_polygon

from ..utils.colors import PlotColors

if TYPE_CHECKING:
    from ..env.env_2d import Env2D
    from .settings import Settings

# TODO: merge all the plot functions in a single one with many kwargs that can be
# passed to control which elements are plotted.

# NOTE: Plots implementation details:
# - To modify the colors of the plot, change the values in the PlotColors class
#   in the colors.py file.
# - The indexes of the generators start from 1 following the convention described in
#   the compute_signature function in the curve_fcns.py file.

# Matplotlib settings
labels_font_size = 10
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

# dictionary with the legend settings
legend_settings = {
    "alignment": "left",
    "title": "Legend",
    "prop": {"weight": "bold"},
    "fancybox": False,
    "shadow": True,
    "bbox_to_anchor": (1, 0.5, 0.2, 0.5),
    "loc": "upper left",
}


def plot_env(env: Env2D, settings: Settings, **kwargs) -> tuple[plt.Figure, plt.Axes]:
    """
    Plot the environment.

    Args:
        env (Env2D): Env object to plot
        settings (Settings): Settings object with the plot settings. Note: the plot
        colors depend on the PlotColors class and not on the Settings object.

    Kwargs:
        show_anchor (bool, **kwargs): flag to display the anchor point
        show_goal (bool, **kwargs): flag to display the goal region
        show_legend (bool, **kwargs): flag to display the legend
        label_generators (bool, **kwargs): flag to label the generators

    Returns:
        tuple[plt.Figure, plt.Axes]: Figure and Axes objects

    Raises:
        ValueError: If any of the kwargs are not of the expected type.
    """
    # Kwargs default values
    show_anchor: bool = True  # show the anchor point
    show_goal: bool = True  # show the goal region
    show_legend: bool = settings.plot.show_legend  # display the legend
    label_generators: bool = True  # label the generators

    # Parse kwargs
    for key, value in kwargs.items():
        if key == "show_anchor":
            if not isinstance(value, bool):
                raise ValueError(
                    f"Expected bool for show_anchor, got {type(value)} instead."
                )
            show_anchor = value
        elif key == "show_goal":
            if not isinstance(value, bool):
                raise ValueError(
                    f"Expected bool for show_goal, got {type(value)} instead."
                )
            show_goal = value
        elif key == "show_legend":
            if not isinstance(value, bool):
                raise ValueError(
                    f"Expected bool for show_legend, got {type(value)} instead."
                )
            show_legend = value
        elif key == "label_generators":
            if not isinstance(value, bool):
                raise ValueError(
                    f"Expected bool for label_generators, got {type(value)} instead."
                )
            label_generators = value
        else:
            pass  # ignore unknown kwargs

    # Create figure object
    fig = plt.figure(figsize=settings.plot.figsize)
    ax = plt.gca()
    ax.set_aspect("equal", "box")
    ax.set_xlim([0, env.size[0]])
    ax.set_ylim([0, env.size[1]])
    ax.grid(True, which="major", linestyle=":", color="gray", linewidth=0.5, zorder=1)
    ax.grid(True, which="minor", linestyle=":", color="gray", linewidth=0.3, zorder=1)
    ax.minorticks_on()

    # Plot the obstacle region
    if not env.obstacle_region.is_empty:
        plot_polygon(
            env.obstacle_region,
            ax=ax,
            color=PlotColors.obstacles_color,
            alpha=1,
            edgecolor=PlotColors.obstacles_edges_color,
            linewidth=1,
            add_points=False,
            zorder=4,
        )
    obs_handle = Patch(color=PlotColors.obstacles_color, label="Obstacles")

    # Plot generators
    if env.generators_list:
        plot_line(
            env.generators,
            ax=ax,
            color=PlotColors.generators_color,
            alpha=1,
            linewidth=1,
            add_points=False,
            zorder=3,
        )
        if label_generators:
            if isinstance(env.generators, MultiLineString):
                for idx, generator in enumerate(env.generators.geoms):
                    ax.text(
                        generator.coords[0][0] + 0.2,
                        generator.coords[0][1] + 0.2,
                        f"$g_{idx+1}$",  # latex mathmode
                        fontsize=6,
                    )
            elif isinstance(env.generators, LineString):
                ax.text(
                    env.generators.coords[0][0] + 0.2,
                    env.generators.coords[0][1] + 0.2,
                    f"$g_{1}$",  # latex mathmode
                    fontsize=6,
                )
    generators_handle = Line2D(
        [], [], color=PlotColors.generators_color, lw=1, label="Generators"
    )

    # Plot the goal region
    if show_goal:
        if not env.goal_region.is_empty:
            plot_polygon(
                env.goal_region,
                ax=ax,
                color=PlotColors.goal_color,
                alpha=1,
                edgecolor=PlotColors.goal_edge_color,
                linewidth=1,
                add_points=False,
                zorder=2,
            )
        goal_handle = Patch(color=PlotColors.goal_color, label="Goal")

    # Plot anchor point
    if show_anchor:
        (anchor_handle,) = plt.plot(
            env.anchor_point[0],
            env.anchor_point[1],
            marker="o",
            markersize=6,
            markerfacecolor=PlotColors.anchor_color,
            alpha=1,
            markeredgecolor=PlotColors.anchor_edge_color,
            markeredgewidth=1,
            zorder=9,
            label="Anchor Point",
            linestyle="None",
        )
        ax.text(
            env.anchor_point[0] - 0.5,
            env.anchor_point[1] - 0.5,
            r"$x_\mathrm{a}$",  # latex mathmode
            fontsize=10,
        )

    # Add labels and title
    ax.set_title(
        settings.plot.title,
        **{
            "fontsize": 12,
            "fontweight": "bold",
        },
    )
    ax.set_xlabel(settings.plot.x_label, rotation=0)
    ax.set_ylabel(settings.plot.y_label, rotation=0)

    # Add legend
    # https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.legend.html#matplotlib.axes.Axes.legend
    # https://stackoverflow.com/questions/4700614/how-to-put-the-legend-outside-the-plot
    if show_legend:
        box = ax.get_position()
        ax.set_position(
            [
                box.x0,
                box.y0,
                box.width * 0.8,
                box.height,
            ]
        )
        handles = [obs_handle, generators_handle]
        if show_goal:
            handles.append(goal_handle)
        if show_anchor:
            handles.append(anchor_handle)
        ax.legend(handles=handles, **legend_settings)

    return fig, ax


def plot_tether(
    env: Env2D, settings: Settings, **kwargs
) -> tuple[plt.Figure, plt.Axes]:
    """
    Plot the environment with the tether configuration and the robot.

    Args:
        env (Env2D): Env object to plot. It contains also the information on the robot
            location and the tether configuration.
        settings (Settings): Settings object with the plot settings.

    Kwargs:
        tether (LineString): Tether object (if not specified the tether configuration
            is obtained from the env object).
        show_legend (bool): Display the legend.
        show_robot (bool): Display the robot location.
        **kwargs: The function accepts all the kwargs of the plot_env function.

    Returns:
        tuple[plt.Figure, plt.Axes]: Figure and Axes objects.

    Raises:
        ValueError: If any of the kwargs are not of the expected type.
    """

    # Kwargs default values
    tether: LineString = None  # tether object
    show_robot: bool = True  # show the robot location
    show_legend: bool = settings.plot.show_legend  # display the legend

    # Parse kwargs
    for key, value in kwargs.items():
        if key == "tether":
            if not isinstance(value, LineString):
                raise ValueError(
                    f"Expected LineString for tether, got {type(value)} instead."
                )
            tether = value
        if key == "show_robot":
            if not isinstance(value, bool):
                raise ValueError(
                    f"Expected bool for show_robot, got {type(value)} instead."
                )
            show_robot = value
        if key == "show_legend":
            if not isinstance(value, bool):
                raise ValueError(
                    f"Expected bool for show_legend, got {type(value)} instead."
                )
            show_legend = value
        else:
            pass  # ignore unknown kwargs

    # Default tether object (if not provided)
    if tether is None:
        tether = env.tether_configuration

    # Create figure object with the plot_world function
    fig, ax = plot_env(env, settings, **kwargs)

    # Plot the tether configuration
    plot_line(
        tether,
        ax=ax,
        color=PlotColors.tether_color,
        linewidth=1.5,
        add_points=False,
        zorder=8,
    )
    tether_handle = Line2D(
        [], [], color=PlotColors.tether_color, lw=1.5, label="Tether"
    )

    # Plot robot
    if show_robot:
        (robot_handle,) = plt.plot(
            env.robot_initial_pos[0],
            env.robot_initial_pos[1],
            marker="o",
            markersize=6,
            markerfacecolor=PlotColors.robot_color,
            alpha=1,
            markeredgecolor=PlotColors.robot_edge_color,
            markeredgewidth=1,
            zorder=10,
            label="Robot",
            linestyle="None",
        )
        ax.text(
            env.robot_initial_pos[0] - 0.5,
            env.robot_initial_pos[1] - 0.5,
            r"$x_\mathrm{r}$",  # latex mathmode
            fontsize=10,
        )

    # Update legend with additional handles and labels
    if show_legend:
        handles = ax.get_legend().legend_handles
        handles.append(tether_handle)
        if show_robot:
            handles.append(robot_handle)
        ax.legend(handles=handles, **legend_settings)

    # Return fig and ax objects
    return fig, ax


def plot_curves(
    env: Env2D, settings: Settings, curves: list[LineString | list], **kwargs
) -> tuple[plt.Figure, plt.Axes]:
    """
    Plot the environment and a number of curves in it.

    Args:
        env (Env2D): World object to plot.
        settings (Settings): Settings object with the plot settings.
        curves (list[Linestring | np.ndarray]): List of curves to plot.

    Kwargs:
        show_legend (bool): Display the legend
        show_points (bool): Display the points of the curves
        label_curves (bool): Display the curve labels
        **kwargs: The function accepts all the kwargs of the plot_env function.

    Returns:
        tuple[plt.Figure, plt.Axes]: Figure and Axes objects.

    Raises:
        ValueError: If any of the kwargs are not of the expected type.
    """

    # Kwargs default values
    show_legend: bool = settings.plot.show_legend  # display the legend
    show_points: bool = False  # display the points along the curve
    label_curves: bool = True  # display the curve labels

    # Parse kwargs
    for key, value in kwargs.items():
        if key == "show_legend":
            if not isinstance(value, bool):
                raise ValueError(
                    f"Expected bool for show_legend, got {type(value)} instead."
                )
            show_legend = value
        elif key == "show_points":
            if not isinstance(value, bool):
                raise ValueError(
                    f"Expected bool for show_points, got {type(value)} instead."
                )
            show_points = value
        elif key == "label_curves":
            if not isinstance(value, bool):
                raise ValueError(
                    f"Expected bool for label_curves, got {type(value)} instead."
                )
            label_curves = value
        else:
            pass

    # Create figure object with the plot_world function
    fig, ax = plot_env(env, settings, **kwargs)

    # Plot curves
    cmap = PlotColors.other_curves_cmap  # colormap for curves
    n = PlotColors.other_curves_n  # number of colors in the cmap
    for idx, curve in enumerate(curves):
        if not isinstance(curve, LineString):
            curve = LineString(curve)
        curve_points = curve.coords
        plot_line(
            curve,
            ax=ax,
            color=cmap[idx % n],
            linewidth=1.5,
            add_points=show_points,
            zorder=8,
        )
        if label_curves:
            ax.text(
                curve_points[0][0] - 0.5,
                curve_points[0][1] - 0.5,
                rf"$\gamma_\mathrm{idx}$",  # latex mathmode
                fontsize=10,
            )
    curves_handle = Line2D(
        [], [], color=PlotColors.tether_color, lw=1.5, label="Curves"
    )

    # Update legend with additional handles and labels
    if show_legend:
        handles = ax.get_legend().legend_handles
        handles.append(curves_handle)
        ax.legend(handles=handles, **legend_settings)

    # Return fig and ax objects
    return fig, ax


def plot_graph(
    env: Env2D, graph: nx.Graph, settings: Settings, **kwargs
) -> tuple[plt.Figure, plt.Axes]:
    """
    Plot the environment and the graph generated by the planner.

    Args:
        env (Env2D): Env object to plot
        graph (nx.Graph): Graph object to plot
        settings (Settings): Settings object with the plot settings

    Kwargs:
        show_legend (bool): Display the legend
        show_node_labels (bool): Display the node labels
        **kwargs: The function accepts all the kwargs of the plot_env function.

    Returns
        tuple[plt.Figure, plt.Axes]: Figure and Axes objects

    Raises:
        ValueError: If any of the kwargs are not of the expected type.
    """
    # Kwargs default values
    show_legend: bool = settings.plot.show_legend  # display the legend
    show_node_labels: bool = False  # display the node labels

    # Parse kwargs
    for key, value in kwargs.items():
        if key == "show_legend":
            if not isinstance(value, bool):
                raise ValueError(
                    f"Expected bool for show_legend, got {type(value)} instead."
                )
            show_legend = value
        elif key == "show_node_labels":
            if not isinstance(value, bool):
                raise ValueError(
                    f"Expected bool for show_node_labels, got {type(value)} instead."
                )
            show_node_labels = value
        else:
            pass  # ignore unknown kwargs

    # Create figure object starting from the plot_tether function
    fig, ax = plot_tether(env, settings, **kwargs)

    # Extract nodes information from graph
    n_nodes = graph.number_of_nodes()
    pos = dict.fromkeys(range(0, n_nodes))
    for idx in range(0, n_nodes):
        pos[idx] = graph.nodes[idx]["pos"]

    # Plot graph
    node_color = np.array(PlotColors.node_color).reshape(1, -1)
    edge_color = np.array(PlotColors.edge_color).reshape(1, -1)
    nodes = nx.draw_networkx_nodes(
        graph,
        pos,
        node_size=4,
        node_color=node_color,
        alpha=1,
        label="Nodes",
    )
    edges = nx.draw_networkx_edges(
        graph,
        pos,
        width=0.8,
        edge_color=edge_color,
        alpha=1,
        label="Edges",
    )
    if show_node_labels:
        nx.draw_networkx_labels(graph, pos, font_size=4)

    # Add handles and labels to legend
    if show_legend:
        handles = ax.get_legend().legend_handles
        handles.extend([nodes, edges])
        ax.legend(handles=handles, **legend_settings)

    # Return fig and ax objects
    return fig, ax


def plot_free_space(
    free_space: np.ndarray[bool], delta: float
) -> tuple[plt.Figure, plt.Axes]:
    """
    Visualizes the free-space diagram from a free-space matrix.

    Args:
        free_space (np.ndarray[bool]): Free-space matrix.
        delta (float): Leash length.

    Returns:
       tuple[plt.Figure, plt.Axes]: Figure and axes objects.
    """
    # Transpose the matrix to visualize x and y correctly
    free_space = free_space.T

    # Define the colormap
    cmap = ListedColormap(["dimgrey", "white"])

    # Generate plot
    fig = plt.figure(figsize=(8, 8))
    ax = plt.gca()
    plt.imshow(
        free_space,
        cmap=cmap,
        vmin=0,
        vmax=1,
        origin="lower",
        extent=[0, free_space.shape[1], 0, free_space.shape[0]],
        aspect="equal",
    )
    ax.set_xlabel("Nodes of curve 1")
    ax.set_ylabel("Nodes of curve 2")
    ax.set_xticks(np.arange(0, free_space.shape[1] + 1, 1))
    ax.set_yticks(np.arange(0, free_space.shape[0] + 1, 1))
    ax.set_xticklabels(list(range(free_space.shape[1] + 1)))
    ax.set_yticklabels(list(range(free_space.shape[0] + 1)))
    ax.grid(color="black", linestyle="--", linewidth=0.5, alpha=0.5)
    ax.set_title(rf"Free-Space Diagram ($\delta$ = {delta:.2f})")

    # Return fig and ax objects
    return fig, ax
