# pylint: disable=too-many-lines

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from shapely.geometry import LineString, MultiLineString, MultiPolygon, Point, Polygon
from shapely.plotting import plot_line, plot_points, plot_polygon

from tethered_planning.utils.colors import PlotColors

if TYPE_CHECKING:
    from tethered_planning.env.env_2d import Env2D

# NOTE: Plots implementation details:
# - To modify the colors of the plot, change the values in the PlotColors class
#   in the colors.py file.
# - The indexes of the generators start from 1 following the convention described in
#   the compute_signature function in the curve_fcns.py file.

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


def plot_env(env: Env2D, **kwargs) -> tuple[plt.Figure, plt.Axes]:
    """
    Plot the environment.

    Args:
        env (Env2D): Env object to plot

    Kwargs:
        show_anchor (bool, **kwargs): flag to display the anchor point
        show_robot (bool, **kwargs): flag to display the robot initial position
        show_tether (bool, **kwargs): flag to display the tether
        tether (LineString | np.ndarray, **kwargs): tether configuration to plot
        show_goal (bool, **kwargs): flag to display the goal region
        show_legend (bool, **kwargs): flag to display the legend
        show_generators (bool, **kwargs): flag to display the generators
        show_generators_labels (bool, **kwargs): add labels to the generators
        points (list[Point | np.ndarray], **kwargs): list of points to plot
        show_robot_anchor_labels (bool, **kwargs): add label to robot anchor and goal
        show_points_labels (bool, **kwargs): add labels to the points
        curves (list[LineString | np.ndarray], **kwargs): list of curves to plot
        show_curves_nodes (bool, **kwargs): flag to display the nodes of the curves
        show_curves_labels (bool, **kwargs): add labels to the curves
        polygons (list[Polygon | MultiPolygon | np.ndarray], **kwargs): list of polygons
            to plot
        show_polygons_nodes (bool, **kwargs): flag to display the nodes of the polygons
        show_polygons_labels (bool, **kwargs): add labels to the polygons
        title (str, **kwargs): plot title
        show_axes_labels (bool, **kwargs): flag to display the x and y axes labels
        target_ax (plt.Axes, **kwargs): Existing Axes object to draw the plot on. If
            None (default), a new figure and axes are created. This kwarg is useful to
            use this plot function as a subplot of a larger figure.
        figsize (np.ndarray | list[float], **kwargs): figure size in cm

    Returns:
        tuple[plt.Figure, plt.Axes]: Figure and Axes objects

    Raises:
        TypeError: If any of the kwargs are not of the expected type.
        ValueError: If any of the kwargs are not recognized.
    """
    # Kwargs default values
    show_anchor: bool = False  # show the anchor point
    show_robot: bool = False  # show the robot location
    show_tether: bool = False  # show the tether
    tether: LineString | np.ndarray | None = None  # tether configuration
    show_goal: bool = True  # show the goal region
    show_legend: bool = False  # display the legend
    show_robot_anchor_labels: bool = True  # display labels for robot anchor and goal
    show_obstacles_labels: bool = True  # label the obstacles
    show_generators: bool = True  # display the generators
    show_generators_labels: bool = True  # label the generators
    points: list[Point | np.ndarray] = []  # list of points to plot
    show_points_labels: bool = False  # label the points
    curves: list[LineString | np.ndarray] = []  # list of curves to plot
    show_curves_nodes: bool = False  # show the nodes (points) of the curves
    show_curves_labels: bool = False  # label the curves
    polygons: list[Polygon | MultiPolygon | np.ndarray] = []  # list of polygons to plot
    show_polygons_nodes: bool = False  # show the nodes (points) of the polygons
    show_polygons_labels: bool = False  # label the polygons
    title: str = ""  # plot title
    show_axes_labels: bool = True  # flag to display the x and y axes labels
    target_ax: plt.Axes | None = None  # Axes object to plot on (if None, create new)
    figsize: np.ndarray | list[float] = np.array([8, 8])  # figure size in cm

    # Parse kwargs
    for key, value in kwargs.items():
        if key == "show_anchor":
            if not isinstance(value, bool):
                raise TypeError(
                    f"Expected bool for show_anchor, got {type(value)} instead."
                )
            show_anchor = value
        elif key == "show_robot":
            if not isinstance(value, bool):
                raise TypeError(
                    f"Expected bool for show_robot, got {type(value)} instead."
                )
            show_robot = value
        elif key == "show_tether":
            if not isinstance(value, bool):
                raise TypeError(
                    f"Expected bool for show_tether, got {type(value)} instead."
                )
            show_tether = value
            if "tether" not in kwargs:
                tether = env.tether_configuration  # automatically select tether
        elif key == "tether":
            if not isinstance(value, (LineString, np.ndarray)):
                raise TypeError(
                    f"Expected LineString or np.ndarray for tether, "
                    f"got {type(value)} instead."
                )
            tether = value
            show_tether = True  # auto enable show_tether (can be overriden manually)
        elif key == "show_goal":
            if not isinstance(value, bool):
                raise TypeError(
                    f"Expected bool for show_goal, got {type(value)} instead."
                )
            show_goal = value
        elif key == "show_robot_anchor_labels":
            if not isinstance(value, bool):
                raise TypeError(
                    "Expected bool for show_robot_anchor_labels, "
                    f"got {type(value)} instead."
                )
            show_robot_anchor_labels = value
        elif key == "show_legend":
            if not isinstance(value, bool):
                raise TypeError(
                    f"Expected bool for show_legend, got {type(value)} instead."
                )
            show_legend = value
        elif key == "show_obstacles_labels":
            if not isinstance(value, bool):
                raise TypeError(
                    f"Expected bool for show_obstacles_labels, "
                    f"got {type(value)} instead."
                )
            show_obstacles_labels = value
        elif key == "show_generators":
            if not isinstance(value, bool):
                raise TypeError(
                    f"Expected bool for show_generators, got {type(value)} instead."
                )
            show_generators = value
        elif key == "show_generators_labels":
            if not isinstance(value, bool):
                raise TypeError(
                    "Expected bool for show_generators_labels, "
                    f"got {type(value)} instead."
                )
            if show_generators is False:
                print(
                    "[PLOT] Warning: show_generators_labels is True but "
                    "show_generators is False. This is inconsistent and will have no "
                    "effect."
                )
            show_generators_labels = value
        elif key == "points":
            if not isinstance(value, list):
                if isinstance(value, (Point, np.ndarray)):
                    value = [value]  # convert to list
                else:
                    raise TypeError(
                        f"Expected list for points, got {type(value)} instead."
                    )
            for i, p in enumerate(value):
                if not isinstance(p, (Point, np.ndarray)):
                    raise TypeError(
                        f"Expected Point or np.ndarray for points[{i}], "
                        f"got {type(p)} instead."
                    )
            points = value
        elif key == "show_points_labels":
            if not isinstance(value, bool):
                raise TypeError(
                    f"Expected bool for show_points_labels, got {type(value)} instead."
                )
            show_points_labels = value
        elif key == "curves":
            if not isinstance(value, list):
                if isinstance(value, (LineString, np.ndarray)):
                    value = [value]  # convert to list
                else:
                    raise TypeError(
                        f"Expected list for curves, got {type(value)} instead."
                    )
            for i, c in enumerate(value):
                if not isinstance(c, (LineString, np.ndarray)):
                    raise TypeError(
                        f"Expected LineString or np.ndarray for curves[{i}], "
                        f"got {type(c)} instead."
                    )
            curves = value
        elif key == "show_curves_nodes":
            if not isinstance(value, bool):
                raise TypeError(
                    f"Expected bool for show_curves_nodes, got {type(value)} instead."
                )
            show_curves_nodes = value
        elif key == "show_curves_labels":
            if not isinstance(value, bool):
                raise TypeError(
                    f"Expected bool for show_curves_labels, got {type(value)} instead."
                )
            show_curves_labels = value
        elif key == "polygons":
            if not isinstance(value, list):
                if isinstance(value, (Polygon, MultiPolygon, np.ndarray)):
                    value = [value]  # convert to list
                else:
                    raise TypeError(
                        f"Expected list for polygons, got {type(value)} instead."
                    )
            for i, poly in enumerate(value):
                if not isinstance(poly, (Polygon, MultiPolygon, np.ndarray)):
                    raise TypeError(
                        f"Expected Polygon, MultiPolygon or np.ndarray for "
                        f"polygons[{i}], got {type(poly)} instead."
                    )
            polygons = value
        elif key == "show_polygons_nodes":
            if not isinstance(value, bool):
                raise TypeError(
                    f"Expected bool for show_polygons_nodes, got {type(value)} instead."
                )
            show_polygons_nodes = value
        elif key == "show_polygons_labels":
            if not isinstance(value, bool):
                raise TypeError(
                    "Expected bool for show_polygons_labels, "
                    f"got {type(value)} instead."
                )
            show_polygons_labels = value
        elif key == "title":
            if not isinstance(value, str):
                raise TypeError(f"Expected str for title, got {type(value)} instead.")
            title = value
        elif key == "show_axes_labels":
            if not isinstance(value, bool):
                raise TypeError(
                    f"Expected bool for show_axes_labels, got {type(value)} instead."
                )
            show_axes_labels = value
        elif key == "target_ax":
            if not isinstance(value, mpl.axes.Axes):
                raise TypeError(
                    f"Expected plt.Axes for target_ax, got {type(value)} instead."
                )
            target_ax = value
        elif key == "figsize":
            if not isinstance(value, (np.ndarray, list)):
                raise TypeError(
                    f"Expected np.ndarray or list for figsize, got {type(value)} "
                    "instead."
                )
            if isinstance(value, list):
                value = np.array(value)
            if value.shape != (2,):
                raise ValueError(
                    f"Expected figsize to be of shape (2,), got {value.shape} instead."
                )
            figsize = value
        else:
            raise ValueError(f"Unknown kwarg: {key}")

    # Create figure and axes objects
    fig: plt.Figure
    ax: plt.Axes
    if target_ax is None:
        fig = plt.figure(figsize=figsize / 2.54)  # create new figure (convert cm to in)
        ax = fig.add_axes([0.15, 0.15, 0.75, 0.75])
    else:
        fig = target_ax.figure
        ax = target_ax

    # Set plot limits, labels, title, ticks, and grid
    ax.set_aspect("equal", "box")
    ax.set_xlim([0, env.size[0]])
    ax.set_ylim([0, env.size[1]])
    ax.grid(True, which="major", linestyle=":", color="gray", linewidth=0.5, zorder=1)
    ax.grid(True, which="minor", linestyle=":", color="gray", linewidth=0.3, zorder=1)
    ax.minorticks_on()
    ax.set_title(
        title,
        **{
            "fontsize": 12,
            "fontweight": "bold",
        },
    )
    if show_axes_labels:
        ax.set_xlabel("$x$", rotation=0)
        ax.set_ylabel("$y$", rotation=0)
    for tick in ax.xaxis.get_major_ticks():
        tick.tick1line.set_visible(False)
        tick.tick2line.set_visible(False)
        tick.label1.set_visible(False)
        tick.label2.set_visible(False)
    for tick in ax.xaxis.get_minor_ticks():
        tick.tick1line.set_visible(False)
        tick.tick2line.set_visible(False)
        tick.label1.set_visible(False)
        tick.label2.set_visible(False)
    for tick in ax.yaxis.get_major_ticks():
        tick.tick1line.set_visible(False)
        tick.tick2line.set_visible(False)
        tick.label1.set_visible(False)
        tick.label2.set_visible(False)
    for tick in ax.yaxis.get_minor_ticks():
        tick.tick1line.set_visible(False)
        tick.tick2line.set_visible(False)
        tick.label1.set_visible(False)
        tick.label2.set_visible(False)
    # ax.set_xticks([])
    # ax.set_xticks([], minor=True)
    # ax.set_yticks([])
    # ax.set_yticks([], minor=True)
    # ax.set_xticklabels([])
    # ax.set_yticklabels([])

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

    # Label the obstacles
    if show_obstacles_labels:
        for idx, obs in enumerate(env.obstacle_polygons):
            centroid = obs.representative_point()
            ax.text(
                centroid.x,
                centroid.y,
                f"$O_{{{idx+1}}}$",
                fontsize=8,
                ha="center",
                va="center",
                zorder=10,
            )

    # Plot generators
    if show_generators:
        if env.generators_list:
            plot_line(
                env.generators,
                ax=ax,
                color=PlotColors.generators_color,
                alpha=1,
                linewidth=0.5,
                add_points=False,
                zorder=3,
            )

            # Label the generators
            if show_generators_labels:
                offset_x = 0.2
                offset_y = 0.2
                if isinstance(env.generators, MultiLineString):
                    for idx, generator in enumerate(env.generators.geoms):
                        ax.text(
                            generator.coords[0][0] + offset_x,
                            generator.coords[0][1] + offset_y,
                            f"$\\sigma_{{{idx+1}}}$",  # latex mathmode
                            fontsize=8,
                            zorder=10,
                        )
                elif isinstance(env.generators, LineString):
                    ax.text(
                        env.generators.coords[0][0] + offset_x,
                        env.generators.coords[0][1] + offset_y,
                        f"$\\sigma_{1}$",  # latex mathmode
                        fontsize=8,
                        zorder=10,
                    )

        # Create a handle for the legend
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
        if show_robot_anchor_labels is True:
            ax.text(
                env.goal_region.centroid.x,
                env.goal_region.centroid.y,
                r"$\mathcal{X}_\mathrm{goal}$",  # latex mathmode
                fontsize=8,
                zorder=10,
            )

    # Plot anchor point
    if show_anchor:
        (anchor_handle,) = ax.plot(
            env.anchor_point[0],
            env.anchor_point[1],
            marker="o",
            markersize=4,
            markerfacecolor=PlotColors.anchor_color,
            alpha=1,
            markeredgecolor=PlotColors.anchor_edge_color,
            markeredgewidth=1,
            zorder=9,
            label="Anchor",
            linestyle="None",
        )
        if show_robot_anchor_labels is True:
            ax.text(
                env.anchor_point[0] + 0.3,
                env.anchor_point[1] - 0.3,
                r"$x_\mathrm{a}$",  # latex mathmode
                fontsize=8,
                zorder=10,
            )

    # Plot robot
    if show_robot:
        (robot_handle,) = ax.plot(
            env.robot_initial_pos[0],
            env.robot_initial_pos[1],
            marker="o",
            markersize=4,
            markerfacecolor=PlotColors.robot_color,
            alpha=1,
            markeredgecolor=PlotColors.robot_edge_color,
            markeredgewidth=1,
            zorder=10,
            label="Robot",
            linestyle="None",
        )
        if show_robot_anchor_labels is True:
            ax.text(
                env.robot_initial_pos[0] - 0.6,
                env.robot_initial_pos[1] - 0.6,
                r"$x_\mathrm{r}$",  # latex mathmode
                fontsize=8,
                zorder=10,
            )

    # Plot tether configuration
    if show_tether:
        if not isinstance(tether, LineString):
            tether = LineString(tether)  # ensure LineString type
        plot_line(
            tether,
            ax=ax,
            color=PlotColors.tether_color,
            linewidth=1.2,
            add_points=False,
            zorder=8,
        )
        tether_handle = Line2D(
            [], [], color=PlotColors.tether_color, lw=1.5, label="Tether"
        )
        if show_robot_anchor_labels is True:
            ax.text(
                tether.coords[7][0] + 0.1,  # env-1: 0.1
                tether.coords[7][1] - 0.4,  # env-1: 0.1
                r"$\gamma$",
                fontsize=8,
                zorder=10,
            )

    # Plot points
    if points:
        for idx, point in enumerate(points):
            if not isinstance(point, Point):
                point = Point(point)
            plot_points(
                point,
                ax=ax,
                color=PlotColors.points_color,
                markersize=6,
                marker=".",
                zorder=10,
            )
            if show_points_labels:
                ax.text(
                    point.coords[0][0] - 0.5,
                    point.coords[0][1] - 0.5,
                    rf"$\gamma_\mathrm{idx}$",  # latex mathmode
                    fontsize=6,
                    zorder=10,
                )
        points_handle = Line2D(
            [],
            [],
            marker=".",
            markersize=6,
            markerfacecolor=PlotColors.points_color,
            label="Points",
            linestyle="None",
        )

    # Add dummy points for padding (to match case with graph)
    dummy_nodes = [
        [0, 0],
        [0, env.size[1]],
        [env.size[0], 0],
        [env.size[0], env.size[1]],
    ]
    for n in dummy_nodes:
        plot_points(
            Point(n),
            ax=ax,
            color=[0, 0, 0, 0],
            markeredgecolor=[0, 0, 0, 0],
            markersize=4,
            marker=".",
            clip_on=False,  # allows overflowing the axes
            zorder=0,
        )

    # Plot curves
    if curves:
        cmap = PlotColors.curves_cmap  # colormap for curves
        n = PlotColors.curves_n  # number of colors in the cmap
        for idx, curve in enumerate(curves):
            if not isinstance(curve, LineString):
                curve = LineString(curve)
            plot_line(
                curve,
                ax=ax,
                color=cmap[idx % n],
                linewidth=1.5,
                add_points=show_curves_nodes,
                zorder=8,
            )
            if show_curves_labels:
                ax.text(
                    curve.coords[0][0] - 0.5,
                    curve.coords[0][1] - 0.5,
                    rf"$\gamma_\mathrm{idx}$",  # latex mathmode
                    fontsize=10,
                )
        curves_handle = Line2D([], [], color=cmap[0], lw=1.5, label="Curves")

    # Plot polygons
    if polygons:
        for idx, poly in enumerate(polygons):
            if not isinstance(poly, Polygon):
                poly = Polygon(poly)
            plot_line(
                poly,
                ax=ax,
                color=PlotColors.polygons_color,
                add_points=show_polygons_nodes,
                zorder=8,
            )
            if show_polygons_labels:
                ax.text(
                    poly.coords[0][0] - 0.5,
                    poly.coords[0][1] - 0.5,
                    rf"$\gamma_\mathrm{idx}$",  # latex mathmode
                    fontsize=10,
                )
        polygons_handle = Patch(
            facecolor=PlotColors.polygons_color,
            edgecolor=PlotColors.polygons_color,
            label="Polygons",
        )

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
        handles = [obs_handle]
        if show_generators:
            handles.append(generators_handle)
        if show_goal:
            handles.append(goal_handle)
        if show_anchor:
            handles.append(anchor_handle)
        if show_robot:
            handles.append(robot_handle)
        if show_tether:
            handles.append(tether_handle)
        if points:
            handles.append(points_handle)
        if curves:
            handles.append(curves_handle)
        if polygons:
            handles.append(polygons_handle)
        ax.legend(handles=handles, **legend_settings)

    return fig, ax


def plot_graph(
    nodes: np.ndarray | list[list],
    edges: np.ndarray | list[list],
    env: Env2D,
    nodes_dual: np.ndarray | list[list] = None,
    edges_dual: np.ndarray | list[list] = None,
    **kwargs,
) -> tuple[plt.Figure, plt.Axes]:
    """
    Plot the environment and a planar graph on top of it.

    Args:
        nodes (np.ndarray | list[list]): List of nodes of the graph
        edges (np.ndarray | list[list]): List of edges of the graph
        env (Env2D): Env object to plot

    Kwargs:
        nodes_dual (np.ndarray | list[list], optional): List of nodes of the dual
            graph. Defaults to None.
        edges_dual (np.ndarray | list[list], optional): List of edges of the dual
            graph. Defaults to None.
        show_dual_graph (bool, optional): Flag to enable the display of the dual
            graph. Defaults to True. Its intended use is to disable the dual graph when
            the dual graph is passed only for the triangles labeling.
        label_nodes (bool, optional): Flag to label the main graph nodes. Defaults to
            False. Mainly intended for debugging purposes.
        label_triangles (bool, optional): Flag to label the triangles (dual nodes).
        **kwargs: The function accepts all the kwargs of the plot_env function.

    Returns
        tuple[plt.Figure, plt.Axes]: Figure and Axes objects

    Raises:
        ValueError: If any of the kwargs are not of the expected type.
        TypeError: If any of the kwargs are not of the expected type.
        NotImplementedError: If label_triangles is True but the dual graph is not
            provided.
    """
    # Default kwarg values
    label_nodes = False  # label the main graph nodes
    label_triangles = False  # label the triangles (dual nodes)
    show_dual_graph = True  # enable the display of the dual graph

    # Parse kwargs
    # Iteration is done over list(keys) to allow deletion of keys during iteration
    for key in list(kwargs.keys()):
        if key == "nodes_dual":
            if not isinstance(kwargs[key], (np.ndarray, list)):
                raise TypeError(
                    f"Expected np.ndarray or list for nodes_dual, "
                    f"got {type(kwargs[key])} instead."
                )
            nodes_dual = kwargs[key]
            del kwargs["nodes_dual"]
        elif key == "edges_dual":
            if not isinstance(kwargs[key], (np.ndarray, list)):
                raise TypeError(
                    f"Expected np.ndarray or list for edges_dual, "
                    f"got {type(kwargs[key])} instead."
                )
            edges_dual = kwargs[key]
            del kwargs["edges_dual"]
        elif key == "show_dual_graph":
            if not isinstance(kwargs[key], bool):
                raise TypeError(
                    "Expected bool for show_dual_graph, "
                    f"got {type(kwargs[key])} instead."
                )
            show_dual_graph = kwargs[key]
            del kwargs["show_dual_graph"]
        elif key == "label_nodes":
            if not isinstance(kwargs[key], bool):
                raise TypeError(
                    "Expected bool for label_nodes, "
                    f"got {type(kwargs[key])} instead."
                )
            label_nodes = kwargs[key]
            del kwargs["label_nodes"]
        elif key == "label_triangles":
            if not isinstance(kwargs[key], bool):
                raise TypeError(
                    "Expected bool for label_triangles, "
                    f"got {type(kwargs[key])} instead."
                )
            label_triangles = kwargs[key]
            del kwargs["label_triangles"]
        else:
            pass  # leave other kwargs for plot_env (ValueError will be raised there)

    # Check kwargs consistency
    if label_triangles is True and nodes_dual is None:
        raise NotImplementedError(
            "The current implementation of the plot_graph function requires the "
            "dual graph nodes to be specified to be able to label the triangles."
        )

    # Default dual graph to empty lists if None (to avoid iteration errors)
    if nodes_dual is None:
        nodes_dual = []
    if edges_dual is None:
        edges_dual = []

    # Default kwargs values to forward to plot_env (if not specified by user)
    if "show_tether" not in kwargs:
        kwargs["show_tether"] = False
    if "show_robot" not in kwargs:
        kwargs["show_robot"] = False
    if "show_anchor" not in kwargs:
        kwargs["show_anchor"] = False
    if "show_goal" not in kwargs:
        kwargs["show_goal"] = False
    if "show_legend" not in kwargs:
        kwargs["show_legend"] = True

    # Initialize figure object starting from the plot_tether function
    fig, ax = plot_env(
        env,
        **kwargs,
    )

    # Plot primary graph
    for n in nodes:
        plot_points(
            Point(n),
            ax=ax,
            color=PlotColors.node_color,
            markeredgecolor=PlotColors.node_color,
            markersize=4,
            marker=".",
            clip_on=False,  # allows overflowing the axes
            zorder=8,
        )
    for e in edges:
        plot_line(
            LineString([nodes[int(e[0])], nodes[int(e[1])]]),
            ax=ax,
            color=PlotColors.edge_color,
            linewidth=1,
            add_points=False,
            clip_on=False,
            zorder=7,
        )

    # Plot dual graph
    if show_dual_graph is True:
        for n in nodes_dual:
            plot_points(
                Point(n),
                ax=ax,
                color=PlotColors.node_dual_color,
                markeredgecolor=PlotColors.node_dual_color,
                markersize=4,
                marker=".",
                clip_on=False,
                zorder=7,
            )
        for e in edges_dual:
            plot_line(
                LineString([nodes_dual[int(e[0])], nodes_dual[int(e[1])]]),
                ax=ax,
                color=PlotColors.edge_dual_color,
                linewidth=1,
                add_points=False,
                clip_on=False,
                zorder=6,
            )

    # Label the nodes of the main graph
    if label_nodes is True:
        for idx, n in enumerate(nodes):
            ax.text(
                n[0] + 0.15,
                n[1] + 0.15,
                f"${idx}$",  # latex mathmode
                fontsize=6,
            )

    # Label the triangles (dual nodes)
    if label_triangles is True:
        for idx, n in enumerate(nodes_dual):
            ax.text(
                n[0] + 0.15,
                n[1] + 0.15,
                f"${idx}$",  # latex mathmode
                fontsize=6,
            )

    # Create legend handles
    nodes_handle = Line2D(
        [],
        [],
        marker=".",
        markersize=6,
        markerfacecolor=PlotColors.node_color,
        markeredgecolor=PlotColors.node_color,
        label="Nodes",
        linestyle="None",
    )
    edges_handle = Line2D(
        [],
        [],
        color=PlotColors.edge_color,
        lw=1,
        label="Edges",
    )
    nodes_dual_handle = None  # to avoid reference before assignment error
    edges_dual_handle = None
    if nodes_dual is not None:
        nodes_dual_handle = Line2D(
            [],
            [],
            marker=".",
            markersize=6,
            markerfacecolor=PlotColors.node_dual_color,
            markeredgecolor=PlotColors.node_dual_color,
            label="Nodes (Dual)",
            linestyle="None",
        )
    if edges_dual is not None:
        edges_dual_handle = Line2D(
            [],
            [],
            color=PlotColors.edge_dual_color,
            lw=1,
            label="Edges (Dual)",
        )

    # Add handles and labels to legend
    if ax.get_legend() is not None:
        handles = ax.get_legend().legend_handles
        handles.extend([nodes_handle, edges_handle])
        if nodes_dual is not None:
            handles.append(nodes_dual_handle)
        if edges_dual is not None:
            handles.append(edges_dual_handle)
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
