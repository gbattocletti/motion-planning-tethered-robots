# pylint: disable=too-many-lines

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from shapely.geometry import LineString, MultiLineString, MultiPolygon, Point, Polygon
from shapely.plotting import plot_line, plot_points, plot_polygon

from ..utils import curves as curves_fcns
from ..utils.colors import PlotColors

if TYPE_CHECKING:
    from mpl_toolkits.mplot3d import Axes3D

    from ..env.env_2d import Env2D
    from ..env.triangulation import Triangulation
    from .settings import Settings

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
        show_generators (bool, **kwargs): flag to display the generators
        show_generators_labels (bool, **kwargs): add labels to the generators

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
    show_legend: bool = settings.plot.show_legend  # display the legend
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
        elif key == "show_legend":
            if not isinstance(value, bool):
                raise TypeError(
                    f"Expected bool for show_legend, got {type(value)} instead."
                )
            show_legend = value
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
        else:
            raise ValueError(f"Unknown kwarg: {key}")

    # Create figure object
    fig = plt.figure(figsize=settings.plot.figsize)
    ax = plt.gca()
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
    ax.set_xlabel(settings.plot.x_label, rotation=0)
    ax.set_ylabel(settings.plot.y_label, rotation=0)

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
    if show_generators:
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

            # Label the generators
            if show_generators_labels:
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

    # Plot tether configuration
    if show_tether:
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
                zorder=8,
            )
            if show_points_labels:
                ax.text(
                    point.coords[0][0] - 0.5,
                    point.coords[0][1] - 0.5,
                    rf"$\gamma_\mathrm{idx}$",  # latex mathmode
                    fontsize=10,
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
    settings: Settings,
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
        settings (Settings): Settings object with the plot settings

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
        settings,
        **kwargs,
    )

    # Plot primary graph
    for n in nodes:
        plot_points(
            Point(n),
            ax=ax,
            color=PlotColors.node_color,
            markeredgecolor=PlotColors.node_color,
            markersize=6,
            marker=".",
            zorder=8,
        )
    for e in edges:
        plot_line(
            LineString([nodes[int(e[0])], nodes[int(e[1])]]),
            ax=ax,
            color=PlotColors.edge_color,
            linewidth=1,
            add_points=False,
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
                markersize=6,
                marker=".",
                zorder=8,
            )
        for e in edges_dual:
            plot_line(
                LineString([nodes_dual[int(e[0])], nodes_dual[int(e[1])]]),
                ax=ax,
                color=PlotColors.edge_dual_color,
                linewidth=1,
                add_points=False,
                zorder=7,
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


def plot_lifted_triangulation(
    triangulation: Triangulation,
    env: Env2D,
    **kwargs,
) -> tuple[plt.Figure, plt.Axes]:
    """
    Plot the lifted triangulation.

    Args:
        triangulation (Triangulation): Triangulation object to plot
        env (Env2D): Env object to plot
        settings (Settings): Settings object with the plot settings. Note: the plot
        colors depend on the PlotColors class and not on the Settings object.

    Kwargs:
        connect_layers (bool): shared edges of triangles plotted on different layers
            are connected with vertical lines. Default is False.
        multi_layer_triangles (bool): triangles can span multiple layers. This means
            that the homotopy class of the triangle is not uniquely defined, and is
            instead evaluated for each vertex separately. Default is False. This option
            cannot be used at the same time of connect_layers and has precedence over it
        custom_sign_order (list[list[int]] | None, optional): custom order in which to
            plot the layers corresponding to the different signatures. This argument
            allows for custom tailoring of the signature order and is intended to be
            used only for plotting specific examples with improved visualization.
        layers_colormap (list[str] | None, optional): colormap to use for the layers
            corresponding to the different signatures. The length must match the number
            of unique signatures in the triangulation, or be one single color to be
            used for all layers. If None (default) a default colormap will be used.

    Returns:
        tuple[plt.Figure, plt.Axes]: Figure and Axes objects

    Raises:
        TypeError: If any of the kwargs are not of the expected type.
        ValueError: If any of the kwargs are not recognized or not consistent.
    """
    # Default kwarg values
    connect_layers: bool = False  # connect the layers with vertical lines
    multi_layer_triangles: bool = False  # triangles can span multiple layers
    custom_sign_order: list[list[int]] | None = None  # custom order for signatures
    layers_colormap: list[str] | None = None  # colormap for the layers

    # Parse Kwargs
    for key in kwargs:
        if key == "connect_layers":
            if not isinstance(kwargs["connect_layers"], bool):
                raise TypeError(
                    "Expected bool for connect_layers, "
                    f"got {type(kwargs['connect_layers'])} instead."
                )
            connect_layers = kwargs["connect_layers"]
        elif key == "multi_layer_triangles":
            if not isinstance(kwargs["multi_layer_triangles"], bool):
                raise TypeError(
                    "Expected bool for multi_layer_triangles, "
                    f"got {type(kwargs['multi_layer_triangles'])} instead."
                )
            multi_layer_triangles = kwargs["multi_layer_triangles"]
        elif key == "custom_sign_order":
            if not isinstance(kwargs["custom_sign_order"], list):
                raise TypeError(
                    "Expected list for custom_sign_order, "
                    f"got {type(kwargs['custom_sign_order'])} instead."
                )
            custom_sign_order = kwargs["custom_sign_order"]
        elif key == "layers_colormap":
            if not isinstance(kwargs["layers_colormap"], (list, type(None))):
                raise TypeError(
                    "Expected plt.Colormap, list or None for layers_colormap, "
                    f"got {type(kwargs['layers_colormap'])} instead."
                )
            layers_colormap = kwargs["layers_colormap"]
        else:
            pass  # ignore other kwargs
    if multi_layer_triangles is True:
        if connect_layers is True:
            print(
                "[PLOT] Warning: multi_layer_triangles is True but connect_layers "
                "is also True. Both cannot be True at the same time so "
                "connect_layers will be set to False."
            )
        connect_layers = False  # override connect_layers (cannot be used toghether)

    ### PREPROCESSING ###
    # Find all unique signatures
    sign_list: list[tuple] = [tuple(tri[1]) for tri in triangulation.triangles_lift]
    sign_set: set[tuple] = {*sign_list}  # get unique signatures
    unique_sign_list: list[list] = [
        list(s) for s in sign_set
    ]  # convert signatures back to lists
    unique_sign_list.sort()  # sort the signatures
    n_sign = len(unique_sign_list)  # number of unique signatures

    # Specify custom order for the signature layers
    if custom_sign_order is not None:
        if len(custom_sign_order) != n_sign:
            raise ValueError(
                f"The length of custom_sign_order {len(custom_sign_order)} does not "
                f"match the number of unique signatures {n_sign} in the triangulation."
            )
        for sign in unique_sign_list:
            if sign not in custom_sign_order:
                raise ValueError(
                    f"The signature {sign} is not present in custom_sign_order"
                )
        unique_sign_list = custom_sign_order  # override the signature order

    # Define colormap for the triangles (organized by layers)
    triangles_colors: list[str] | str
    if layers_colormap is not None:
        if len(layers_colormap) != n_sign and len(layers_colormap) != 1:
            raise ValueError(
                "The length of layers_colormap must be either 1 (single color for "
                f"all layers) or match the number of unique signatures {n_sign} in "
                "the triangulation."
            )
        if len(layers_colormap) == 1:
            triangles_colors = layers_colormap[0]
        else:
            triangles_colors = []

    ### GENERATE FIGURE ###
    # Initialize 3d axes
    fig: plt.Figure = plt.figure(figsize=(8, 8))
    ax: Axes3D = fig.add_subplot(projection="3d")

    # Set limits and aspect
    ax.set_xlim(0, env.size[0])
    ax.set_ylim(0, env.size[1])
    ax.set_zlim(0, n_sign)
    ax.set_aspect("equalxy")

    # Set view angle
    ax.view_init(elev=30, azim=45, roll=15)

    # Plot layers and label them by h signature
    layer_list: list[np.ndarray] = []
    for layer_idx in range(n_sign):

        # Define the rectangle of the layer
        padding: float = 0  # extend layer beyond env limits for better visualization
        layer = np.array(
            [
                [0 - padding, 0 - padding, layer_idx],
                [env.size[0] + padding, 0 - padding, layer_idx],
                [env.size[0] + padding, env.size[1] + padding, layer_idx],
                [0 - padding, env.size[1] + padding, layer_idx],
                [0 - padding, 0 - padding, layer_idx],  # repeat to close rectangle
            ]
        )
        layer_list.append(layer)

    # Plot triangles in layer
    layer_list = np.array(layer_list)
    ax.add_collection3d(
        Poly3DCollection(
            layer_list,
            facecolors="lightgrey",
            edgecolors="black",
            alpha=0.2,
        )
    )

    # PLOT LIFTED TRIANGULATION
    # Plot each layer of the lifted triangulation
    triangles_3d_list: list[np.ndarray] = []  # list of triangle vertices in 3D
    for layer_idx, sign in enumerate(unique_sign_list):

        # Select triangles with the same signature and plot them on the same level
        triangle_idx_list: list[int] = [
            tri[0] for tri in triangulation.triangles_lift if tri[1] == sign
        ]

        # Define triangle and add it to list
        for triangle_idx in triangle_idx_list:

            # get indexes to triangle vertices (coords are in triangulation.vertices)
            vertices_idx: np.ndarray = triangulation.triangles[triangle_idx]
            vertices_idx = vertices_idx.astype(int)  # ensure right data type
            v1: np.ndarray = triangulation.vertices[vertices_idx[0], :]
            v2: np.ndarray = triangulation.vertices[vertices_idx[1], :]
            v3: np.ndarray = triangulation.vertices[vertices_idx[2], :]

            # Build triangle by collecting [x, y, z] coordinates and add to list
            if multi_layer_triangles is True:

                # Find signature index for each vertex separately. The signature of a
                # vertex is defined as the signature of the centroid (signature of the
                # triangle) plus the signature of the path from the centroid to the
                # vertex. The signature is simplified and then the corresponding index
                # from the unique_sign_list is found. This allows for triangles that
                # span multiple layers.
                sign_1 = curves_fcns.simplify_signature(
                    sign
                    + curves_fcns.compute_signature(
                        np.array([triangulation.vertices_dual[triangle_idx], v1]),
                        env,
                        simplify=False,
                    )
                )
                sign_2 = curves_fcns.simplify_signature(
                    sign
                    + curves_fcns.compute_signature(
                        np.array([triangulation.vertices_dual[triangle_idx], v2]),
                        env,
                        simplify=False,
                    )
                )
                sign_3 = curves_fcns.simplify_signature(
                    sign
                    + curves_fcns.compute_signature(
                        np.array([triangulation.vertices_dual[triangle_idx], v3]),
                        env,
                        simplify=False,
                    )
                )
                layer_idx_1: int = unique_sign_list.index(list(sign_1))
                layer_idx_2: int = unique_sign_list.index(list(sign_2))
                layer_idx_3: int = unique_sign_list.index(list(sign_3))
                print([layer_idx_1, layer_idx_2, layer_idx_3])

                # use signature index for z coordinate of each vertex
                triangles_3d_list.append(
                    np.array(
                        [
                            [v1[0], v1[1], layer_idx_1],
                            [v2[0], v2[1], layer_idx_2],
                            [v3[0], v3[1], layer_idx_3],
                            [v1[0], v1[1], layer_idx_1],
                        ]
                    )
                )

            else:

                # use layer index for z coordinate of all vertices
                triangles_3d_list.append(
                    np.array(
                        [
                            [v1[0], v1[1], layer_idx],
                            [v2[0], v2[1], layer_idx],
                            [v3[0], v3[1], layer_idx],
                            [v1[0], v1[1], layer_idx],
                        ]
                    )
                )

            # Add color for the triangle to the color list
            if layers_colormap is not None:
                triangles_colors.append(layers_colormap[layer_idx])

    # Plot triangles
    triangles_3d_list = np.array(triangles_3d_list)
    ax.add_collection3d(
        Poly3DCollection(
            triangles_3d_list,
            facecolors=triangles_colors,
            edgecolors="black",
            alpha=0.7,
        )
    )

    # Add connections between layers (optional, alternative to multi_layer_triangles)
    if connect_layers:

        # Collect all the rectangles
        rect_list: list[np.ndarray] = []  # list of vertical connecting rectangles
        for edge in triangulation.edges_lift:

            # check if signatures of centroids are different
            if (
                triangulation.triangles_lift[edge[0]][1]
                != triangulation.triangles_lift[edge[1]][1]
            ):
                # get edge vertices (common vertices between the two triangles)
                triang_1 = triangulation.triangles_lift[int(edge[0])]  # 2d triangle idx
                triang_2 = triangulation.triangles_lift[int(edge[1])]
                vert_idx_1 = triangulation.triangles[triang_1[0]]  # vertices indexes
                vert_idx_2 = triangulation.triangles[triang_2[0]]
                edge_idx = np.intersect1d(vert_idx_1, vert_idx_2)  # common indexes
                if len(edge_idx) != 2:
                    continue  # TEMP for debugging (should not happen)
                v1 = triangulation.vertices[edge_idx[0], :]  # edge vertex coordinates
                v2 = triangulation.vertices[edge_idx[1], :]

                # Find the layer indexes of the two triangles sharing the edge
                layer_idx_1 = unique_sign_list.index(
                    list(triangulation.triangles_lift[edge[0]][1])
                )
                layer_idx_2 = unique_sign_list.index(
                    list(triangulation.triangles_lift[edge[1]][1])
                )

                # Build connecting rectangle between layers
                rect_list.append(
                    np.array(
                        [
                            [v1[0], v1[1], layer_idx_1],
                            [v1[0], v1[1], layer_idx_2],
                            [v2[0], v2[1], layer_idx_2],
                            [v2[0], v2[1], layer_idx_1],
                            [v1[0], v1[1], layer_idx_1],  # repeat to close rectangle
                        ]
                    )
                )

        # Plot rectangles between layers
        rect_list = np.array(rect_list)
        ax.add_collection3d(
            Poly3DCollection(
                rect_list,
                facecolors="black",
                edgecolors="black",
                alpha=0.4,
            )
        )

    # Add title and labels
    ax.set_title("Lifted Triangulation")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("$h$")

    # Label z ticks with the signature
    zticks = np.arange(0, n_sign, 1)
    zticklabels: list[str] = []
    for s in unique_sign_list:
        parts = []
        for i in s:
            char = f"\\sigma_{{{abs(i)}}}"
            if i < 0:
                char += "^{-1}"
            parts.append(char)
        if not parts:
            word = "`` ''"  # empty signature
        else:
            word = "``$" + "".join(parts) + "$''"  # latex mathmode
        zticklabels.append(word)
    ax.set_zticks(zticks)
    ax.set_zticklabels(zticklabels)

    # Add legend
    # TODO: implement legend + add kwarg to enable/disable it

    return fig, ax
