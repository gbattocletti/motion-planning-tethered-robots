from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from mpl_toolkits.mplot3d import proj3d
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from shapely.plotting import plot_polygon

from tethered_planning.utils import colors
from tethered_planning.utils import curves as curves_fcns
from tethered_planning.utils import plot
from tethered_planning.utils.colors import CmdColors, PlotColors

if TYPE_CHECKING:
    from mpl_toolkits.mplot3d.axes3d import Axes3D

    from tethered_planning.env.env_2d import Env2D
    from tethered_planning.env.triangulation import Triangulation


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


def get_unique_signatures(
    triangulation: Triangulation,
    order: list[list[int]] | None = None,
) -> list[list[int]]:
    """
    Find all unique signatures in the triangulation and return them as a sorted list.
    Optionally, a custom order for the signatures can be specified.

    Args:
        triangulation (Triangulation): Triangulation object to analyze
        order (list[list[int]] | None, optional): custom order in which to
            return the unique signatures. Its length must match the number of unique
            signatures in the triangulation. Default is None.

    Returns:
        list[list[int]]: List of unique signatures, each represented as a list of ints

    Raises:
        ValueError: If custom_sign_order is provided and its length does not match the
            number of unique signatures in the triangulation, or if it contains
            signatures not present in the triangulation.
    """
    # TODO: move this to a method of curves (or a new signatures.py module?)

    # Generate list of signatures
    sign_list: list[tuple] = [tuple(tri[1]) for tri in triangulation.vertices_dual_lift]

    # Get unique signatures
    sign_set: set[tuple] = {*sign_list}
    unique_sign_list: list[list] = [
        list(s) for s in sign_set
    ]  # convert signatures back to lists
    unique_sign_list.sort()  # sort the signatures
    n_sign = len(unique_sign_list)  # number of unique signatures

    # If a custom order is specified, check its validity and apply it
    if order is not None:
        if len(order) != n_sign:
            print(
                f"{CmdColors.WARNING}[PLOT]{CmdColors.ENDC} The length of order "
                f"{len(order)} does not match the number of unique signatures "
                f"{n_sign} in the triangulation. Only the signatures in order will be "
                "plotted."
            )
        for sign in order:
            if sign not in unique_sign_list:
                raise ValueError(
                    f"The signature {sign} is not present in unique_sign_list."
                )
        unique_sign_list = order  # override the signature order

    # Return the list of unique signatures
    return unique_sign_list


def plot_2d(
    triangulation: Triangulation,
    env: Env2D,
    **kwargs,
) -> tuple[plt.Figure, plt.Axes]:
    """
    Plot the lifted triangulation in a layer-by-layer fashion, where each layer is
    displayed as a 2D plot.

    Args:
        triangulation (Triangulation): Triangulation object to plot
        env (Env2D): Env object to plot

    Kwargs:
        custom_sign_order (list[list[int]] | None, optional): custom order in which to
            plot the layers corresponding to the different signatures. The order is
            applied starting from the top-left corner of the figure, going row-wise.
            This argument allows for custom tailoring of the signature order and is
            intended to be used only for plotting specific examples with improved
            visualization.
        layers_cmap (list[str] | None, optional): colormap to use for the layers
            corresponding to the different signatures. The length must match the number
            of unique signatures in the triangulation, or be one single color to be
            used for all layers. If None (default) a default colormap will be used.
        max_cols (int, optional): maximum number of columns in the figure. Default is 4.
        add_env_subplot (bool, optional): whether to add a subplot with the environment
            at the beginning of the figure. Default is True.
        show_obstacles (bool, optional): whether to show obstacles in the layers
            subplots. Default is False.
        start_idx_cmap (int, optional): starting index for the colormap to use for the
            layers (to skip some colors at the beginning of the colormap). Default is 0.
        fig_size (np.ndarray | list[float], optional): figure size in cm

    Returns:
        tuple[plt.Figure, plt.Axes]: Figure and Axes objects

    Raises:
        TypeError: If any of the kwargs are not of the expected type.
        ValueError: If any of the kwargs are not recognized or not consistent.
    """
    # TODO: some of the kwargs are the same as in plot_3d, consider merging them
    #       and having a common function to parse them

    # Default kwarg values
    custom_sign_order: list[list[int]] | None = None  # custom order for signatures
    layers_cmap: list[str] | None = None  # colormap for the layers
    max_cols: int = 4  # max number of columns in the figure
    add_env_subplot: bool = True  # add subplot with the env at the beginning
    show_obstacles: bool = False  # show obstacles in the env subplot
    start_idx_cmap: int = 0  # skip some colors at the beginning of the cmap
    figsize: np.ndarray = np.array([8, 8])  # figure size in cm

    # Parse kwargs
    for key in kwargs:
        if key == "custom_sign_order":
            if not isinstance(kwargs["custom_sign_order"], (list, type(None))):
                raise TypeError(
                    "Expected list or None for custom_sign_order, "
                    f"got {type(kwargs['custom_sign_order'])} instead."
                )
            custom_sign_order = kwargs["custom_sign_order"]
        elif key == "layers_colormap":
            if not isinstance(kwargs["layers_colormap"], (list, str, type(None))):
                raise TypeError(
                    "Expected list, str, or None for layers_colormap, "
                    f"got {type(kwargs['layers_colormap'])} instead."
                )
            layers_cmap = kwargs["layers_colormap"]
        elif key == "max_cols":
            if not isinstance(kwargs["max_cols"], int):
                raise TypeError(
                    "Expected int for max_cols, "
                    f"got {type(kwargs['max_cols'])} instead."
                )
            if kwargs["max_cols"] <= 0:
                raise ValueError("max_cols must be a positive integer.")
            max_cols = kwargs["max_cols"]
        elif key == "add_env_subplot":
            if not isinstance(kwargs["add_env_subplot"], bool):
                raise TypeError(
                    "Expected bool for add_env_subplot, "
                    f"got {type(kwargs['add_env_subplot'])} instead."
                )
            add_env_subplot = kwargs["add_env_subplot"]
        elif key == "show_obstacles":
            if not isinstance(kwargs["show_obstacles"], bool):
                raise TypeError(
                    "Expected bool for show_obstacles, "
                    f"got {type(kwargs['show_obstacles'])} instead."
                )
            show_obstacles = kwargs["show_obstacles"]
        elif key == "start_idx_cmap":
            if not isinstance(kwargs["start_idx_cmap"], int):
                raise TypeError(
                    "Expected int for start_idx_cmap, "
                    f"got {type(kwargs['start_idx_cmap'])} instead."
                )
            if kwargs["start_idx_cmap"] < 0:
                raise ValueError("start_idx_cmap must be a non-negative integer.")
            start_idx_cmap = kwargs["start_idx_cmap"]
        elif key == "figsize":
            if not isinstance(kwargs["figsize"], (np.ndarray, list)):
                raise TypeError(
                    "Expected np.ndarray or list for figsize, got "
                    f"{type(kwargs['figsize'])} instead."
                )
            if isinstance(kwargs["figsize"], list):
                figsize = np.array(kwargs["figsize"])
            else:
                figsize = kwargs["figsize"]
            if figsize.shape != (2,):
                raise ValueError(
                    f"Expected figsize to have shape (2,), got {figsize.shape} instead."
                )
        else:
            print(f"{CmdColors.WARNING}[PLOT]{CmdColors.ENDC} Unknown kwarg: {key}")

    ### PREPROCESSING ###
    # Find all unique signatures
    unique_sign_list = get_unique_signatures(triangulation, order=custom_sign_order)
    n_sign = len(unique_sign_list)  # number of unique signatures
    if add_env_subplot is True:
        n_sign += 1  # account for env subplot

    # Validate layers cmap
    if layers_cmap is None:
        if start_idx_cmap + n_sign > len(PlotColors.layers_cmap):
            layers_cmap = [
                *PlotColors.layers_cmap[start_idx_cmap:n_sign],
                *PlotColors.layers_cmap[0:start_idx_cmap],
            ]
        else:
            layers_cmap = PlotColors.layers_cmap[
                start_idx_cmap : start_idx_cmap + n_sign
            ]
    n_cmap: int = len(layers_cmap)
    if n_cmap != n_sign:
        raise ValueError(
            "The length of layers_colormap must match the number of unique signatures."
        )

    # Define number of rows and columns in the figure
    n_rows: int = int(np.ceil(n_sign / max_cols))  # number of rows in the figure
    n_cols: int = min(n_sign, max_cols)  # number of columns in the figure

    ### GENERATE FIGURE ###
    # Initialize figure and axes
    fig: plt.Figure
    axs: np.ndarray[plt.Axes]
    fig, axs = plt.subplots(
        n_rows,
        n_cols,
        figsize=figsize / 2.54,
        constrained_layout=True,
    )

    # Plot the environment in the first subplot
    if add_env_subplot:
        if len(axs.shape) > 1:
            target_ax = axs[0, 0]
        else:
            target_ax = axs[0]
        plot.plot_env(
            env,
            show_generators=True,
            show_generators_labels=True,
            show_anchor=True,
            show_tether=False,
            show_robot=False,
            show_goal=False,
            show_legend=False,
            show_axes_labels=False,
            target_ax=target_ax,
        )
        ax_shift = 1  # start other plots from 1 (0 is used by env)
    else:
        ax_shift = 0  # start other plots from 0 (no env plot)

    # Plot each layer of the lifted triangulation
    idx: int  # index of the subplot
    ax: plt.Axes  # individual axis objects found by iterating over the axs array
    for idx, ax in enumerate(axs.ravel()[ax_shift:], start=0):

        # Check if subplot is within range of signatures
        if idx + ax_shift >= n_sign:
            ax.axis("off")  # hide unused subplots
            continue

        # Get signature for this layer
        sign = unique_sign_list[idx]

        # Set plot limits, labels, title, ticks, and grid
        ax.set_aspect("equal", "box")
        ax.set_xlim([0, env.size[0]])
        ax.set_ylim([0, env.size[1]])
        # ax.set_xlabel("$x$", rotation=0)
        # ax.set_ylabel("$y$", rotation=0)
        ax.grid(
            True,
            which="major",
            linestyle=":",
            color="gray",
            linewidth=0.5,
            zorder=1,
        )
        ax.grid(
            True,
            which="minor",
            linestyle=":",
            color="gray",
            linewidth=0.3,
            zorder=1,
        )
        ax.minorticks_on()
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xticklabels([])
        ax.set_yticklabels([])

        # Add title
        chars: list[str] = []  # list of characters
        word: str
        for i in sign:
            char = f"\\sigma_{{{abs(i)}}}"
            if i < 0:
                char += "^{-1}"
            chars.append(char)
        if not chars:
            word = "``\\;''"  # empty signature
            title_offset = -0.15
        else:
            word = "``$" + "".join(chars) + "$''"  # latex mathmode
            title_offset = -0.16
        ax.set_title(
            word,
            **{
                "fontsize": 8,
                "fontweight": "bold",
            },
            y=title_offset,  # title below plot
        )

        # Add obstacles (optional)
        if show_obstacles and not env.obstacle_region.is_empty:
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

        # Collect triangles with the same signature and plot them on the same level
        idx_list: list[int] = [
            i for i, s in triangulation.vertices_dual_lift if s == sign
        ]
        for triangle_idx in idx_list:
            plot_polygon(
                triangulation.triang.geoms[triangle_idx],
                ax=ax,
                color=(
                    layers_cmap[idx % n_cmap]
                    if isinstance(layers_cmap, list)  # check if list of colors
                    else layers_cmap
                ),
                alpha=0.7,
                edgecolor="black",
                linewidth=1,
                add_points=False,
                zorder=5,
            )

    # Return figure and axes objects
    return fig, axs


def plot_3d(
    triangulation: Triangulation,
    env: Env2D,
    **kwargs,
) -> tuple[plt.Figure, plt.Axes]:
    """
    Plot the lifted triangulation in 3D, where the different layers are organized along
    the vertical axis according to their signature.

    Args:
        triangulation (Triangulation): Triangulation object to plot
        env (Env2D): Env object to plot

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
        show_layer_area (bool, optional): wether to show the black rectangle bounding
            each signature layer.
        show_obstacles (bool, optional): show obstacles extruded in 3D.
        pov (list[float] | None, optional): point of view for the 3D plot expressed as
            a list of 3 angles [elevation, azimuth, roll]. Angles are expressed in deg.
            If None (default) a default point of view will be used.
        figsize (np.ndarray | list[float], **kwargs): figure size in cm


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
    layers_cmap: list[str] | None = None  # colormap for the layers
    show_layer_area: bool = True
    show_obstacles: bool = False
    pov: list[float] | None = None  # point of view for the 3D plot
    figsize: np.ndarray = np.array([8, 8])  # figure size in cm

    # Parse Kwargs
    for key, value in kwargs.items():
        if key == "connect_layers":
            if not isinstance(value, bool):
                raise TypeError(
                    "Expected bool for connect_layers, " f"got {type(value)} instead."
                )
            connect_layers = value
        elif key == "multi_layer_triangles":
            if not isinstance(value, bool):
                raise TypeError(
                    "Expected bool for multi_layer_triangles, "
                    f"got {type(value)} instead."
                )
            multi_layer_triangles = value
        elif key == "custom_sign_order":
            if not isinstance(value, (list, type(None))):
                raise TypeError(
                    "Expected list or None for custom_sign_order, "
                    f"got {type(value)} instead."
                )
            custom_sign_order = value
        elif key == "layers_colormap":
            if not isinstance(value, (list, type(None))):
                raise TypeError(
                    "Expected list or None for layers_colormap, "
                    f"got {type(value)} instead."
                )
            layers_cmap = value
        elif key == "show_layer_area":
            if not isinstance(value, bool):
                raise ValueError(
                    f"Expected bool for show_layer_area, got {type(value)} instead."
                )
            show_layer_area = value
        elif key == "show_obstacles":
            if not isinstance(value, bool):
                raise ValueError(
                    f"Expected bool for show_obstacles, got {type(value)} instead."
                )
            show_obstacles = value
        elif key == "pov":
            if not isinstance(value, (list, type(None))):
                raise TypeError(
                    f"Expected list or None for pov, got {type(value)} instead."
                )
            pov = value
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
    if multi_layer_triangles is True:
        if connect_layers is True:
            print(
                "[PLOT] Warning: multi_layer_triangles is True but connect_layers "
                "is also True. Both cannot be True at the same time so "
                "connect_layers will be set to False."
            )
        connect_layers = False  # override connect_layers (cannot be used toghether)
    if pov is None:
        pov = [15, 35, 0]  # default value

    ### PREPROCESSING ###
    # Find all unique signatures
    unique_sign_list = get_unique_signatures(triangulation, order=custom_sign_order)
    n_sign = len(unique_sign_list)  # number of unique signatures

    # Validate layers cmap
    if layers_cmap is None:
        layers_cmap = PlotColors.layers_cmap[0:n_sign]
    n_cmap: int = len(layers_cmap)

    ### GENERATE FIGURE ###
    # Initialize 3d axes
    figsize = figsize / 2.54
    fig: plt.Figure = plt.figure(figsize=figsize)  # convert cm to in
    ax: Axes3D = fig.add_subplot(projection="3d", computed_zorder=False)

    # Initialize variables for depth ordering
    artists = []  # every depth-sortable artist
    artists_points = []  # artists representative point (e.g. obstacle centroid)

    # Set limits and aspect
    ax.set_xlim(0, env.size[0])
    ax.set_ylim(0, env.size[1])
    ax.set_zlim(0, n_sign - 1)
    ax.set_box_aspect([1, 1, 1.7])  # ax.set_aspect("equalxy")
    ax.view_init(elev=pov[0], azim=pov[1], roll=pov[2])

    # Plot layers and label them by h signature
    if show_layer_area is True:
        layer_list: list[np.ndarray] = []
        for layer_idx in range(n_sign):

            # Define the rectangle of the layer
            padding: float = 0  # extend layer beyond env limits for better plotting
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

    # Plot obstacles
    if show_obstacles is True:
        for obs in env.obstacle_vertices:
            n = len(obs)
            bottom = np.column_stack([obs, np.full(n, 0)])
            top = np.column_stack([obs, np.full(n, n_sign - 1)])
            faces = [bottom, top]
            for i in range(n):
                j = (i + 1) % n
                side = [bottom[i], bottom[j], top[j], top[i]]
                faces.append(side)
            obs_faces = Poly3DCollection(
                faces,
                facecolor="gray",
                alpha=0.6,
                edgecolor="black",
            )
            artists.append(obs_faces)
            artists_points.append(
                np.array([obs[:, 0].mean(), obs[:, 1].mean(), 0.5 * (0 + n_sign)])
            )
            ax.add_collection3d(obs_faces)

    # Plot layer bounding boxes
    if show_layer_area is True:
        layer_list = np.array(layer_list)
        ax.add_collection3d(
            Poly3DCollection(
                layer_list,
                facecolors="lightgrey",
                edgecolors="black",
                alpha=0.0,
            )
        )

    # PLOT LIFTED TRIANGULATION
    # Plot each layer of the lifted triangulation
    for layer_idx, sign in enumerate(unique_sign_list):

        # Select triangles with the same signature and plot them on the same level
        triangle_idx_list: list[int] = [
            tri[0] for tri in triangulation.vertices_dual_lift if tri[1] == sign
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
                # Find layer index for each vertex
                # If sign_i is not found (i.e., layer with that signature does not
                # exist), default to the layer where the triangle centroid lies
                try:
                    layer_idx_1: int = unique_sign_list.index(list(sign_1))
                except ValueError:
                    layer_idx_1: int = unique_sign_list.index(list(sign))
                try:
                    layer_idx_2: int = unique_sign_list.index(list(sign_2))
                except ValueError:
                    layer_idx_2: int = unique_sign_list.index(list(sign))
                try:
                    layer_idx_3: int = unique_sign_list.index(list(sign_3))
                except ValueError:
                    layer_idx_3: int = unique_sign_list.index(list(sign))

                # Select for the triangle to the color list. If multiple indexes are
                # present (i.e., triangle spans multiple layers) the color of the
                # triangle is obtained by mixing the colors corresponding to the layers
                layer_idx = np.unique(np.array([layer_idx_1, layer_idx_2, layer_idx_3]))
                c1 = layers_cmap[layer_idx[0] % n_cmap]
                c2 = layers_cmap[layer_idx[-1] % n_cmap]
                color = colors.combine_colors(c1, c2)

                # use signature index for z coordinate of each vertex
                tri_pts = np.array(
                    [
                        [v1[0], v1[1], layer_idx_1],
                        [v2[0], v2[1], layer_idx_2],
                        [v3[0], v3[1], layer_idx_3],
                        [v1[0], v1[1], layer_idx_1],
                    ]
                )
                tri_artist = Poly3DCollection(
                    [tri_pts],
                    facecolors=color,
                    edgecolors="black",
                    alpha=0.7,
                )
                ax.add_collection3d(tri_artist)
                artists.append(tri_artist)
                artists_points.append(tri_pts.mean(axis=0))

            else:

                # use layer index for z coordinate of all vertices
                tri_pts = np.array(
                    [
                        [v1[0], v1[1], layer_idx],
                        [v2[0], v2[1], layer_idx],
                        [v3[0], v3[1], layer_idx],
                        [v1[0], v1[1], layer_idx],
                    ]
                )
                color = layers_cmap[layer_idx % n_cmap]
                tri_artist = Poly3DCollection(
                    [tri_pts],
                    facecolors=color,
                    edgecolors="black",
                    alpha=0.7,
                )
                ax.add_collection3d(tri_artist)
                artists.append(tri_artist)
                artists_points.append(tri_pts.mean(axis=0))

    # Add connections between layers (optional, alternative to multi_layer_triangles)
    if connect_layers:

        # Collect all the rectangles
        rect_list: list[np.ndarray] = []  # list of vertical connecting rectangles
        for edge in triangulation.edges_dual_lift:

            # check if signatures of centroids are different
            if (
                triangulation.vertices_dual_lift[edge[0]][1]
                != triangulation.vertices_dual_lift[edge[1]][1]
            ):
                # get edge vertices (common vertices between the two triangles)
                triang_1 = triangulation.vertices_dual_lift[
                    int(edge[0])
                ]  # 2d triangle idx
                triang_2 = triangulation.vertices_dual_lift[int(edge[1])]
                vert_idx_1 = triangulation.triangles[triang_1[0]]  # vertices indexes
                vert_idx_2 = triangulation.triangles[triang_2[0]]
                edge_idx = np.intersect1d(vert_idx_1, vert_idx_2)  # common indexes
                if len(edge_idx) != 2:
                    continue  # TEMP for debugging (should not happen)
                v1 = triangulation.vertices[edge_idx[0], :]  # edge vertex coordinates
                v2 = triangulation.vertices[edge_idx[1], :]

                # Find the layer indexes of the two triangles sharing the edge
                layer_idx_1 = unique_sign_list.index(
                    list(triangulation.vertices_dual_lift[edge[0]][1])
                )
                layer_idx_2 = unique_sign_list.index(
                    list(triangulation.vertices_dual_lift[edge[1]][1])
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

    # Set zorder for all artists to plot with correct depth
    artists_points = np.array(artists_points)
    _, _, depths = proj3d.proj_transform(
        artists_points[:, 0], artists_points[:, 1], artists_points[:, 2], ax.get_proj()
    )
    order = np.flip(np.argsort(depths))
    for rank, idx in enumerate(order):
        artists[idx].set_zorder(rank)

    # Add title and labels
    ax.set_title("")
    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")
    ax.set_zlabel("$h$")
    ax.xaxis.labelpad = -17  # smaller = closer
    ax.yaxis.labelpad = -17
    ax.zaxis.labelpad = 5
    ax.set_xticks([])
    ax.set_xticks([], minor=True)
    ax.set_yticks([])
    ax.set_yticks([], minor=True)
    ax.set_xticklabels([])
    ax.set_yticklabels([])

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
            # word = "``$" + "".join(parts) + "$''"  # latex mathmode
            word = "$" + "".join(parts) + "$"  # latex mathmode (no quotes on signature)
        zticklabels.append(word)
    ax.set_zticks(zticks)
    ax.set_zticklabels(zticklabels)

    # Customize grid, panes, and axes
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_alpha(0)
    ax.yaxis.pane.set_alpha(0)
    ax.zaxis.pane.set_alpha(0)
    ax.set_position([0.08, 0.04, 1, 1])  # left, bottom, width, height

    return fig, ax


def plot_3d_plotly(
    triangulation: Triangulation,
    env: Env2D,
    **kwargs,
) -> go.Figure:
    """
    Plot the lifted triangulation in 3D with plotly to ensure better rendering of the
    simplicial complex, since plotly natively manages depth rendering. Some options that
    are available in the matplotlib version of this function are currently disabled for
    ease of development.

    Args:
        triangulation (Triangulation): Triangulation object to plot
        env (Env2D): Env object to plot

    Kwargs:
        custom_sign_order (list[list[int]] | None, optional): custom order in which to
            plot the layers corresponding to the different signatures. This argument
            allows for custom tailoring of the signature order and is intended to be
            used only for plotting specific examples with improved visualization.
        layers_colormap (list[str] | None, optional): colormap to use for the layers
            corresponding to the different signatures. The length must match the number
            of unique signatures in the triangulation, or be one single color to be
            used for all layers. If None (default) a default colormap will be used.
        show_layer_area (bool, optional): wether to show the black rectangle bounding
            each signature layer.
        show_obstacles (bool, optional): show obstacles extruded in 3D.
        plot_entanglement_free_simplices (bool, optional): plot only simplices that are
            entanglement free.
        pov (list[float] | None, optional): point of view for the 3D plot expressed as
            a list of 3 angles [elevation, azimuth, roll]. Angles are expressed in deg.
            If None (default) a default point of view will be used.

    Note:
        connect_layers: set to False
        multi_layer_triangles: set to True
        fig_size: unused

    Returns:
        go.Figure: Figure and Axes objects

    Raises:
        TypeError: If any of the kwargs are not of the expected type.
        ValueError: If any of the kwargs are not recognized or not consistent.
    """
    # Default kwarg values
    custom_sign_order: list[list[int]] | None = None  # custom order for signatures
    layers_cmap: list[str] | None = None  # colormap for the layers
    show_layer_area: bool = True
    show_obstacles: bool = False
    plot_entanglement_free_simplices: bool = False
    pov: list[float] | None = None  # point of view for the 3D plot

    # Parse Kwargs
    for key, value in kwargs.items():
        if key == "connect_layers":
            print("Warning: conenct_layers cannot currently be changed.")
        elif key == "multi_layer_triangles":
            print("Warning: multi_layer_triangles cannot currently be changed.")
        elif key == "custom_sign_order":
            if not isinstance(value, (list, type(None))):
                raise TypeError(
                    "Expected list or None for custom_sign_order, "
                    f"got {type(value)} instead."
                )
            custom_sign_order = value
        elif key == "layers_colormap":
            if not isinstance(value, (list, type(None))):
                raise TypeError(
                    "Expected list or None for layers_colormap, "
                    f"got {type(value)} instead."
                )
            layers_cmap = value
        elif key == "show_layer_area":
            if not isinstance(value, bool):
                raise ValueError(
                    f"Expected bool for show_layer_area, got {type(value)} instead."
                )
            show_layer_area = value
        elif key == "plot_entanglement_free_simplices":
            if not isinstance(value, bool):
                raise ValueError(
                    "Expected bool for plot_entanglement_free_simplices, "
                    f"got {type(value)} instead."
                )
            plot_entanglement_free_simplices = value
        elif key == "show_obstacles":
            if not isinstance(value, bool):
                raise ValueError(
                    f"Expected bool for show_obstacles, got {type(value)} instead."
                )
            show_obstacles = value
        elif key == "pov":
            if not isinstance(value, (list, type(None))):
                raise TypeError(
                    f"Expected list or None for pov, got {type(value)} instead."
                )
            pov = value
        elif key == "figsize":
            print("Figsize cannot be changed in plotly")
        else:
            raise ValueError(f"Unknown kwarg: {key}")

    # Find all unique signatures
    unique_sign_list = get_unique_signatures(triangulation, order=custom_sign_order)
    n_sign = len(unique_sign_list)  # number of unique signatures

    # Validate layers cmap
    # CHECKME: does this only work if n_sign <= 10?
    if layers_cmap is None:
        layers_cmap = PlotColors.layers_cmap[0:n_sign]
    n_cmap: int = len(layers_cmap)

    # Initialize collection of artists
    artists = []

    # Plot layers and label them by h signature
    if show_layer_area is True:
        for layer_idx in range(n_sign):

            # Define the rectangle of the layer
            padding: float = 0  # extend layer beyond env limits for better plotting
            layer = np.array(
                [
                    [0 - padding, 0 - padding, layer_idx],
                    [env.size[0] + padding, 0 - padding, layer_idx],
                    [env.size[0] + padding, env.size[1] + padding, layer_idx],
                    [0 - padding, env.size[1] + padding, layer_idx],
                    [0 - padding, 0 - padding, layer_idx],  # repeat to close rectangle
                ]
            )
            artists.append(
                go.Mesh3d(
                    x=layer[:, 0],
                    y=layer[:, 1],
                    z=layer[:, 2],
                    i=[0, 0],
                    j=[1, 2],
                    k=[2, 3],
                    color="lightgray",
                    opacity=0.2,
                    flatshading=True,
                )
            )
            loop = np.vstack([layer, layer[0]])  # close back to first vertex
            artists.append(
                go.Scatter3d(
                    x=loop[:, 0],
                    y=loop[:, 1],
                    z=loop[:, 2],
                    mode="lines",
                    line=dict(color="black", width=2),
                    showlegend=False,
                )
            )

    # Plot obstacles
    if show_obstacles is True:
        for obs in env.obstacle_vertices:
            n = len(obs)

            # Add faces of obstacles
            x_obs = np.concatenate([obs[:, 0], obs[:, 0]])
            y_obs = np.concatenate([obs[:, 1], obs[:, 1]])
            z_obs = np.concatenate([np.full(n, 0), np.full(n, n_sign - 1)])
            i, j, k = [], [], []
            for t in range(1, n - 1):  # bottom cap
                i += [0]
                j += [t]
                k += [t + 1]
            for t in range(1, n - 1):  # top cap
                i += [n]
                j += [n + t]
                k += [n + t + 1]
            for a in range(n):  # side walls
                b = (a + 1) % n
                i += [a, a]
                j += [b, n + b]
                k += [n + b, n + a]
            artists.append(
                go.Mesh3d(
                    x=x_obs,
                    y=y_obs,
                    z=z_obs,
                    i=i,
                    j=j,
                    k=k,
                    color="gray",
                    opacity=0.6,
                    flatshading=True,
                )
            )

            # Add edges of obstacles
            x_edge, y_edge, z_edge = [], [], []
            for a in list(range(n)) + [0]:  # bottom loop
                x_edge.append(obs[a, 0])
                y_edge.append(obs[a, 1])
                z_edge.append(0)
            x_edge.append(None)
            y_edge.append(None)
            z_edge.append(None)
            for a in list(range(n)) + [0]:  # top loop
                x_edge.append(obs[a, 0])
                y_edge.append(obs[a, 1])
                z_edge.append(n_sign - 1)
            x_edge.append(None)
            y_edge.append(None)
            z_edge.append(None)
            for a in range(n):  # verticals
                x_edge += [obs[a, 0], obs[a, 0], None]
                y_edge += [obs[a, 1], obs[a, 1], None]
                z_edge += [0, n_sign - 1, None]
            artists.append(
                go.Scatter3d(
                    x=x_edge,
                    y=y_edge,
                    z=z_edge,
                    mode="lines",
                    line=dict(color="black", width=2),
                    showlegend=False,
                )
            )

    # Plot each layer of the lifted triangulation
    for layer_idx, sign in enumerate(unique_sign_list):

        # Select triangles with the same signature and plot them on the same level
        triangle_idx_list: list[int, int] = [
            (i, tri[0])
            for i, tri in enumerate(triangulation.vertices_dual_lift)
            if tri[1] == sign
        ]

        # Define triangle and add it to list
        for triangle_lifted_idx, triangle_idx in triangle_idx_list:

            # Check if triangle is present in entanglement-free simplicial complex. If
            # not, skip it and move to next index.
            if (
                plot_entanglement_free_simplices is True
                and triangulation.entanglement_triangles_lift[triangle_lifted_idx]
                is False
            ):
                continue

            # get indexes to triangle vertices (coords are in triangulation.vertices)
            vertices_idx: np.ndarray = triangulation.triangles[triangle_idx]
            vertices_idx = vertices_idx.astype(int)  # ensure right data type
            v1: np.ndarray = triangulation.vertices[vertices_idx[0], :]
            v2: np.ndarray = triangulation.vertices[vertices_idx[1], :]
            v3: np.ndarray = triangulation.vertices[vertices_idx[2], :]

            # Build triangle by collecting [x, y, z] coordinates and add to list.
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
            # Find layer index for each vertex
            # If sign_i is not found (i.e., layer with that signature does not
            # exist), default to the layer where the triangle centroid lies
            try:
                layer_idx_1: int = unique_sign_list.index(list(sign_1))
            except ValueError:
                layer_idx_1: int = unique_sign_list.index(list(sign))
            try:
                layer_idx_2: int = unique_sign_list.index(list(sign_2))
            except ValueError:
                layer_idx_2: int = unique_sign_list.index(list(sign))
            try:
                layer_idx_3: int = unique_sign_list.index(list(sign_3))
            except ValueError:
                layer_idx_3: int = unique_sign_list.index(list(sign))

            # Select for the triangle to the color list. If multiple indexes are
            # present (i.e., triangle spans multiple layers) the color of the
            # triangle is obtained by mixing the colors corresponding to the layers
            layer_idx = np.unique(np.array([layer_idx_1, layer_idx_2, layer_idx_3]))
            c1 = layers_cmap[layer_idx[0] % n_cmap]
            c2 = layers_cmap[layer_idx[-1] % n_cmap]
            color = colors.combine_colors(c1, c2)

            # Add triangle face
            tri_pts = np.array(
                [
                    [v1[0], v1[1], layer_idx_1],  # signature index for z coordinate
                    [v2[0], v2[1], layer_idx_2],
                    [v3[0], v3[1], layer_idx_3],
                    [v1[0], v1[1], layer_idx_1],
                ]
            )
            artists.append(
                go.Mesh3d(
                    x=tri_pts[:, 0],
                    y=tri_pts[:, 1],
                    z=tri_pts[:, 2],
                    i=[0],
                    j=[1],
                    k=[2],
                    color=color,
                    flatshading=True,
                )
            )

            # Add triangle edges
            loop = np.vstack([tri_pts, tri_pts[0]])
            artists.append(
                go.Scatter3d(
                    x=loop[:, 0],
                    y=loop[:, 1],
                    z=loop[:, 2],
                    mode="lines",
                    line=dict(color="black", width=2),
                    showlegend=False,
                )
            )

    # Convert pov to eye
    if pov is None:
        pov = [15, 35, 2]  # default value
    e = np.radians(pov[0])
    a = np.radians(pov[1])
    r = pov[2]

    # Generate figure
    fig = go.Figure(data=artists)
    fig.update_layout(
        scene=dict(
            aspectmode="manual",
            aspectratio=dict(x=1, y=1, z=0.65),
            xaxis=dict(
                range=[0, 10],
                title="",
                showticklabels=False,
                ticks="",
                showbackground=True,
                backgroundcolor="white",
                gridcolor="lightgrey",
                zeroline=True,
            ),
            yaxis=dict(
                range=[0, 10],
                title="",
                showticklabels=False,
                ticks="",
                showbackground=True,
                backgroundcolor="white",
                gridcolor="lightgrey",
                zeroline=True,
            ),
            zaxis=dict(
                range=[0, n_sign - 1],
                title="",
                showticklabels=False,
                ticks="",
                showbackground=True,
                backgroundcolor="white",
                gridcolor="lightgrey",
                zeroline=True,
            ),
            camera=dict(
                eye=dict(
                    x=r * np.cos(e) * np.cos(a),
                    y=r * np.cos(e) * np.sin(a),
                    z=r * np.sin(e),
                ),
                projection=dict(type="orthographic"),
            ),
        ),
        margin=dict(l=0, r=0, t=0, b=0),
        showlegend=False,
    )
    return fig
