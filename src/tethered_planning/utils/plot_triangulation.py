from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from shapely.plotting import plot_polygon

from tethered_planning.utils import colors
from tethered_planning.utils import curves as curves_fcns
from tethered_planning.utils import plot
from tethered_planning.utils.colors import PlotColors

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


def _get_unique_signatures(
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
    sign_list: list[tuple] = [tuple(tri[1]) for tri in triangulation.triangles_lift]

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
            raise ValueError(
                f"The length of order {len(order)} does not "
                f"match the number of unique signatures {n_sign} in the triangulation."
            )
        for sign in unique_sign_list:
            if sign not in order:
                raise ValueError(f"The signature {sign} is not present in order")
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
        else:
            pass  # ignore other kwargs

    # Function settings
    max_cols: int = 4  # max number of columns in the figure
    add_env_subplot: bool = True  # add subplot with the env at the beginning
    show_obstacles: bool = False  # show obstacles in the env subplot

    ### PREPROCESSING ###
    # Find all unique signatures
    unique_sign_list = _get_unique_signatures(triangulation, order=custom_sign_order)
    n_sign = len(unique_sign_list)  # number of unique signatures

    # Validate layers cmap
    if layers_cmap is None:
        layers_cmap = PlotColors.layers_cmap[0:n_sign]
    n_cmap: int = len(layers_cmap)

    # Define number of rows and columns in the figure
    n_rows: int = int(np.ceil(n_sign / max_cols))  # number of rows in the figure
    n_cols: int = min(n_sign, max_cols)  # number of columns in the figure

    # Check if the env subplot can be added
    add_env_subplot = bool(add_env_subplot is True and n_sign < n_cols * n_rows)

    ### GENERATE FIGURE ###
    # Initialize figure and axes
    fig: plt.Figure
    axs: np.ndarray[plt.Axes]
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(8, 8))

    # Plot the environment in the first subplot
    if add_env_subplot:
        plot.plot_env(
            env,
            show_generators=True,
            show_generators_labels=True,
            show_anchor=True,
            show_tether=False,
            show_robot=False,
            show_goal=False,
            show_legend=False,
            target_ax=axs[0, 0],
        )
        start_idx = 1  # start other plots from 1 (0 is used by env)
    else:
        start_idx = 0  # start other plots from 0 (no env plot)

    # Plot each layer of the lifted triangulation
    idx: int  # index of the subplot
    ax: plt.Axes  # individual axis objects found by iterating over the axs array
    for idx, ax in enumerate(axs.ravel()[start_idx:], start=0):

        # Check if subplot is within range of signatures
        if idx >= n_sign:
            ax.axis("off")  # hide unused subplots
            continue

        # Get signature for this layer
        sign = unique_sign_list[idx]

        # Set plot limits, labels, title, ticks, and grid
        ax.set_aspect("equal", "box")
        ax.set_xlim([0, env.size[0]])
        ax.set_ylim([0, env.size[1]])
        ax.set_xlabel("$x$", rotation=0)
        ax.set_ylabel("$y$", rotation=0)
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

        # Add title
        chars: list[str] = []  # list of characters
        word: str
        for i in sign:
            char = f"\\sigma_{{{abs(i)}}}"
            if i < 0:
                char += "^{-1}"
            chars.append(char)
        if not chars:
            word = "`` ''"  # empty signature
        else:
            word = "``$" + "".join(chars) + "$''"  # latex mathmode
        ax.set_title(
            word,
            **{
                "fontsize": 12,
                "fontweight": "bold",
            },
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
        idx_list: list[int] = [i for i, s in triangulation.triangles_lift if s == sign]
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
    layers_colormap: list[str] | None = None  # colormap for the layers
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
            layers_colormap = value
        elif key == "pov":
            if not isinstance(value, (list, type(None))):
                raise TypeError(
                    f"Expected list or None for pov, got {type(value)} instead."
                )
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
    unique_sign_list = _get_unique_signatures(triangulation, order=custom_sign_order)
    n_sign = len(unique_sign_list)  # number of unique signatures

    # Define colormap for the triangles (organized by layers)
    triangles_cmap: list[str] | str
    if layers_colormap is not None:
        if len(layers_colormap) != n_sign and len(layers_colormap) != 1:
            raise ValueError(
                "The length of layers_colormap must be either 1 (single color for "
                f"all layers) or match the number of unique signatures {n_sign} in "
                "the triangulation."
            )
        if len(layers_colormap) == 1:
            triangles_cmap = layers_colormap[0]
        else:
            triangles_cmap = []

    ### GENERATE FIGURE ###
    # Initialize 3d axes
    figsize = figsize / 2.54
    fig: plt.Figure = plt.figure(figsize=figsize)  # convert cm to in
    ax: Axes3D = fig.add_subplot(projection="3d")

    # Set limits and aspect
    ax.set_xlim(0, env.size[0])
    ax.set_ylim(0, env.size[1])
    ax.set_zlim(0, n_sign)
    ax.set_box_aspect([1, 1, 1.7])  # ax.set_aspect("equalxy")
    ax.view_init(elev=pov[0], azim=pov[1], roll=pov[2])

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
            alpha=0.0,
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
                # Find layer index for each vertex (default to sign if not found)
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

                # Add color for the triangle to the color list. If multiple indexes are
                # present (i.e., triangle spans multiple layers) the color of the
                # triangle is obtained by mixing the colors corresponding to the layers
                if layers_colormap is not None:
                    layer_idx = np.unique(
                        np.array([layer_idx_1, layer_idx_2, layer_idx_3])
                    )
                    c1 = layers_colormap[layer_idx[0]]
                    c2 = layers_colormap[layer_idx[-1]]
                    color = colors.combine_colors(c1, c2)
                    triangles_cmap.append(color)

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
                    triangles_cmap.append(layers_colormap[layer_idx])

    # Plot triangles
    triangles_3d_list = np.array(triangles_3d_list)
    if layers_colormap is None:
        triangles_cmap = "cyan"
    ax.add_collection3d(
        Poly3DCollection(
            triangles_3d_list,
            facecolors=triangles_cmap,
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
