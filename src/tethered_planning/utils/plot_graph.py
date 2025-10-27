from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from tethered_planning.utils.colors import PlotColors

if TYPE_CHECKING:
    from mpl_toolkits.mplot3d.axes3d import Axes3D

    from tethered_planning.env.env_2d import Env2D
    from tethered_planning.env.grid_graph import GridGraph


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
    graph: GridGraph,
    order: list[list[int]] | None = None,
) -> list[list[int]]:
    """
    Find all unique signatures in the graph and return them as a sorted list.
    Optionally, a custom order for the signatures can be specified.

    Args:
        graph (GridGraph): Graph object to analyze
        order (list[list[int]] | None, optional): custom order in which to
            return the unique signatures. Its length must match the number of unique
            signatures in the graph. Default is None.

    Returns:
        list[list[int]]: List of unique signatures, each represented as a list of ints

    Raises:
        ValueError: If custom_sign_order is provided and its length does not match the
            number of unique signatures in the graph, or if it contains signatures not
            present in the graph.
    """
    # Initialize list of unique signatures
    sign_list: list[tuple] = [tuple(node[1]) for node in graph.vertices_lift]

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
                f"match the number of unique signatures {n_sign} in the graph."
            )
        for sign in unique_sign_list:
            if sign not in order:
                raise ValueError(f"The signature {sign} is not present in order")
        unique_sign_list = order  # override the signature order

    # Return the list of unique signatures
    return unique_sign_list


def plot_3d(
    graph: GridGraph,
    env: Env2D,
    **kwargs,
) -> tuple[plt.Figure, plt.Axes]:
    """
    Generate a 3D plot of the homotopy augmented grid graph.

    Note: the function copies

    Args:
        graph (GridGraph): GridGraph object to plot
        env (Env2D): Env object to plot

    Kwargs:
        custom_sign_order (list[list[int]] | None, optional): custom order in which to
            plot the layers corresponding to the different signatures. This argument
            allows for custom tailoring of the signature order and is intended to be
            used only for plotting specific examples with improved visualization.
        layers_colormap (list[str] | None, optional): colormap to use for the layers
            corresponding to the different signatures. The length must match the number
            of unique signatures in the graph, or be one single color to be
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
    custom_sign_order: list[list[int]] | None = None  # custom order for signatures
    show_signatures: bool = True  # whether to show signature labels on the z axis
    cmap: list[str] | None = None  # colormap for the layers
    pov: list[float] | None = None  # point of view for the 3D plot
    figsize: np.ndarray = np.array([8, 8])  # figure size in cm

    # Parse Kwargs
    for key, value in kwargs.items():
        if key == "custom_sign_order":
            if not isinstance(value, (list, type(None))):
                raise TypeError(
                    "Expected list or None for custom_sign_order, "
                    f"got {type(value)} instead."
                )
            custom_sign_order = value
        elif key == "show_signatures":
            if not isinstance(value, bool):
                raise TypeError(
                    f"Expected bool for show_signatures, got {type(value)} instead."
                )
            show_signatures = value
        elif key == "layers_colormap":
            if not isinstance(value, (list, type(None))):
                raise TypeError(
                    "Expected list or None for layers_colormap, "
                    f"got {type(value)} instead."
                )
            cmap = value
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
    if pov is None:
        pov = [15, 35, 0]  # default value

    ### PREPROCESSING ###
    # Find all unique signatures
    unique_sign_list = _get_unique_signatures(graph, order=custom_sign_order)
    n_sign = len(unique_sign_list)  # number of unique signatures

    # Validate layers cmap
    if cmap is None:
        cmap = PlotColors.layers_cmap[0:n_sign]
    n_cmap: int = len(cmap)

    ### GENERATE FIGURE ###
    # Initialize 3d axes
    figsize = figsize / 2.54
    fig: plt.Figure = plt.figure(figsize=figsize)  # convert cm to in
    ax: Axes3D = fig.add_subplot(projection="3d")

    # Set limits and aspect
    ax.set_xlim(0, env.size[0])
    ax.set_ylim(0, env.size[1])
    ax.set_zlim(0, n_sign)
    ax.set_box_aspect([1, 1, 1.5])  # ax.set_aspect("equalxy")
    ax.view_init(elev=pov[0], azim=pov[1], roll=pov[2])

    # Plot nodes
    for node in graph.vertices_lift:
        # Extract node info
        idx = node[0]
        xy = graph.vertices[idx]
        sign = node[1]

        # Get z coordinate based on signature
        z = unique_sign_list.index(sign)

        # Plot node
        ax.scatter(
            xy[0],
            xy[1],
            z,
            color=(
                cmap[z % n_cmap]
                if isinstance(cmap, list)  # check if list of colors
                else cmap
            ),
            s=4,
            zorder=2,
        )

    # Plot edges
    for edge in graph.edges_lift:
        # Extract start and end node info
        node_1 = graph.vertices_lift[edge[0]]
        node_2 = graph.vertices_lift[edge[1]]
        xy_1 = graph.vertices[node_1[0]]
        xy_2 = graph.vertices[node_2[0]]

        # Get z coordinates based on signature
        sign_1 = node_1[1]
        sign_2 = node_2[1]
        z_1 = unique_sign_list.index(sign_1)
        z_2 = unique_sign_list.index(sign_2)

        # Define edge line in 3D
        line_3d = np.array(
            [
                [xy_1[0], xy_1[1], z_1],
                [xy_2[0], xy_2[1], z_2],
            ]
        )

        # Plot edge line
        ax.plot(
            line_3d[:, 0],
            line_3d[:, 1],
            line_3d[:, 2],
            color=PlotColors.edge_color,
            linewidth=0.5,
            zorder=1,
        )

    # Plot layers to make graph more readable
    layer_list: list[np.ndarray] = []
    for layer_idx in range(n_sign):

        # Define the layers rectangles
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

    # Plot the layers
    layer_list = np.array(layer_list)
    ax.add_collection3d(
        Poly3DCollection(
            layer_list,
            facecolors="lightgrey",
            edgecolors="black",
            alpha=0.05,
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
    if show_signatures:
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
                word = "$" + "".join(parts) + "$"  # latex mathmode (no quotes)
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
