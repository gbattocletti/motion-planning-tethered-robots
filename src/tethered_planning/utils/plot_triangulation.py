import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from tethered_planning.env.env_2d import Env2D
from tethered_planning.env.triangulation import Triangulation
from tethered_planning.utils import colors
from tethered_planning.utils import curves as curves_fcns


def plot_3d(
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
            if not isinstance(kwargs["custom_sign_order"], (list, type(None))):
                raise TypeError(
                    "Expected list or None for custom_sign_order, "
                    f"got {type(kwargs['custom_sign_order'])} instead."
                )
            custom_sign_order = kwargs["custom_sign_order"]
        elif key == "layers_colormap":
            if not isinstance(kwargs["layers_colormap"], (list, type(None))):
                raise TypeError(
                    "Expected list or None for layers_colormap, "
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
                    triangles_colors.append(color)

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
    if layers_colormap is None:
        triangles_colors = "cyan"
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

    return fig, ax
