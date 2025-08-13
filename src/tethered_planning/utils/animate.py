from __future__ import annotations

from typing import TYPE_CHECKING

import networkx as nx
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation

from . import plot

if TYPE_CHECKING:
    from ..env.env_2d import Env2D
    from .settings import Settings


def animate(frames: list[nx.Graph], env: Env2D, settings: Settings) -> FuncAnimation:
    """
    Generates an animation from a list of graphs.

    Args:
        frames (list[nx.Graph]): list of Graph objects to plot in sequence
        env (Env2D): Env2D object with the environment data. Assumes static environment.
        settings (Settings): Settings object with the animation parameters

    Returns:
        FuncAnimation: The animation object
    """

    # Initialize figure by plotting the environment
    fig, _ = plot.plot_env(env, settings)
    plt.xticks([])
    plt.yticks([])

    # Determine number of frames to generate
    # NOTE: the +2 ensures that the end graph is displayed
    n = int(len(frames) / settings.anim.speed_up_factor) + 2

    # Initialize graph
    graph = frames[0]
    n_nodes = graph.number_of_nodes()
    pos = dict.fromkeys(range(0, n_nodes))
    for idx in range(0, n_nodes):
        pos[idx] = graph.nodes[idx]["pos"]
    nodes = nx.draw_networkx_nodes(graph, pos, node_size=settings.plot.node_size)
    edges = nx.draw_networkx_edges(graph, pos, alpha=settings.plot.edge_alpha)

    # Set artists as animated to speed up animation generation
    nodes.set_animated(True)
    edges.set_animated(True)

    # Generate the animation
    anim = FuncAnimation(
        fig,
        animation_step,
        fargs=(frames, nodes, edges, settings),
        frames=n,
        interval=settings.anim.time_step,
        blit=True,
    )

    # Return the animation object for saving
    return anim


def animation_step(
    frame_idx: int,
    frames: list[nx.Graph],
    nodes: plt.PathCollection,
    edges: plt.LineCollection,
    settings: Settings,
) -> tuple[plt.PathCollection, plt.LineCollection]:
    """
    Updates the nodes and edges artists to the next animation frame.

    Args:
        frame_idx (int): current frame index
        frames (list[nx.Graph]): list of Graph objects corresponding to the anim frames
        nodes (plt.PathCollection): PathCollection object representing the graph nodes
        edges (plt.LineCollection): LineCollection object representing the graph edges
        settings (Settings): settings object containing animation parameters

    Returns:
        tuple[plt.PathCollection, plt.LineCollection]: updated nodes and edges artists
    """
    # Compute the frame index to plot
    idx = frame_idx * settings.anim.speed_up_factor  # apply speed up factor
    if idx < len(frames):
        graph = frames[idx]
    else:
        graph = frames[-1]

    # Extract relevant information from nodes
    # TODO: check if this operation can be vectorized/made more efficient
    n_nodes = graph.number_of_nodes()
    pos = dict.fromkeys(range(0, n_nodes))
    for i in range(0, n_nodes):
        pos[i] = graph.nodes[i]["pos"]

    # Draw nodes and edges
    # TODO: it would be more efficient to use artist setters
    # nodes.set_paths(pos)
    # edges.set_segments(...)
    nodes = nx.draw_networkx_nodes(graph, pos, node_size=settings.plot.node_size)
    edges = nx.draw_networkx_edges(graph, pos, alpha=settings.plot.edge_alpha)

    # Return updated artists
    return (nodes, edges)
