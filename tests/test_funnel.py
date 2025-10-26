import os

import matplotlib.pyplot as plt
import pytest

from tethered_planning.env import env_2d
from tethered_planning.env.triangulation import Triangulation
from tethered_planning.utils import plot
from tethered_planning.utils.settings import Settings


@pytest.fixture(name="settings")
def fixture_settings(env_name):
    os.chdir(os.path.dirname(os.path.abspath(__file__)))  # Move to script directory
    settings = Settings()
    settings.env_name = env_name
    return settings


@pytest.fixture(name="env")
def fixture_env(settings):
    env = env_2d.Env2D(settings)
    return env


def show_plot():

    # plot settings
    SHOW_PLOT = True
    BLOCKING = True
    WAIT_TIME = 1

    # function logic
    if SHOW_PLOT:
        if not BLOCKING:
            plt.show(block=BLOCKING)
            plt.pause(WAIT_TIME)
            plt.close()
        else:
            plt.show()  # wait on user to close plot and continue


@pytest.mark.parametrize(
    "env_name, path, p_init, p_end",
    [
        (
            "test_env_1.yaml",
            [1, 0, 2, 4, 6, 7],
            None,
            None,
        ),
        (
            "test_env_1.yaml",
            [1, 0, 2, 4, 6, 7, 5, 3, 1, 0, 2],  # test with loop
            None,
            None,
        ),
        (
            "test_env_1.yaml",
            [2, 0, 1, 3, 5, 7, 6, 4, 2, 0, 1],  # test with loop in reverse direction
            None,
            None,
        ),
        (
            "test_env_1.yaml",
            [0, 0],  # test with degenerate path
            None,
            None,
        ),
        (
            "test_env_5.yaml",
            [8, 10, 4, 0, 1, 5],
            None,
            None,
        ),
        (
            "test_env_5.yaml",
            [2, 3, 7, 9, 11, 13, 12],
            None,
            None,
        ),
        (
            "test_env_5.yaml",
            [2, 6, 8, 10, 4, 0, 1, 5, 7, 9, 11, 13],
            None,
            None,
        ),
        (
            "test_env_5.yaml",
            [2, 6, 8, 10, 4, 0, 1, 5, 7, 9, 11, 13, 12],
            [1, 1.5],  # custom initial point
            [8.5, 6],  # custom end point
        ),
        (
            "test_env_5.yaml",
            [7, 5, 1, 0, 4, 10, 12, 13, 11, 9, 7, 3, 2, 6],  # test with loop
            None,
            None,
        ),
        (
            "test_env_5.yaml",
            [7, 5, 1, 0, 4, 10, 12, 13, 11, 9, 7, 3, 2, 6],  # test with loop
            None,
            [2.8, 1.8],  # custom end point
        ),
        (
            "test_env_5.yaml",
            [7, 5, 1, 0, 4, 10, 12, 13, 11, 9, 7, 3, 2, 6],  # test with loop
            None,
            [1, 0.8],  # custom end point
        ),
        (
            "test_env_5.yaml",  # TODO: debug this test
            [5, 7, 3, 2],
            [3.5, 4.5],
            [1.7, 3],
        ),
        (
            "test_env_5.yaml",
            [5, 7, 3, 2, 6],
            [3.5, 4.5],
            [2, 1.5],
        ),
    ],
)
def test_homotopic_shortest_path(env, path, p_init, p_end):
    """
    Test the homotopic shortest path computation on a triangulated environment.

    Args:
        env: The 2D triangulated environment.
        path: The path to shorten.
        p_init: Initial point of the path. If none uses centroid of path[0].
        p_end: End point of the path. If none uses centroid of path[-1].
    """

    # Create triangulation
    triang = Triangulation(env)
    triang.triangulate()

    plot.plot_graph(
        triang.vertices,
        triang.edges,
        env,
        nodes_dual=triang.vertices_dual,
        edges_dual=triang.edges_dual,
        show_dual_graph=True,
        label_nodes=False,
        label_triangles=True,
        show_generators_labels=True,
        show_legend=False,
    )

    # Define nodes and edges of initial path
    path_nodes = [triang.vertices_dual[tri_idx] for tri_idx in path]
    path_nodes = [p_init] + path_nodes if p_init is not None else path_nodes
    path_nodes += [p_end] if p_end is not None else []
    path_len = len(path_nodes)
    path_edges = [(i, i + 1) for i in range(path_len - 1)]

    # Compute shortest homotopic path
    shortest_path_nodes = triang.homotopic_shortest_path(
        path,
        p_init=p_init,
        p_end=p_end,
    )
    shortest_path_len = len(shortest_path_nodes)
    shortest_path_edges = [(i, i + 1) for i in range(shortest_path_len - 1)]

    # Plot initial and shortened path (shortened is in red)
    plot.plot_graph(
        path_nodes,
        path_edges,
        env,
        nodes_dual=shortest_path_nodes,
        edges_dual=shortest_path_edges,
        show_dual_graph=True,
        label_nodes=False,
        label_triangles=False,
        show_generators_labels=True,
        show_legend=False,
    )

    # Show and/or save figure
    show_plot()


@pytest.mark.parametrize(
    "env_name, p1, s1, p2, s2",
    [
        (
            "test_env_1.yaml",
            [3, 5],
            [],
            [4.5, 9],
            [],
        ),
        (
            "test_env_1.yaml",
            [3, 5],
            [1],
            [4.5, 9],
            [-1],
        ),
        (
            "test_env_5.yaml",  # TODO: debug
            [6, 0.5],
            [],
            [1.5, 2.5],
            [-2, -1],
        ),
        (
            "test_env_5.yaml",
            [8, 8.5],
            [1, 2],
            [8, 4.5],
            [],
        ),
        (
            "test_env_5.yaml",
            [3.5, 4.5],
            [-2],
            [4.5, 3.5],
            [-2],
        ),
        (
            "test_env_5.yaml",
            [3.5, 4.5],
            [-2],
            [6, 0.5],
            [],
        ),
        (
            "test_env_5.yaml",
            [3.5, 4.5],
            [-2],
            [1.7, 3],
            [-2, -1],
        ),
        (
            "test_env_5.yaml",
            [3.5, 4.5],
            [-2],
            [2, 1.5],
            [-2, -1],
        ),
    ],
)
def test_geodesic_distance(env, p1, s1, p2, s2):
    """
    Test the geodesic distance computation on a triangulated environment.

    Args:
        env: The 2D triangulated environment.
        p1: Initial point of the path.
        s1: Signature of the initial point.
        p2: End point of the path.
        s2: Signature of end point.
    """

    # Create triangulation
    triang = Triangulation(env)
    triang.triangulate()
    triang.set_max_dist(50.0)
    triang.set_max_triangles(50)
    triang.lift_triangulation()

    plot.plot_graph(
        triang.vertices,
        triang.edges,
        env,
        nodes_dual=triang.vertices_dual,
        edges_dual=triang.edges_dual,
        show_dual_graph=False,
        label_nodes=False,
        label_triangles=True,
        show_generators_labels=True,
        show_legend=False,
    )

    # Compute geodesic distance
    distance, path = triang.geodesic_distance(p1, s1, p2, s2)
    print(f"Geodesic distance   between ({p1}, {s1}) and ({p2}, {s2}): {distance}")

    # Find nodes for plot
    path_len = len(path)
    path_edges = [(i, i + 1) for i in range(path_len - 1)]

    # Plot geodesic path
    plot.plot_graph(
        path,
        path_edges,
        env,
        show_dual_graph=False,
        label_nodes=False,
        label_triangles=False,
        show_generators_labels=True,
        show_legend=False,
    )

    # Show and/or save figure
    show_plot()
