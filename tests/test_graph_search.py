import os

import matplotlib.pyplot as plt
import pytest

from tethered_planning.env import env_2d
from tethered_planning.env.triangulation import Triangulation
from tethered_planning.plan import graph_search
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
    "env_name, idx_1, idx_2",
    [
        (
            "test_env_5.yaml",
            0,
            3,
        ),
    ],
)
def test_dijkstra(env, idx_1, idx_2):

    triang = Triangulation(env)
    triang.triangulate()
    triang.set_max_dist(1000.0)
    triang.set_max_triangles(50)
    triang.lift_triangulation(check_distance=False)

    # Compute path between (p1, s1) and (p2, s2)
    alpha_lift: list[int] = graph_search.a_star_search(
        triang.vertices_dual_lift,
        triang.edges_dual_lift,
        idx_1,  # triangle index in the lifted triangulation
        idx_2,
        h_augmented=True,
        nodes_2d=triang.vertices_dual,
        use_heuristic=False,  # dijkstra
    )

    # Project the representative path onto the 2D triangulation
    alpha: list[int] = [triang.vertices_dual_lift[idx][0] for idx in alpha_lift]

    path = [triang.vertices_dual[idx] for idx in alpha]
    path_len = len(path)
    edges = [(i, i + 1) for i in range(path_len - 1)]

    # Plot initial and shortened path (shortened is in red)
    plot.plot_graph(
        path,
        edges,
        env,
        nodes_dual=triang.vertices_dual,
        edges_dual=triang.edges_dual,
        show_dual_graph=False,
        label_nodes=False,
        label_triangles=True,
        show_generators_labels=False,
        show_legend=False,
    )

    # Show and/or save figure
    show_plot()
