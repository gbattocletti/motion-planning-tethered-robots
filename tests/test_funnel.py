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
    "env_name",
    [
        "test_env_1.yaml",
        "test_env_5.yaml",
    ],
)
def test_funnel_shortest_path(settings, env):

    # Create triangulation
    triang = Triangulation(env)
    triang.triangulate()

    # Generate figures
    plot.plot_env(
        env,
        settings,
        show_goal=False,
        show_anchor=True,
    )

    # TODO: execute shortest path algorithm
    # TODO: extract shortest path as graph to plot it

    plot.plot_graph(
        triang.vertices,
        triang.edges,
        env,
        settings,
        nodes_dual=triang.vertices_dual,
        edges_dual=triang.edges_dual,
        show_dual_graph=True,
        label_nodes=False,
        label_triangles=False,
        show_generators_labels=True,
    )

    # Show and/or save figure
    show_plot()
