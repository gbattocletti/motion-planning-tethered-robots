import os

import matplotlib.pyplot as plt
import pytest

from tethered_planning.env import env_2d
from tethered_planning.env.grid_graph import GridGraph
from tethered_planning.utils import plot, plot_graph
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
    "env_name, custom_sign_order, allow_boundary_overlap",
    [
        (
            "test_env_1.yaml",
            [[-1, -1], [-1], [], [1], [1, 1]],
            False,
        ),
        (
            "test_env_1.yaml",
            [
                [-1, -1, -1],
                [-1, -1],
                [-1],
                [],
                [1],
                [1, 1],
                [1, 1, 1],
                [1, 1, 1, 1],
            ],
            True,
        ),
        (
            "test_env_5.yaml",
            None,
            True,
        ),
    ],
)
def test_grid_graph(env, custom_sign_order, allow_boundary_overlap):

    # Create grid graph
    graph = GridGraph(env)
    graph.DEBUG = True  # Enable debug info
    graph.set_max_dist(20.0)
    graph.set_grid_resolution(0.5, 0.5)
    graph.build_homotopy_augmented_graph(allow_boundary_overlap=allow_boundary_overlap)

    # Visualize base environment
    plot.plot_env(
        env,
        show_goal=False,
        show_anchor=True,
    )

    # Visualize grid graph
    plot_graph.plot_3d(
        graph,
        env,
        custom_sign_order=custom_sign_order,
    )
    show_plot()
