import os

import matplotlib.pyplot as plt
import pytest

from tethered_planning.env import env_2d
from tethered_planning.env.triangulation import Triangulation
from tethered_planning.utils import plot, plot_triangulation
from tethered_planning.utils.colors import CustomColors
from tethered_planning.utils.settings import Settings


@pytest.fixture(scope="module", name="plot_settings")
def fixture_plot_settings():
    SHOW_PLOT = True
    BLOCKING = True
    WAIT_TIME = 1
    return SHOW_PLOT, BLOCKING, WAIT_TIME


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


def show_plot(SHOW_PLOT, BLOCKING, WAIT_TIME):
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
def test_triangulation(env, plot_settings):

    # Unpack fixtures
    SHOW_PLOT, BLOCKING, WAIT_TIME = plot_settings

    # Create triangulation
    triang = Triangulation(env)
    triang.triangulate()

    plot.plot_env(
        env,
        show_goal=False,
        show_anchor=True,
    )

    # Generate figure
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
    )

    # Show and/or save figure
    show_plot(SHOW_PLOT, BLOCKING, WAIT_TIME)


@pytest.mark.parametrize(
    "env_name, length, n_triangs, check_distance, order, cmap",
    [
        (
            "test_env_1.yaml",
            1000.0,
            40,
            False,  # skip distance check
            [[1, 1, 1], [1, 1], [1], [], [-1], [-1, -1]],
            CustomColors.layers_cmap[0:6],
        ),
        (
            "test_env_5.yaml",
            1000.0,
            40,
            False,  # skip distance check
            [[1, 2], [2], [1], [], [-1], [-2], [-2, -1]],
            CustomColors.layers_cmap[0:7],
        ),
        (
            "test_env_5.yaml",
            50.0,
            100,
            True,
            None,
            None,
        ),
    ],
)
def test_lift_triangulation(
    env,
    length,
    n_triangs,
    check_distance,
    order,
    cmap,
    plot_settings,
):

    # Unpack fixtures
    SHOW_PLOT, BLOCKING, WAIT_TIME = plot_settings

    # Create triangulation and lift it
    triang = Triangulation(env)
    triang.set_max_dist(length)  # large value to avoid max_dist stopping criterion
    triang.set_max_triangles(n_triangs)  # secondary stopping criterion
    triang.triangulate()
    triang.lift_triangulation(check_distance=check_distance)

    # Generate env plot
    plot.plot_env(
        env,
        show_goal=False,
        show_anchor=True,
    )

    # Generate 2D plot
    plot_triangulation.plot_2d(
        triang,
        env,
        custom_sign_order=order,
        layers_colormap=cmap,
    )

    # Generate 3D plot
    plot_triangulation.plot_3d(
        triang,
        env,
        connect_layers=True,
        multi_layer_triangles=True,
        custom_sign_order=order,
        layers_colormap=cmap,
    )

    # Show and/or save figure
    show_plot(SHOW_PLOT, BLOCKING, WAIT_TIME)
