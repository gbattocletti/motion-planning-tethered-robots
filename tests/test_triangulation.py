import os

import matplotlib.pyplot as plt
import pytest

from tethered_planning.env import env_2d
from tethered_planning.env.triangulation import Triangulation
from tethered_planning.utils import io, plot
from tethered_planning.utils.colors import CustomColors
from tethered_planning.utils.settings import Settings


@pytest.fixture(scope="module", name="plot_settings")
def fixture_plot_settings():
    SHOW_PLOT = True
    BLOCKING = True
    WAIT_TIME = 1
    SAVE_PLOT = False
    return SHOW_PLOT, BLOCKING, WAIT_TIME, SAVE_PLOT


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
        "test_env_2.yaml",
        "test_env_3.yaml",
        "test_env_4.yaml",
        "test_env_5.yaml",
    ],
)
def test_triangulation(settings, env, plot_settings):

    # Unpack fixtures
    SHOW_PLOT, BLOCKING, WAIT_TIME, SAVE_PLOT = plot_settings

    # Create triangulation
    triang = Triangulation(env)
    triang.triangulate()

    # Generate figure
    fig, _ = plot.plot_graph(
        triang.vertices,
        triang.edges,
        env,
        settings,
        nodes_dual=triang.vertices_dual,
        edges_dual=triang.edges_dual,
        show_dual_graph=True,
        label_nodes=True,
        label_triangles=True,
    )

    # Show and/or save figure
    show_plot(SHOW_PLOT, BLOCKING, WAIT_TIME)
    if SAVE_PLOT:
        fig_name = "env"
        io.save_figure(fig, settings, fig_name, "png")


@pytest.mark.parametrize(
    "env_name, order, cmap",
    [
        (
            "test_env_1.yaml",
            [[1, 1, 1], [1, 1], [1], [], [-1], [-1, -1]],
            CustomColors.layers_cmap[0:6],
        ),
        # ("test_env_2.yaml", None, None),
        # ("test_env_3.yaml", None, None),
        # ("test_env_4.yaml", None, None),
        # ("test_env_5.yaml", None, None),
    ],
)
def test_lift_triangulation(settings, env, order, cmap, plot_settings):

    # Unpack fixtures
    SHOW_PLOT, BLOCKING, WAIT_TIME, SAVE_PLOT = plot_settings

    # Create triangulation and lift it
    triang = Triangulation(env)
    triang.triangulate()
    triang.lift_triangulation()

    # Generate plot of lifted triangulation
    fig, _ = plot.plot_lifted_triangulation(
        triang,
        env,
        connect_layers=True,
        multi_layer_triangles=True,
        custom_sign_order=order,
        layers_colormap=cmap,
    )

    # Show and/or save figure
    show_plot(SHOW_PLOT, BLOCKING, WAIT_TIME)
    if SAVE_PLOT:
        fig_name = "lifted_triangulation"
        io.save_figure(fig, settings, fig_name, "png")
