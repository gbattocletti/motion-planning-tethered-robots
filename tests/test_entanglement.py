import logging
import os

import matplotlib.pyplot as plt
import pytest

from tethered_planning.env import env_2d
from tethered_planning.utils import curves, entanglement, plot
from tethered_planning.utils.colors import CmdColors
from tethered_planning.utils.settings import Settings

logger = logging.getLogger(__name__)


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
        "test_env_6.yaml",
        "test_env_7.yaml",
    ],
)
def test_entanglement(env: env_2d.Env2D, plot_settings):

    # Log info
    logging.info(
        f"{CmdColors.OKBLUE}[TestCurveFcns]{CmdColors.ENDC} Running " "test_signature."
    )

    # Unpack fixtures
    SHOW_PLOT, BLOCKING, WAIT_TIME = plot_settings

    # Create curve
    curve = curves.generate_curve(
        env,
        init_point=env.anchor_point,
        show_goal=False,
    )

    # Check entanglement and print results
    logging.info(f"Signature convex hull: {entanglement.convex_hull(curve, env)}")
    logging.info(
        f"Signature linear homotopy: {entanglement.linear_homotopy(curve, env)}"
    )
    # logging.info(f"Signature null homotopy: {entanglement.null_homotopy(curve, env)}")
    logging.info(
        "Signature local visibility homotopy: "
        f"{entanglement.local_visibility_homotopy(curve, env)}"
    )

    # Plot curve
    plot.plot_env(
        env,
        curves=[curve],
        show_anchor=True,
        show_goal=False,
        show_generators_labels=False,
    )

    # Show and/or save figure
    show_plot(SHOW_PLOT, BLOCKING, WAIT_TIME)
