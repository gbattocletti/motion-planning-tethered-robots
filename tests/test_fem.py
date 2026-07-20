import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import pytest

from tethered_planning.env import env_2d
from tethered_planning.tether.fem import TetherFEM2D
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


def test_constant_force():

    # Log info
    logging.info(
        f"{CmdColors.OKBLUE}[TestCurveFcns]{CmdColors.ENDC} Running test_entanglement."
    )

    # Create tether object
    env: env_2d.Env2D = 0  # TODO --> from fixture
    model = TetherFEM2D()  # TODO
    # TODO simulate


def test_position_control():

    # Log info
    logging.info(
        f"{CmdColors.OKBLUE}[TestCurveFcns]{CmdColors.ENDC} Running test_entanglement."
    )

    # Create tether object
    # Create tether object
    env: env_2d.Env2D = 0  # TODO --> from fixture
    model = TetherFEM2D()  # TODO
    # TODO simulate

    # Run simulation
    def robot_position(t):
        """Smooth trajectory: robot moves right while holding depth 8 m."""
        s = min(t / 10.0, 1.0)  # 10 s ramp, then hold
        s = s * s * (3 - 2 * s)  # smoothstep (starts/ends at rest)
        return np.array([4.0 * s, -10.0 + 2.0 * s])

    # Run simulation
    dt = 5e-4
    t = 0.0
    for k in range(int(20.0 / dt)):
        t += dt
        state = model.step(robot_position(t), dt=dt)
    R = model.reaction_force()  # force required AT the end node
    tether_force_on_robot = -R  # what the tether does to the robot
    tether_force_on_robot = -R  # what the tether does to the robot
