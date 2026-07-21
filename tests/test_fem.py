import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import pytest
from tqdm import tqdm

from tethered_planning.env import env_2d
from tethered_planning.tether.fem import TetherFEM2D
from tethered_planning.utils import plot_fem
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
    ],
)
@pytest.mark.parametrize(
    "curve",
    [np.column_stack([np.full(30, 8.0), np.linspace(8.0, 3.0, 30)])],
)
def test_free_drift_1(env: env_2d.Env2D, curve: np.ndarray):

    # Log info
    logging.info(
        f"{CmdColors.OKBLUE}[TestFEM]{CmdColors.ENDC} Running test_free_drift_1."
    )

    # Create tether object
    env.tether_state = curve
    env.anchor_point = curve[0]
    n_nodes = curve.shape[0]
    env.goal_vertices = np.array([3, 5])
    tether_fem = TetherFEM2D(
        env=env,
        n_nodes=n_nodes,
        state=curve,
        input_mode="force",
        dt=1e-4,
        medium="water",
        water_current=np.array([0.1, 0.0]),
        gravity=True,
    )

    # Define time vector
    t_end = 10.0
    n_steps: int = int(t_end / tether_fem.dt)

    # Simulate FEM with position input
    state_mat = np.zeros([n_steps + 1, n_nodes, 6])  # tether state
    state_mat[0] = tether_fem.state.copy()
    for k in tqdm(range(n_steps)):
        tether_fem.step(np.array([0, 0]))
        state_mat[k + 1, :, :] = tether_fem.state.copy()

    # Plot result
    indexes = np.linspace(0, n_steps, 10, dtype=int)
    snapshots = []
    for i in indexes:
        snapshots.append(state_mat[i, :, :2])
    plot_fem.plot_fem(
        env=env,
        tether_init=state_mat[0, :, :2],
        tether_final=state_mat[-1, :, :2],
        tether_snapshots=snapshots,
        show_plot=True,
    )


@pytest.mark.parametrize(
    "env_name",
    [
        "test_env_1.yaml",
    ],
)
@pytest.mark.parametrize(
    "curve",
    [np.column_stack([np.full(30, 4.0), np.linspace(8.0, 2.0, 30)])],
)
def test_free_drift_2(env: env_2d.Env2D, curve: np.ndarray):

    # Log info
    logging.info(
        f"{CmdColors.OKBLUE}[TestFEM]{CmdColors.ENDC} Running test_free_drift_2."
    )

    # Create tether object
    env.tether_state = curve
    env.anchor_point = curve[0]
    n_nodes = curve.shape[0]
    env.goal_vertices = np.array([3, 5])
    tether_fem = TetherFEM2D(
        env=env,
        n_nodes=n_nodes,
        state=curve,
        input_mode="force",
        dt=1e-4,
        medium="water",
        water_current=np.array([0.3, 0.0]),
        gravity=True,
    )

    # Define time vector
    t_end = 30.0
    n_steps: int = int(t_end / tether_fem.dt)

    # Simulate FEM with position input
    state_mat = np.zeros([n_steps + 1, n_nodes, 6])  # tether state
    state_mat[0] = tether_fem.state.copy()
    for k in tqdm(range(n_steps)):
        tether_fem.step(np.array([0, 0]))
        state_mat[k + 1, :, :] = tether_fem.state.copy()

    # Plot result
    indexes = np.linspace(0, n_steps, 10, dtype=int)
    snapshots = []
    for i in indexes:
        snapshots.append(state_mat[i, :, :2])
    plot_fem.plot_fem(
        env=env,
        tether_init=state_mat[0, :, :2],
        tether_final=state_mat[-1, :, :2],
        tether_snapshots=snapshots,
        show_plot=True,
    )


@pytest.mark.parametrize(
    "env_name",
    [
        "test_env_1.yaml",
    ],
)
def test_free_drift_3(env: env_2d.Env2D):

    # Log info
    logging.info(
        f"{CmdColors.OKBLUE}[TestFEM]{CmdColors.ENDC} Running " "test_drift_3."
    )

    # Define initial tether shape (serpentine)
    n_nodes = 30
    s = np.linspace(0.0, 1.0, n_nodes)
    curve = np.column_stack([8.0 + 1.0 * np.sin(4.0 * np.pi * s), 8.0 - 5.0 * s])

    # Create tether object
    env.tether_state = curve
    env.anchor_point = curve[0]
    env.goal_vertices = np.array([8, 8])
    tether_fem = TetherFEM2D(
        env=env,
        n_nodes=n_nodes,
        state=curve,
        input_mode="force",
        dt=1e-4,
        medium="water",  # no buoyancy; flow comes from wind
        wind=np.array([-0.5, -0.1]),  # no wind
        gravity=False,
    )

    # Define time vector
    t_end = 10.0
    n_steps = int(t_end / tether_fem.dt)

    # Run simulation
    state_mat = np.zeros([n_steps + 1, n_nodes, 6])
    reaction = np.zeros([n_steps + 1, 2])
    state_mat[0, :, :2] = curve
    for k in tqdm(range(n_steps)):
        tether_fem.step(np.array([0, 0]))
        state_mat[k + 1, :, :] = tether_fem.state.copy()
        reaction[k + 1, :] = -tether_fem.reaction_force_endpoint()

    # Generate plot
    # Plot result
    indexes = np.linspace(0, n_steps, 20, dtype=int)
    snapshots = []
    for i in indexes:
        snapshots.append(state_mat[i, :, :2])
    plot_fem.plot_fem(
        env=env,
        tether_init=state_mat[0, :, :2],
        tether_final=state_mat[-1, :, :2],
        tether_snapshots=snapshots,
        show_plot=True,
    )


@pytest.mark.parametrize(
    "env_name",
    [
        "test_env_1.yaml",
    ],
)
@pytest.mark.parametrize(
    "curve",
    [np.column_stack([np.full(30, 8.0), np.linspace(8.0, 3.0, 30)])],
)
def test_position_control(env: env_2d.Env2D, curve: np.ndarray):

    # Log info
    logging.info(
        f"{CmdColors.OKBLUE}[TestFEM]{CmdColors.ENDC} Running test_position_control."
    )

    # Create tether object
    env.tether_state = curve
    env.anchor_point = curve[0]
    n_nodes = curve.shape[0]
    env.goal_vertices = np.array([3, 5])
    tether_fem = TetherFEM2D(
        env=env,
        n_nodes=n_nodes,
        state=curve,
        input_mode="position",
        dt=1e-4,
        medium="water",
        water_current=np.array([0.1, 0.0]),
        gravity=True,
    )

    # Define time vector
    t_end = 3.0
    n_steps: int = int(t_end / tether_fem.dt)

    # Motion profile
    u_x = np.linspace(curve[-1, 0], curve[-1, 0] - 2, n_steps)
    u_y = np.linspace(curve[-1, 1], curve[-1, 1] + 2, n_steps)

    # Simulate FEM with position input
    state_mat = np.zeros([n_steps + 1, n_nodes, 6])  # tether state
    state_mat[0] = tether_fem.state.copy()
    reaction = np.zeros([n_steps + 1, 2])
    for k in tqdm(range(n_steps)):
        tether_fem.step(np.array([u_x[k], u_y[k]]))
        state_mat[k + 1, :, :] = tether_fem.state.copy()
        reaction[k + 1, :] = -tether_fem.reaction_force_endpoint()

    # Plot result
    indexes = np.linspace(0, n_steps, 10, dtype=int)
    snapshots = []
    for i in indexes:
        snapshots.append(state_mat[i, :, :2])
    plot_fem.plot_fem(
        env=env,
        tether_init=state_mat[0, :, :2],
        tether_final=state_mat[-1, :, :2],
        trajectory=np.column_stack([u_x, u_y]),
        tether_snapshots=snapshots,
        show_plot=True,
    )


@pytest.mark.parametrize(
    "env_name",
    [
        "test_env_1.yaml",
    ],
)
def test_position_control_serpentine(env: env_2d.Env2D):

    # Log info
    logging.info(
        f"{CmdColors.OKBLUE}[TestFEM]{CmdColors.ENDC} Running "
        "test_position_control_serpentine."
    )

    # Define initial tether shape (serpentine)
    n_nodes = 30
    s = np.linspace(0.0, 1.0, n_nodes)
    curve = np.column_stack([8.0 + 1.0 * np.sin(4.0 * np.pi * s), 8.0 - 5.0 * s])

    # Create tether object
    env.tether_state = curve
    env.anchor_point = curve[0]
    env.goal_vertices = np.array([8, 8])
    tether_fem = TetherFEM2D(
        env=env,
        n_nodes=n_nodes,
        state=curve,
        input_mode="position",
        dt=2e-4,
        medium="air",  # no buoyancy; flow comes from wind
        wind=np.zeros(2),  # no wind
        gravity=False,
    )

    # Define time vector
    t_end = 10.0
    n_steps = int(t_end / tether_fem.dt)
    t = np.arange(1, n_steps + 1) * tether_fem.dt

    # Motion profile (auto generated)
    def smoothstep(a):  # C1 ramp: zero velocity at a = 0
        a = np.clip(a, 0.0, 1.0)
        return a * a * (3.0 - 2.0 * a)

    p0 = curve[-1].copy()  # start the motion at the tip (8, 3)
    envelope = smoothstep(t / 5.0)  # grow the amplitude over the first 5 s
    amp_x, amp_y = 1.2, 1.2
    w = 2.0 * np.pi / 12.0  # x period 12 s; y at double frequency
    u_x = p0[0] + envelope * amp_x * np.sin(w * t)  # figure eight
    u_y = p0[1] + envelope * amp_y * np.sin(2.0 * w * t)

    # Run simulation
    state_mat = np.zeros([n_steps + 1, n_nodes, 6])
    reaction = np.zeros([n_steps + 1, 2])
    state_mat[0, :, :2] = curve
    for k in tqdm(range(n_steps)):
        tether_fem.step(np.array([u_x[k], u_y[k]]))
        state_mat[k + 1, :, :] = tether_fem.state.copy()
        reaction[k + 1, :] = -tether_fem.reaction_force_endpoint()

    # Generate plot
    # Plot result
    indexes = np.linspace(0, n_steps, 20, dtype=int)
    snapshots = []
    for i in indexes:
        snapshots.append(state_mat[i, :, :2])
    plot_fem.plot_fem(
        env=env,
        tether_init=state_mat[0, :, :2],
        tether_final=state_mat[-1, :, :2],
        trajectory=np.column_stack([u_x, u_y]),
        tether_snapshots=snapshots,
        show_plot=True,
    )
