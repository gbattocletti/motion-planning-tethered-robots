""" "
Perform trajectory planning and execution.
"""

import os
import pickle

import numpy as np
from scipy.interpolate import splev, splprep
from shapely import LineString
from tqdm import tqdm

from tethered_planning.env import env_2d
from tethered_planning.env.triangulation import Triangulation
from tethered_planning.tether.fem import TetherFEM2D
from tethered_planning.utils import curves, plot, plot_fem
from tethered_planning.utils.settings import Settings

# Script settings
env_name = "env_5"
length_max = 15.0
goal = np.array([7, 1])
n_nodes = 30  # nodes in FEM model
resampling: str = "linear"  # {linear, spline}
manually_select_tether: bool = False
manually_select_path: bool = True
tether_at_equilibrium: bool = True

########################################################################################
# Load data ############################################################################
########################################################################################
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Select datafile to use
datapath: str
match (env_name, length_max):
    case ("env_3b", 10):
        datapath = "results/entanglement_free_model/comparison-19.pkl"
    case ("env_3b", 12.5):
        datapath = "results/entanglement_free_model/comparison-22.pkl"
    case ("env_3b", 15.0):
        datapath = "results/entanglement_free_model/comparison-25.pkl"
    case ("env_3", 10):
        datapath = "results/entanglement_free_model/comparison-28.pkl"
    case ("env_3", 12.5):
        datapath = "results/entanglement_free_model/comparison-31.pkl"
    case ("env_3", 15.0):
        datapath = "results/entanglement_free_model/comparison-34.pkl"
    case ("env_4", 10):
        datapath = "results/entanglement_free_model/comparison-37.pkl"
    case ("env_4", 12.5):
        datapath = "results/entanglement_free_model/comparison-40.pkl"
    case ("env_4", 15.0):
        datapath = "results/entanglement_free_model/comparison-43.pkl"
    case ("env_5", 10):
        datapath = "results/entanglement_free_model/comparison-46.pkl"
    case ("env_5", 12.5):
        datapath = "results/entanglement_free_model/comparison-49.pkl"
    case ("env_5", 15.0):
        datapath = "results/entanglement_free_model/comparison-52.pkl"
    case _:
        raise ValueError("Invalid combination of env and tether length")

# Create settings and env (load along with simplicial complex model)
objects: list = []
with open(datapath, "rb") as openfile:
    while True:
        try:
            objects.append(pickle.load(openfile))
        except EOFError:
            break
data: dict = objects[0]
settings: Settings = data["settings"]
env: env_2d.Env2D = data["env"]
triang: Triangulation = data["triangulation"]
anchor: np.ndarray = env.anchor_point
settings: Settings = Settings()
settings.env_name = env_name
env: env_2d.Env2D = env_2d.Env2D(settings)
env.goal_vertices = goal


########################################################################################
# Initialize tether ####################################################################
########################################################################################

# Initialize plot to show tethers
fig, ax = plot.plot_env(
    env,
    show_tether=False,
    show_robot=True,
    show_anchor=True,
    show_goal=False,
    show_legend=False,
    show_generators=False,
    show_curves_labels=False,
    show_robot_anchor_labels=False,
    show_generators_labels=False,
    show_obstacles_labels=False,
    show_axes_labels=False,
    figsize=[8, 8],
)
ax.set_xlabel("")
ax.set_ylabel("")
ax.set_xticklabels([])
ax.set_yticklabels([])
ax.plot(goal[0], goal[1], color="green", marker="o", markersize=2, zorder=10)

# Define tether curve
tether: np.ndarray
if manually_select_tether is True:
    tether = curves.generate_curve(
        env,
        init_point=env.anchor_point,
        check_self_intersection=False,
        max_points=200,
        show_goal=True,
        show_robot_anchor_labels=False,
        show_legend=False,
        output_type="numpy",
    )
else:
    tether = np.array(
        [
            [3.0, 7.0],
            [2.85162806, 6.74118832],
            [2.74420947, 6.70090634],
            [2.4890903, 6.6740517],
            [2.16683451, 6.82175227],
            [2.13997986, 6.92917086],
            [2.20711648, 7.21114468],
            [2.27425311, 7.35884525],
            [2.40852635, 7.57368244],
            [2.65021819, 7.73481034],
            [3.01275596, 7.89593823],
            [3.22759315, 7.96307486],
            [3.44243035, 8.00335683],
            [3.65726754, 8.03021148],
            [4.08694193, 7.92279288],
            [4.19436052, 7.80194696],
            [4.46290702, 7.69452837],
            [4.71802618, 7.60053709],
            [4.90600873, 7.54682779],
            [5.06713662, 7.50654582],
            [5.34911044, 7.46626385],
            [5.63108426, 7.46626385],
            [5.8727761, 7.46626385],
            [6.10104062, 7.50654582],
            [6.32930514, 7.50654582],
            [6.6381336, 7.58710977],
            [7.04095334, 7.68110104],
            [7.22893588, 7.78851964],
            [7.34978181, 7.88251091],
            [7.47062773, 8.03021148],
            [7.64518295, 8.17791205],
            [8.06143001, 8.258476],
            [8.35683115, 8.27190332],
            [8.85364216, 7.90936556],
            [8.85364216, 7.76166499],
            [8.85364216, 7.56025512],
            [8.66565962, 7.3856999],
            [8.35683115, 7.3319906],
            [8.12856663, 7.26485398],
            [8.02114804, 7.17086271],
            [7.90030211, 6.96945284],
            [7.84659282, 6.60691507],
            [7.83316549, 6.24437731],
            [7.83316549, 6.04296744],
            [7.83316549, 5.81470292],
            [7.83316549, 5.73413897],
            [7.72574689, 5.65357503],
            [7.564619, 5.70728432],
            [7.34978181, 5.86841222],
            [7.32292716, 6.01611279],
            [7.32292716, 6.29808661],
            [7.4034911, 6.83517959],
            [7.39006378, 6.92917086],
            [7.20208124, 7.21114468],
            [7.09466264, 7.30513595],
            [6.97381672, 7.31856328],
        ]
    )
length_curve = curves.measure_length(tether)
if length_curve > length_max:
    raise ValueError(f"Tether is too long {length_curve}>{length_max}")
signature = curves.compute_signature(tether, env, simplify=True)
print("\nTether input:")
print(tether)
ax.plot(
    tether[:, 0],
    tether[:, 1],
    "-o",
    color="#004381",
    linewidth=1,
    markersize=2,
    zorder=8,
)

# Resample tether
if resampling == "linear":
    tether_segments = np.linalg.norm(np.diff(tether, axis=0), axis=1)  # (n-1,)
    tether_spacing = np.concatenate([[0], np.cumsum(tether_segments)])  # (n,)
    tether_spacing_new = np.linspace(0, tether_spacing[-1], n_nodes)
    tether_new = np.empty((n_nodes, tether.shape[1]))
    for d in range(tether.shape[1]):
        tether_new[:, d] = np.interp(tether_spacing_new, tether_spacing, tether[:, d])
elif resampling == "spline":
    spline, _ = splprep(tether.T, s=0)  # parametric spline through points
    tether_new = np.array(splev(np.linspace(0, 1, n_nodes), spline)).T
tether = tether_new
print("\nTether resampled:")
print(tether)
ax.plot(
    tether[:, 0],
    tether[:, 1],
    "-o",
    color="#690000",
    linewidth=1,
    markersize=2,
    zorder=8,
)

# Create tether object
tether_fem = TetherFEM2D(
    env=env,
    n_nodes=n_nodes,
    state=tether,
    input_mode="position",
    dt=1e-4,  # dt
    medium="water",  # should match setting used in real simulation
    water_current=np.array([0.0, 0.0]),
    gravity=False,  # should match setting used in real simulation
)

# Simulate tether with no endpoint motion
# NOTE: this step is optional and can be performed to remove all internal forces from
# the tether by letting the simulation run and the tether come at a rest .
t_preprocessing: float = 3.0
for k in tqdm(range(int(t_preprocessing / tether_fem.dt))):
    tether_fem.step(np.array([0, 0]))
ax.plot(
    tether_fem.state[:2, 0],
    tether_fem.state[:2, 1],
    "-o",
    color="#004E58",
    linewidth=1,
    markersize=2,
    zorder=8,
)

# Save plot with tethers
fig.savefig(
    "results/trajectory_planning/tether-preprocessing.png", dpi=1200, format="png"
)

# Update env object
env.tether_state = tether
env.tether_configuration = LineString(tether)
env.robot_initial_pos = tether[-1]

########################################################################################
# Plan + execute trajectory ############################################################
########################################################################################

# Define time vector
t_end: float = 60.0
n_steps: int = int(t_end / tether_fem.dt)
k_plot: list[int] = [0, n_steps - 1]  # TODO
k_snapshots: list[int] = [0]  # TODO
snapshots: list[np.ndarray] = []

# Motion profile
path: np.ndarray
if manually_select_path is True:
    path = curves.generate_curve(
        env,
        init_point=env.robot_initial_pos,
        check_self_intersection=True,
        max_points=50,
        show_goal=True,
        show_robot_anchor_labels=False,
        show_legend=False,
        output_type="numpy",
    )
else:
    path = np.array([[]])  # TODO

# Resample path
if resampling == "linear":
    path_segments = np.linalg.norm(np.diff(path, axis=0), axis=1)  # (n-1,)
    path_spacing = np.concatenate([[0], np.cumsum(path_segments)])  # (n,)
    path_spacing_new = np.linspace(0, path_spacing[-1], n_steps)
    path_new = np.empty((n_steps, path.shape[1]))
    for d in range(path.shape[1]):
        path_new[:, d] = np.interp(path_spacing_new, path_new, path[:, d])
elif resampling == "spline":
    spline, _ = splprep(path.T, s=0)  # parametric spline through points
    path_new = np.array(splev(np.linspace(0, 1, n_steps), spline)).T
path = path_new

# Initialize data structures
state_mat = np.zeros([n_steps + 1, n_nodes, 6])  # tether state
state_mat[0] = tether_fem.state.copy()

# Run simulation
for k in tqdm(range(n_steps)):

    # Simulate FEM time step
    tether_fem.step(np.array([0, 0]))  # no motion of endpoint # TEMP
    # tether_fem.step(path[k, :])
    state_mat[k + 1, :, :] = tether_fem.state.copy()

    # Collect snapshots for plots
    if k in k_snapshots:
        snapshots.append(state_mat[k, :, :2])

    # Plot results
    if k in k_plot:
        fig, ax = plot_fem.plot_fem(
            env=env,
            tether_init=state_mat[0, :, :2],
            tether_final=state_mat[k, :, :2],
            trajectory=np.column_stack(path[:k, :]),
            tether_snapshots=snapshots,
            show_plot=True,
            figsize=np.array([4.2, 4.2]),
        )
        fig.savefig(f"results/trajectory_planning/step_{k}.png", dpi=1200, format="png")
