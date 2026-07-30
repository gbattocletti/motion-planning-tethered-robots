""" "
Perform trajectory planning and execution.
"""

import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import splev, splprep
from shapely import LineString, Point
from tqdm import tqdm

from tethered_planning.env import env_2d
from tethered_planning.env.triangulation import Triangulation
from tethered_planning.plan import graph_search, traj_nlp
from tethered_planning.tether.fem import TetherFEM2D
from tethered_planning.utils import curves, plot, plot_fem
from tethered_planning.utils.settings import Settings

# Script settings
env_name = "env_5"
length_max = 15.0
goal = np.array([7.5, 1.0])
n_nodes = 40  # nodes in FEM model
resampling: str = "linear"  # {linear, spline}
manually_select_tether: bool = False
run_tether_preprocessing: bool = False
path_selection_method: str = "path_planning"  # {manual, prespecified, path_planning}
path_id: int = 1  # {1, 2, 3} to track prespecified paths
t_end: float = 15.0  # simulation tiime

########################################################################################
# Load data ############################################################################
########################################################################################
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Select datafile to use
datapath: str
match (env_name, length_max):
    case ("env_3b", 10):
        datapath = "results/entanglement_free_model/comparison-21.pkl"
    case ("env_3b", 12.5):
        datapath = "results/entanglement_free_model/comparison-24.pkl"
    case ("env_3b", 15.0):
        datapath = "results/entanglement_free_model/comparison-27.pkl"
    case ("env_3", 10):
        datapath = "results/entanglement_free_model/comparison-30.pkl"
    case ("env_3", 12.5):
        datapath = "results/entanglement_free_model/comparison-33.pkl"
    case ("env_3", 15.0):
        datapath = "results/entanglement_free_model/comparison-36.pkl"
    case ("env_4", 10):
        datapath = "results/entanglement_free_model/comparison-39.pkl"
    case ("env_4", 12.5):
        datapath = "results/entanglement_free_model/comparison-42.pkl"
    case ("env_4", 15.0):
        datapath = "results/entanglement_free_model/comparison-45.pkl"
    case ("env_5", 10):
        datapath = "results/entanglement_free_model/comparison-48.pkl"
    case ("env_5", 12.5):
        datapath = "results/entanglement_free_model/comparison-51.pkl"
    case ("env_5", 15.0):
        datapath = "results/entanglement_free_model/comparison-54.pkl"
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
triang: Triangulation = data["triangulation_entanglement"]
anchor: np.ndarray = env.anchor_point
settings: Settings = Settings()
settings.env_name = env_name
env: env_2d.Env2D = env_2d.Env2D(settings)
env.goal_vertices = goal


########################################################################################
# Initialize + preprocess tether #######################################################
########################################################################################

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

# Initialize plot to show tethers
fig, ax = plot.plot_env(
    env,
    show_tether=False,
    show_robot=False,
    show_anchor=False,
    show_goal=False,
    show_legend=False,
    show_generators=False,
    show_curves_labels=False,
    show_robot_anchor_labels=False,
    show_generators_labels=False,
    show_obstacles_labels=False,
    show_axes_labels=False,
    figsize=[5, 5],
)
ax.set_xlabel("")
ax.set_ylabel("")
ax.set_xticklabels([])
ax.set_yticklabels([])
ax.plot(goal[0], goal[1], color="green", marker="o", markersize=3, zorder=10)
ax.plot(tether[-1, 0], tether[-1, 1], color="red", marker="o", markersize=3, zorder=10)
ax.plot(anchor[0], anchor[1], color="blue", marker="o", markersize=3, zorder=10)
ax.plot(
    tether[:, 0],
    tether[:, 1],
    "-o",
    color="#00818A",
    linewidth=1,
    markersize=1.5,
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
tether = tether_new  # resampled tether
ax.plot(
    tether[:, 0],
    tether[:, 1],
    "-o",
    color="#003274",
    linewidth=1,
    markersize=1.5,
    zorder=8,
)

# Create tether object
tether_fem = TetherFEM2D(
    env=env,
    n_nodes=n_nodes,
    state=tether,
    input_mode="position",
    dt=5e-4,  # dt
    medium="water",  # should match setting used in real simulation
    water_current=np.array([0.0, 0.0]),
    gravity=False,  # should match setting used in real simulation
)

# Simulate tether with no endpoint motion
# This step serves to relax internal forces in tether and reach an initial equilibrium.
if run_tether_preprocessing is True:
    print("[Running preprocessing simulation]")
    t_preprocessing: float = 10.0
    for k in tqdm(range(int(t_preprocessing / tether_fem.dt))):
        tether_fem.step(tether[-1])  # fixed endpoint
else:
    tether_fem.state[:, :2] = np.array(
        [
            [3.0, 7.0],
            [2.65878581, 6.9266135],
            [2.30977623, 6.92839616],
            [1.99549386, 7.0801655],
            [1.84047946, 7.39285463],
            [1.94311465, 7.72642644],
            [2.22594507, 7.93091259],
            [2.56805339, 7.99999969],
            [2.9170702, 7.999999],
            [3.26530764, 7.97668273],
            [3.61212372, 7.93754649],
            [3.95782714, 7.88956534],
            [4.30320924, 7.83932334],
            [4.64917522, 7.79327345],
            [4.99640723, 7.75801792],
            [5.34498333, 7.740477],
            [5.69391921, 7.74800486],
            [6.04058329, 7.78846286],
            [6.3799594, 7.86992805],
            [6.703833, 7.999999],
            [7.000001, 8.18465579],
            [7.28121703, 8.39136731],
            [7.5841701, 8.56466109],
            [7.92505148, 8.6395556],
            [8.26249475, 8.55045403],
            [8.49640988, 8.29143668],
            [8.5586657, 7.94802446],
            [8.47490878, 7.60921059],
            [8.31259873, 7.30023299],
            [8.13463011, 7.000001],
            [7.999999, 6.67799546],
            [7.95467586, 6.33193517],
            [7.87931231, 5.99115512],
            [7.66366457, 5.71674474],
            [7.31635552, 5.68245051],
            [7.0711712, 5.93081977],
            [7.000001, 6.27249804],
            [7.000001, 6.62151407],
            [7.000001, 6.97053042],
            [6.97381672, 7.31856328],
        ]
    )
    tether_fem.state[:, 2:] = np.zeros([n_nodes, 4])

# Save plot with tether preprocessing
ax.plot(
    tether_fem.state[:, 0],
    tether_fem.state[:, 1],
    "-o",
    color="#000000",
    linewidth=1,
    markersize=1.5,
    zorder=8,
)
fig.savefig(
    "results/trajectory_planning/initial-preprocessing.png",
    dpi=1200,
    format="png",
    bbox_inches="tight",
)
plt.close(fig)

# Plot and save initial conditions
fig, ax = plot.plot_env(
    env,
    show_tether=False,
    show_robot=False,
    show_anchor=False,
    show_goal=False,
    show_legend=False,
    show_generators=False,
    show_curves_labels=False,
    show_robot_anchor_labels=False,
    show_generators_labels=False,
    show_obstacles_labels=False,
    show_axes_labels=False,
    figsize=[5, 5],
)
ax.set_xlabel("")
ax.set_ylabel("")
ax.set_xticklabels([])
ax.set_yticklabels([])
ax.plot(goal[0], goal[1], color="green", marker="o", markersize=3, zorder=10)
ax.plot(tether[-1, 0], tether[-1, 1], color="red", marker="o", markersize=3, zorder=10)
ax.plot(anchor[0], anchor[1], color="blue", marker="o", markersize=3, zorder=10)
ax.plot(
    tether[:, 0],
    tether[:, 1],
    "-o",
    color="#000000",
    linewidth=1,
    markersize=1.5,
    zorder=8,
)
fig.savefig(
    "results/trajectory_planning/initial-conditions.png",
    dpi=1200,
    format="png",
    bbox_inches="tight",
)
plt.close(fig)

# Plot and save initial conditions
fig, ax = plot.plot_env(
    env,
    show_tether=False,
    show_robot=False,
    show_anchor=False,
    show_goal=False,
    show_legend=False,
    show_generators=False,
    show_curves_labels=False,
    show_robot_anchor_labels=False,
    show_generators_labels=False,
    show_obstacles_labels=False,
    show_axes_labels=False,
    figsize=[5, 5],
)
ax.set_xlabel("")
ax.set_ylabel("")
ax.set_xticklabels([])
ax.set_yticklabels([])
ax.plot(goal[0], goal[1], color="green", marker="o", markersize=3, zorder=10)
ax.plot(tether[-1, 0], tether[-1, 1], color="red", marker="o", markersize=3, zorder=10)
ax.plot(anchor[0], anchor[1], color="blue", marker="o", markersize=3, zorder=10)
ax.plot(
    tether_fem.state[:, 0],
    tether_fem.state[:, 1],
    "-o",
    color="#000000",
    linewidth=1,
    markersize=1.5,
    zorder=8,
)
fig.savefig(
    "results/trajectory_planning/initial-conditions-fem.png",
    dpi=1200,
    format="png",
    bbox_inches="tight",
)
plt.close(fig)

# Update env object
env.tether_state = tether
env.tether_configuration = LineString(tether)
env.robot_initial_pos = tether[-1]

########################################################################################
# Plan and trajectory planning #########################################################
########################################################################################

# Update FEM environmental conditions
tether_fem.current = np.array([0.0, 0.0])
tether_fem.wind = np.array([0.0, 0.0])
tether_fem.gravity_enabled = False
n_steps: int = int(t_end / tether_fem.dt)  # number of simulation steps

# Path planning
traj: np.ndarray
path: np.ndarray
if path_selection_method == "manual":
    # manually define path by drawing it on plot
    path = curves.generate_curve(
        env,
        init_point=env.robot_initial_pos,
        check_self_intersection=True,
        max_points=50,
        show_goal=True,
        show_tether=True,
        show_robot=True,
        show_anchor=True,
        show_robot_anchor_labels=False,
        show_legend=False,
        output_type="numpy",
    )
    traj = path  # TODO: apply traj optimization
elif path_selection_method == "prespecified":
    # select path among prespecified options and use it as constant speed trajectory
    match path_id:
        case 1:
            path = np.array(
                [
                    [6.97381672, 7.31856328],
                    [5.96676737, 7.18429003],
                    [5.53709298, 6.28465928],
                    [5.7653575, 4.84793555],
                    [6.34273246, 3.66633098],
                    [6.81268882, 2.18932528],
                    [7.47062773, 0.98086606],
                    [7.50000000, 1.00000000],
                ]
            )
        case _:
            raise ValueError("Invalid path id")
    traj = path  # TODO: apply traj optimization
elif path_selection_method == "path_planning":
    # perform path planning on simplicial complex model
    print("[Running path planning]")
    triang_goal = int(
        triang.triang_tree.query(
            Point(goal),
            predicate="intersects",
        )[0]
    )
    triang_robot = int(
        triang.triang_tree.query(
            Point(env.robot_initial_pos),
            predicate="intersects",
        )[0]
    )
    triang_goal_lift_list = [
        idx
        for idx, tri in enumerate(triang.vertices_dual_lift)
        if tri[0] == triang_goal and triang.entanglement_vertices_dual_lift[idx] is True
    ]
    triang_robot_lift = [
        idx
        for idx, tri in enumerate(triang.vertices_dual_lift)
        if ((tri[0] == triang_robot) and (tri[1] == signature))
    ][0]

    # graph search
    paths_lift = []  # path through lifted space
    for triang_goal_lift in triang_goal_lift_list:
        paths_lift.append(
            graph_search.a_star_search(
                triang.vertices_dual_lift,
                triang.edges_dual_lift,
                triang_robot_lift,
                triang_goal_lift,
                h_augmented=True,
                nodes_2d=triang.vertices_dual,
                use_heuristic=False,
            )
        )
    path = paths_lift[0]  # TODO: allow selection of different indexes

    # trajectory optimization
    params = traj_nlp.TrajParams(
        n_steps=40,
        dt=0.25,
        control_mode="force",
        max_speed=1.2,
        max_acceleration=1.5,
        obstacle_clearance=0.05,
    )
    corridor_triangles, geodesic, edge_normals, edge_offsets = traj_nlp.find_corridor(
        triang,
        path,
        env.robot_initial_pos,
        goal,
        params.obstacle_clearance,
    )
    geodesic_length = np.sum(np.linalg.norm(np.diff(geodesic, axis=0), axis=1))
    print(
        f"corridor: {len(corridor_triangles)} triangles, "
        f"geodesic length {geodesic_length:.2f} m"
    )

    solution = traj_nlp.solve_nlp(
        edge_normals,
        edge_offsets,
        geodesic,
        env.robot_initial_pos,
        goal,
        params,
    )
    positions = solution["positions"]
    velocities = solution["velocities"]
    inputs = solution["inputs"]
    triangle_of_knot = solution["triangle_of_knot"]
    traj = positions  # TEMP

########################################################################################
# Simulation ###########################################################################
########################################################################################

# Initialize data structures for plots and tether snapshots
n_plots: int = 10
k_step_plot: float = n_steps / (n_plots - 1)
k_plot: list[int] = [int(round(i * k_step_plot)) for i in range(n_plots)]
n_snapshots: int = 20
k_step_snapshots: float = n_steps / (n_snapshots - 1)
k_snapshots: list[int] = [int(round(i * k_step_snapshots)) for i in range(n_snapshots)]
snapshots: list[np.ndarray] = []

# Resample path
if resampling == "linear":
    traj_segments = np.linalg.norm(np.diff(traj, axis=0), axis=1)  # (n-1,)
    traj_spacing = np.concatenate([[0], np.cumsum(traj_segments)])  # (n,)
    traj_spacing_new = np.linspace(0, traj_spacing[-1], n_steps)
    traj_new = np.empty((n_steps, traj.shape[1]))
    for d in range(traj.shape[1]):
        traj_new[:, d] = np.interp(traj_spacing_new, traj_spacing, traj[:, d])
elif resampling == "spline":
    spline, _ = splprep(traj.T, s=0)  # parametric spline through points
    traj_new = np.array(splev(np.linspace(0, 1, n_steps), spline)).T
traj = traj_new

# Initialize data structures
state_mat = np.zeros([n_steps + 1, n_nodes, 6])  # tether state
state_mat[0] = tether_fem.state.copy()

# Run simulation
print("[Running main simulation]")
for k in tqdm(range(n_steps)):

    # Simulate FEM time step
    tether_fem.step(traj[k, :])
    state_mat[k + 1, :, :] = tether_fem.state.copy()

    # Collect snapshots for plots
    if k in k_snapshots:
        snapshots.append(state_mat[k, :, :2])

    # Plot results
    if k in k_plot:
        fig, ax = plot_fem.plot_fem(
            env=env,
            tether_init=state_mat[0, :, :2],
            tether_final=state_mat[k, :, :2] if k > 0 else None,
            trajectory=traj[:k, :] if k > 0 else None,
            tether_snapshots=snapshots,
            show_plot=False,
            figsize=np.array([5, 5]),
        )
        fig.savefig(
            f"results/trajectory_planning/path-{path_id}-step-{k}.png",
            dpi=1200,
            format="png",
            bbox_inches="tight",
        )
        plt.close(fig)

# Plot final configuration
fig, ax = plot_fem.plot_fem(
    env=env,
    tether_init=state_mat[0, :, :2],
    tether_final=state_mat[-1, :, :2],
    trajectory=traj[:, :],
    tether_snapshots=snapshots,
    show_plot=False,
    figsize=np.array([5, 5]),
)
fig.savefig(
    f"results/trajectory_planning/path-{path_id}-step-{k}.png",
    dpi=1200,
    format="png",
    bbox_inches="tight",
)
plt.close(fig)
plt.close(fig)
