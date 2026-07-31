""" "
Perform trajectory planning and execution.
"""

import os
import pickle
import time

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
env_name = "env_4"
length_max = 15.0
goal = np.array([7.0, 7.0])
n_nodes = 30  # nodes in FEM model
resampling: str = "linear"  # {linear, spline}
manually_select_tether: bool = False
run_tether_preprocessing: bool = False
path_selection_method: str = "path_planning"  # {manual, prespecified, path_planning}
t_end: float = 15.0  # simulation time

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
    print(tether)
else:
    tether = np.array(
        [
            [4.5, 3.5],
            [4.59718026, 3.34407519],
            [4.73145351, 3.0755287],
            [4.71802618, 2.83383686],
            [4.69117153, 2.40416247],
            [4.47633434, 2.26988922],
            [3.99295065, 2.10876133],
            [3.64384021, 2.18932528],
            [3.49613964, 2.63242699],
            [3.48271232, 3.03524673],
            [3.61698557, 3.39778449],
            [3.68412219, 3.74689493],
            [3.45585767, 4.12286002],
            [3.22759315, 4.40483384],
            [2.86505539, 4.52567976],
            [2.69050017, 4.47197046],
            [2.34138973, 4.24370594],
            [2.00570661, 4.25713327],
            [1.83115139, 4.43168849],
            [1.92514267, 4.64652568],
            [2.34138973, 4.83450822],
            [2.52937227, 5.15676401],
            [2.5427996, 5.31789191],
            [2.51594495, 5.50587445],
            [2.44880832, 5.76099362],
            [2.40852635, 5.93554884],
            [2.435381, 6.28465928],
            [2.60993622, 6.6203424],
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
ax.plot(tether[-1, 0], tether[-1, 1], color="blue", marker="o", markersize=3, zorder=10)
ax.plot(anchor[0], anchor[1], color="red", marker="o", markersize=3, zorder=10)
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
    t_preprocessing: float = 5.0
    for k in tqdm(range(int(t_preprocessing / tether_fem.dt))):
        tether_fem.step(tether[-1])  # fixed endpoint
    print(tether_fem.state[:, :2])
else:
    # Manually copy + paste preprocessed tether here to skip numerical preprocessing
    tether_fem.state[:, :2] = np.array(
        [
            [4.5, 3.5],
            [4.63966853, 3.23866551],
            [4.74479477, 2.961625],
            [4.77925643, 2.66732012],
            [4.71179729, 2.37878552],
            [4.53415796, 2.14162099],
            [4.27387598, 2.00000074],
            [3.9798227, 1.96345734],
            [3.7015133, 2.0651766],
            [3.51013359, 2.29139921],
            [3.44693, 2.58089558],
            [3.49745266, 2.87287228],
            [3.61522614, 3.14477746],
            [3.7486513, 3.40935394],
            [3.84169376, 3.69068298],
            [3.82697647, 3.98663271],
            [3.66081585, 4.23197616],
            [3.388655, 4.34915665],
            [3.09234297, 4.35060484],
            [2.79890144, 4.30943395],
            [2.50260389, 4.30616631],
            [2.22723176, 4.41558747],
            [2.0551887, 4.65684236],
            [2.03919597, 4.95272585],
            [2.13515804, 5.23307242],
            [2.275084, 5.49426908],
            [2.41000597, 5.75808536],
            [2.51141371, 6.03650846],
            [2.57317789, 6.32631554],
            [2.60993622, 6.6203424],
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
ax.plot(tether[-1, 0], tether[-1, 1], color="blue", marker="o", markersize=3, zorder=10)
ax.plot(anchor[0], anchor[1], color="red", marker="o", markersize=3, zorder=10)
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
ax.plot(tether[-1, 0], tether[-1, 1], color="blue", marker="o", markersize=3, zorder=10)
ax.plot(anchor[0], anchor[1], color="red", marker="o", markersize=3, zorder=10)
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
tether_fem.flow = np.array([0.0, 0.0])
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
    # prespecified path
    path = np.array([])
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
    path = paths_lift[3]

    # trajectory optimization
    params = traj_nlp.TrajParams(
        n_steps=200,
        dt=0.5,
        control_mode="force",
        max_speed=2.0,
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

    # Compute trajectory
    t_init = time.process_time()
    solution = traj_nlp.solve_nlp(
        edge_normals,
        edge_offsets,
        geodesic,
        env.robot_initial_pos,
        goal,
        params,
    )
    t_traj = time.process_time() - t_init
    print(f"[Trajectory optimization completed in {t_traj:.4f}s]")

    # Extract solution
    positions = solution["positions"]
    velocities = solution["velocities"]
    inputs = solution["inputs"]
    triangle_of_knot = solution["triangle_of_knot"]
    traj = positions  # NOTE: position control for simulation

# Plot and save trajectory
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
ax.plot(tether[-1, 0], tether[-1, 1], color="blue", marker="o", markersize=3, zorder=10)
ax.plot(anchor[0], anchor[1], color="red", marker="o", markersize=3, zorder=10)
ax.plot(
    tether_fem.state[:, 0],
    tether_fem.state[:, 1],
    "-o",
    color="#000000",
    linewidth=1,
    markersize=1.5,
    zorder=8,
)
ax.plot(
    traj[:, 0],
    traj[:, 1],
    "-",
    color="#006CD1",
    linewidth=1.2,
    zorder=9,
)
fig.savefig(
    "results/trajectory_planning/trajectory.png",
    dpi=1200,
    format="png",
    bbox_inches="tight",
)
plt.close(fig)

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
            f"results/trajectory_planning/step-{k}.png",
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
    f"results/trajectory_planning/step-{k}.png",
    dpi=1200,
    format="png",
    bbox_inches="tight",
)
plt.close(fig)
