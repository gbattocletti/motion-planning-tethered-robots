""" "
Perform trajectory planning and execution.
"""

import os
import pickle
import time

import matplotlib.pyplot as plt
import numpy as np
from shapely import Point

from tethered_planning.env import env_2d
from tethered_planning.env.triangulation import Triangulation
from tethered_planning.plan import graph_search, traj_nlp
from tethered_planning.utils import curves, plot
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
    color="#000000",
    linewidth=1,
    markersize=1.5,
    zorder=8,
)

########################################################################################
# Path and trajectory planning #########################################################
########################################################################################
# TODO:
# 1. update code NLP and MINLP with better warmstarting and initialization
#       --> check if they can be turned in MIQP and QP with appropriate constraints,
#           e.g. by making the velocity constraint a box rather than a norm
# 2. duplicate this code to run both MINLP (MIQP?) and NLP (QP?)
# 3. print stats of both methods (time, traj length, cost)


# Path planning
traj: np.ndarray
path: np.ndarray

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
    tether[:, 0],
    tether[:, 1],
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
