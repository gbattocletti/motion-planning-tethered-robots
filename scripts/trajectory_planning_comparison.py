""" "
Perform trajectory planning and execution.
"""

import datetime
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
from shapely import LineString, Point

from tethered_planning.env import env_2d
from tethered_planning.env.triangulation import Triangulation
from tethered_planning.plan import graph_search, traj_nlp
from tethered_planning.utils import curves, plot
from tethered_planning.utils.settings import Settings

# Scenario settings
env_name = "env_5"
length_max = 15.0
goal = np.array([7.0, 1.0])

# Trajectory optimization parameters
n_steps = 30  # most likely one to need updating to get feasible solution
dt = 0.5
max_speed = 2.0
max_acceleration = 1.0
obstacle_clearance = 0.2
weight_tracking = 1.0
weight_input = 2.0
weight_smoothness = 1.0

# Toggle which trajectory optimization methods to run
manually_select_tether: bool = False
run_qp_gurobi: bool = False
run_nlp_ipopt: bool = False
run_miqp_gurobi: bool = True
run_minlp_knitro: bool = False

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
            [3.0, 7.0],
            [3.05303793, 6.84860692],
            [3.09331991, 6.64719705],
            [3.44243035, 6.15038604],
            [4.11379658, 5.59986573],
            [4.2346425, 5.46559248],
            [4.55689829, 5.30446459],
            [5.01342732, 5.15676401],
            [5.56394763, 5.08962739],
            [6.27559584, 5.12990937],
            [6.65156093, 5.25075529],
            [6.98724404, 5.46559248],
            [7.17522659, 5.65357503],
            [7.37663646, 5.85498489],
            [7.5109097, 6.02954011],
            [7.564619, 6.16381336],
            [7.59147365, 6.31151393],
        ]
    )
length_curve = curves.measure_length(tether)
if length_curve > length_max:
    raise ValueError(f"Tether is too long {length_curve}>{length_max}")
signature = curves.compute_signature(tether, env, simplify=True)

# Update env object
env.tether_state = tether
env.tether_configuration = LineString(tether)
env.robot_initial_pos = tether[-1]

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
    figsize=[10, 10],
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
    "results/trajectory_planning_comparison/initial_conditions.png",
    dpi=1200,
    format="png",
    bbox_inches="tight",
)
plt.close(fig)

########################################################################################
# Path planning ########################################################################
########################################################################################

# Path planning
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
path = paths_lift[0]

########################################################################################
# Trajectory optimization (initialization) #############################################
########################################################################################

# Create parameters object for trajectory optimization
params = traj_nlp.TrajParams(
    n_steps=n_steps,
    dt=dt,
    control_mode="force",
    max_speed=max_speed,
    max_acceleration=max_acceleration,
    obstacle_clearance=obstacle_clearance,
    weight_tracking=weight_tracking,
    weight_input=weight_input,
    weight_smoothness=weight_smoothness,
)

# Compute corridor and geodesic for initialization of trajectory optimization
corridor_triangles, geodesic, edge_normals, edge_offsets = traj_nlp.corridor(
    triang,
    path,
    env.robot_initial_pos,
    goal,
    params.obstacle_clearance,
)

# Log parameters and settings
with open("results/trajectory_planning_comparison/log.txt", "w", encoding="utf-8") as f:
    f.write("Trajectory planning comparison\n")
    f.write(f"[{datetime.datetime.now()}]\n")
    f.write(f"env: {env_name}\n")
    f.write(f"length_max: {length_max}\n")
    f.write(f"goal: {goal}\n")
    f.write(f"robot: {env.robot_initial_pos}\n")
    f.write(f"tether: {tether}\n")
    f.write("\nparameters:\n")
    f.write(f"\tn_steps: {n_steps}\n")
    f.write(f"\tdt: {dt}\n")
    f.write(f"\tmax_speed: {max_speed}\n")
    f.write(f"\tmax_acceleration: {max_acceleration}\n")
    f.write(f"\tobstacle_clearance: {obstacle_clearance}\n")
    f.write(f"\tweight_tracking: {weight_tracking}\n")
    f.write(f"\tweight_input: {weight_input}\n")
    f.write(f"\tweight_smoothness: {weight_smoothness}\n\n")

    str_csv = ""  # init string for csv output

    # Trajectory optimization QP #######################################################
    if run_qp_gurobi is True:
        solution = traj_nlp.solve_nlp(
            edge_normals,
            edge_offsets,
            geodesic,
            env.robot_initial_pos,
            goal,
            params,
            solver="gurobi",
            verbose=False,
        )
        print(
            f"QP (gurobi):\t t: {solution['solve_time']:.4f}s, "
            f"cost: {solution['cost']:.8f}"
        )
        f.write(
            f"QP (gurobi):\t t: {solution['solve_time']:.4f}s, "
            f"cost: {solution['cost']:.8f}\n"
        )
        str_csv += f"{solution['solve_time']:.4f}, {solution['cost']:.8f}, "

        # Extract solution
        positions = solution["positions"]
        velocities = solution["velocities"]
        inputs = solution["inputs"]
        triangle_of_knot = solution["triangle_of_knot"]

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
            figsize=[10, 10],
        )
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.plot(goal[0], goal[1], color="green", marker="o", markersize=3, zorder=10)
        ax.plot(
            tether[-1, 0],
            tether[-1, 1],
            color="blue",
            marker="o",
            markersize=3,
            zorder=10,
        )
        ax.plot(anchor[0], anchor[1], color="red", marker="o", markersize=3, zorder=10)
        ax.plot(
            tether[:, 0],
            tether[:, 1],
            "-",
            color="#000000",
            linewidth=1,
            zorder=8,
        )
        ax.plot(
            positions[:, 0],
            positions[:, 1],
            "-o",
            color="#006CD1",
            linewidth=1.2,
            markersize=1.5,
            zorder=9,
        )
        fig.savefig(
            "results/trajectory_planning_comparison/trajectory_qp.png",
            dpi=1200,
            format="png",
            bbox_inches="tight",
        )
        plt.close(fig)
    else:
        str_csv += "N/A, N/A, "

    # Trajectory optimization NLP ######################################################
    if run_nlp_ipopt is True:
        solution = traj_nlp.solve_nlp(
            edge_normals,
            edge_offsets,
            geodesic,
            env.robot_initial_pos,
            goal,
            params,
            solver="ipopt",
            verbose=False,
        )
        print(
            f"NLP (ipopt):\t t: {solution['solve_time']:.4f}s, "
            f"cost: {solution['cost']:.8f}"
        )
        f.write(
            f"NLP (ipopt):\t t: {solution['solve_time']:.4f}s, "
            f"cost: {solution['cost']:.8f}\n"
        )
        str_csv += f"{solution['solve_time']:.4f}, {solution['cost']:.8f}, "

        # Extract solution
        positions = solution["positions"]
        velocities = solution["velocities"]
        inputs = solution["inputs"]
        triangle_of_knot = solution["triangle_of_knot"]

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
            figsize=[10, 10],
        )
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.plot(goal[0], goal[1], color="green", marker="o", markersize=3, zorder=10)
        ax.plot(
            tether[-1, 0],
            tether[-1, 1],
            color="blue",
            marker="o",
            markersize=3,
            zorder=10,
        )
        ax.plot(anchor[0], anchor[1], color="red", marker="o", markersize=3, zorder=10)
        ax.plot(
            tether[:, 0],
            tether[:, 1],
            "-",
            color="#000000",
            linewidth=1,
            zorder=8,
        )
        ax.plot(
            positions[:, 0],
            positions[:, 1],
            "-o",
            color="#006CD1",
            linewidth=1.2,
            markersize=1.5,
            zorder=9,
        )
        fig.savefig(
            "results/trajectory_planning_comparison/trajectory_nlp.png",
            dpi=1200,
            format="png",
            bbox_inches="tight",
        )
        plt.close(fig)
    else:
        str_csv += "N/A, N/A, "

    # Trajectory optimization MIQP #####################################################
    if run_miqp_gurobi is True:
        solution = traj_nlp.solve_minlp(
            edge_normals,
            edge_offsets,
            geodesic,
            env.robot_initial_pos,
            goal,
            params,
            solver="gurobi",
            verbose=False,
        )
        print(
            f"MIQP (gurobi):\t t: {solution['solve_time']:.4f}s, "
            f"cost: {solution['cost']:.8f}"
        )
        f.write(
            f"MIQP (gurobi):\t t: {solution['solve_time']:.4f}s, "
            f"cost: {solution['cost']:.8f}\n"
        )
        str_csv += f"{solution['solve_time']:.4f}, {solution['cost']:.8f}, "

        # Extract solution
        positions = solution["positions"]
        velocities = solution["velocities"]
        inputs = solution["inputs"]
        triangle_of_knot = solution["triangle_of_knot"]

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
            figsize=[10, 10],
        )
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.plot(goal[0], goal[1], color="green", marker="o", markersize=3, zorder=10)
        ax.plot(
            tether[-1, 0],
            tether[-1, 1],
            color="blue",
            marker="o",
            markersize=3,
            zorder=10,
        )
        ax.plot(anchor[0], anchor[1], color="red", marker="o", markersize=3, zorder=10)
        ax.plot(
            tether[:, 0],
            tether[:, 1],
            "-",
            color="#000000",
            linewidth=1,
            zorder=8,
        )
        ax.plot(
            positions[:, 0],
            positions[:, 1],
            "-o",
            color="#006CD1",
            linewidth=1.2,
            markersize=1.5,
            zorder=9,
        )
        fig.savefig(
            "results/trajectory_planning_comparison/trajectory_miqp.png",
            dpi=1200,
            format="png",
            bbox_inches="tight",
        )
        plt.close(fig)
    else:
        str_csv += "N/A, N/A, "

    # Trajectory optimization MINLP ####################################################
    if run_minlp_knitro is True:
        solution = traj_nlp.solve_minlp(
            edge_normals,
            edge_offsets,
            geodesic,
            env.robot_initial_pos,
            goal,
            params,
            solver="knitro",
            verbose=False,
            solver_options={"numthreads": 8, "mip_numthreads": 8},
        )
        print(
            f"MINLP (knitro):\t t: {solution['solve_time']:.4f}s, "
            f"cost: {solution['cost']:.8f}"
        )
        f.write(
            f"MINLP (knitro):\t t: {solution['solve_time']:.4f}s, "
            f"cost: {solution['cost']:.8f}\n"
        )
        str_csv += f"{solution['solve_time']:.4f}, {solution['cost']:.8f}"

        # Extract solution
        positions = solution["positions"]
        velocities = solution["velocities"]
        inputs = solution["inputs"]
        triangle_of_knot = solution["triangle_of_knot"]

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
            figsize=[10, 10],
        )
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.plot(goal[0], goal[1], color="green", marker="o", markersize=3, zorder=10)
        ax.plot(
            tether[-1, 0],
            tether[-1, 1],
            color="blue",
            marker="o",
            markersize=3,
            zorder=10,
        )
        ax.plot(anchor[0], anchor[1], color="red", marker="o", markersize=3, zorder=10)
        ax.plot(
            tether[:, 0],
            tether[:, 1],
            "-",
            color="#000000",
            linewidth=1,
            zorder=8,
        )
        ax.plot(
            positions[:, 0],
            positions[:, 1],
            "-o",
            color="#006CD1",
            linewidth=1.2,
            markersize=1.5,
            zorder=9,
        )
        fig.savefig(
            "results/trajectory_planning_comparison/trajectory_minlp.png",
            dpi=1200,
            format="png",
            bbox_inches="tight",
        )
        plt.close(fig)
    else:
        str_csv += "N/A, N/A"

    f.write("\n")
    f.write(str_csv + "\n")

f.close()
