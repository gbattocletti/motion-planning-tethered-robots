""" "
Perform trajectory planning and execution.
"""

import os
import pickle
import time

import matplotlib.pyplot as plt
import numpy as np
from shapely import LineString, Point

from tethered_planning.env import env_2d
from tethered_planning.env.triangulation import Triangulation
from tethered_planning.plan import graph_search, traj_nlp
from tethered_planning.utils import curves, plot
from tethered_planning.utils.settings import Settings

# Script settings
env_name = "env_5"
length_max = 15.0
goal = np.array([7.0, 1.0])
manually_select_tether: bool = False
run_qp_gurobi: bool = True
run_nlp_ipopt: bool = True
run_miqp_gurobi: bool = True
run_minlp_knitro: bool = True

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
            [3.26787513, 6.64719705],
            [3.29472978, 6.47264183],
            [3.17388385, 6.12353139],
            [2.99932863, 5.92212152],
            [2.66364552, 5.5864384],
            [2.26082578, 5.37160121],
            [1.91171534, 5.29103726],
            [1.5088956, 5.10305472],
            [1.20006714, 4.96878147],
            [1.03893924, 4.91507217],
            [0.97180262, 4.75394428],
            [0.94494797, 4.60624371],
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

params = traj_nlp.TrajParams(
    n_steps=30,
    dt=0.5,
    control_mode="force",
    max_speed=2.0,
    max_acceleration=1,
    obstacle_clearance=0.2,
)
corridor_triangles, geodesic, edge_normals, edge_offsets = traj_nlp.corridor(
    triang,
    path,
    env.robot_initial_pos,
    goal,
    params.obstacle_clearance,
)

with open(
    "results/trajectory_planning_comparison/log.txt", "wb", encoding="utf-8"
) as f:
    f.write(f"env: {env_name}\n")
    f.write(f"length_max: {length_max}\n")
    f.write(f"goal: {goal}\n")
    f.write(f"robot: {env.robot_initial_pos}\n")
    f.write(f"tether: {tether}\n")

    # Trajectory optimization QP #######################################################
    if run_qp_gurobi is True:
        t_init = time.process_time()
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
        t_traj = time.process_time() - t_init
        print(f"QP (gurobi):\t t: {t_traj:.4f}s, cost: {solution['cost']:.8f}")
        f.write(f"QP (gurobi):\t t: {t_traj:.4f}s, cost: {solution['cost']:.8f}\n")

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

    # Trajectory optimization NLP ######################################################
    if run_nlp_ipopt is True:
        t_init = time.process_time()
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
        t_traj = time.process_time() - t_init
        print(f"NLP (ipopt):\t t: {t_traj:.4f}s, cost: {solution['cost']:.8f}")
        f.write(f"NLP (ipopt):\t t: {t_traj:.4f}s, cost: {solution['cost']:.8f}\n")

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

    # Trajectory optimization MIQP #####################################################
    if run_miqp_gurobi is True:
        t_init = time.process_time()
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
        t_traj = time.process_time() - t_init
        print(f"MIQP (gurobi):\t t: {t_traj:.4f}s, cost: {solution['cost']:.8f}")
        f.write(f"MIQP (gurobi):\t t: {t_traj:.4f}s, cost: {solution['cost']:.8f}\n")

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

    # Trajectory optimization MINLP ####################################################
    if run_minlp_knitro is True:
        t_init = time.process_time()
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
        t_traj = time.process_time() - t_init
        print(f"MINLP (knitro):\t t: {t_traj:.4f}s, cost: {solution['cost']:.8f}")
        f.write(f"MINLP (knitro):\t t: {t_traj:.4f}s, cost: {solution['cost']:.8f}\n")

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
f.close()
