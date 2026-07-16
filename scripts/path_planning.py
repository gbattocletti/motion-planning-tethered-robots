"""
Generates a plot showing all paths between the robot and the goal location that are in
an entanglement-free homotopy class. The plot is generated for the same environment and
tether length with different entanglement definitions:
- no entanglement definition
- obstacle free convex hull
- obstacle free linear homotopy
- linear visibility homotopy
Each plot is computed both on the simplicial complex model and homotopy augmented graph
computed for each of these definitions. Therefore, for each run of the script 8 plots
are generated and saved.

NOTE: Some elements in this script, such as the label and some points that are added to
the plots, are tailored to the specific environment defined in env_4.yaml. If the
environment is changed, these elements may need to be adjusted accordingly.
"""

import datetime
import os
import pickle
import time
from collections import defaultdict  # , deque
from collections.abc import Callable

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from packaging import version
from shapely import LineString, Point

from tethered_planning.env import env_2d
from tethered_planning.env.grid_graph import GridGraph
from tethered_planning.env.triangulation import Triangulation

# from tethered_planning.plan import graph_search
from tethered_planning.utils import curves, entanglement, plot
from tethered_planning.utils.colors import CmdColors
from tethered_planning.utils.settings import Settings

# Script settings
SELECT_TETHER_MANUALLY: bool = False
env_name = "env_4"
length_max = 15

# Select experiments
if env_name == "env_4" and length_max == 10:
    experiments = [
        ["results/entanglement_free_model/comparison-25.pkl", "convex_hull"],
        ["results/entanglement_free_model/comparison-26.pkl", "linear_homotopy"],
        ["results/entanglement_free_model/comparison-27.pkl", "local_visibility"],
    ]
elif env_name == "env_4" and length_max == 12:
    experiments = [
        ["results/entanglement_free_model/comparison-28.pkl", "convex_hull"],
        ["results/entanglement_free_model/comparison-29.pkl", "linear_homotopy"],
        ["results/entanglement_free_model/comparison-30.pkl", "local_visibility"],
    ]
elif env_name == "env_4" and length_max == 15:
    experiments = [
        ["results/entanglement_free_model/comparison-31.pkl", "convex_hull"],
        ["results/entanglement_free_model/comparison-32.pkl", "linear_homotopy"],
        ["results/entanglement_free_model/comparison-33.pkl", "local_visibility"],
    ]

# Check matplotlib version
if version.parse(mpl.__version__) <= version.parse("3.7"):
    mpl.use("pgf")
else:
    print(
        f"{CmdColors.WARNING}[IO]{CmdColors.ENDC} PGF export is not supported "
        "in this version of matplotlib. The figure will not be saved."
    )
    SAVE_PGF = False

# Move to script directory
abspath = os.path.abspath(__file__)
dir_name = os.path.dirname(abspath)
os.chdir(dir_name)

# Initialize data structures
settings: Settings
env: env_2d.Env2D
anchor: np.ndarray
robot: np.ndarray
goal: np.ndarray
tether: np.ndarray
colors = [
    "#0050A0",
    "#158AFF",
    "#071F36",
    "#28BBFF",
    "#000BA0",
    "#009CA1",
    "#0EEFFF",
    "#2200E4",
    "#004B55",
    "#246AB1",
]

# Reset log file
with open("results/path-planning.txt", "w", encoding="utf-8") as f:
    f.write(f"[Started at: {datetime.datetime.now()}]\n")
f.close()

# Iterate over experiments and perform path planning
for idx, [filename, definition] in list(enumerate(experiments)):

    # Load data
    objects: list = []
    with open(filename, "rb") as openfile:
        while True:
            try:
                objects.append(pickle.load(openfile))
            except EOFError:
                break
    data: dict = objects[0]
    if idx == 0:
        settings = data["settings"]
        env = data["env"]
        length: float = env.tether_length
        anchor = env.anchor_point
        if SELECT_TETHER_MANUALLY is True:
            tether = curves.generate_curve(
                env,
                init_point=env.anchor_point,
                check_self_intersection=True,
                show_goal=False,
                output_type="numpy",
            )
            print(tether)
        else:
            tether = np.array(
                [
                    [4.5, 3.5],
                    [4.11379658, 4.0020141],
                    [3.26787513, 4.33769721],
                    [2.5427996, 4.36455186],
                    [2.40852635, 4.78079893],
                    [2.18026183, 5.19704599],
                    [2.13997986, 5.5327291],
                ]
            )
        signature = curves.compute_signature(tether, env, simplify=True)
        robot = tether[-1]
        goal = np.array([1.0, 8.3])  # NOTE: change to select different goal
        env.robot_initial_pos = robot
        env.tether_state = tether
        env.tether_configuration = LineString(tether)
        env.goal_vertices = goal

    # Sanity check: verify that config is not entangled
    entanglement_function: Callable
    if definition == "convex_hull":
        entanglement_function = entanglement.convex_hull
    elif definition == "linear_homotopy":
        entanglement_function = entanglement.linear_homotopy
    elif definition == "local_visibility_homotopy":
        entanglement_function = entanglement.local_visibility_homotopy
    if entanglement_function(tether, env) is not True:
        raise ValueError(
            f"Tether configuration is entangled w.r.t. definiton {definition}"
        )

    # Load data
    # NOTE: the _entanglement data contain both R and N data structures
    triang: Triangulation = data["triangulation_entanglement"]
    graph: GridGraph = data["graph_2_entanglement"]
    triang: Triangulation

    # Perform path planning on triangulations
    # NOTE: planning on triang_R is performed only once as the simplicial complex
    # triang_R is the same independently from the entanglement definition, as the
    # definition is ignored when building them.
    # TODO: perform planning also for definition-free case (only during 1st loop)

    # Find triangles with goal, robot, and anchor in base triangulation
    triang_goal = int(
        triang.triang_tree.query(
            Point(goal),
            predicate="intersects",
        )[0]
    )
    triang_robot = int(
        triang.triang_tree.query(
            Point(robot),
            predicate="intersects",
        )[0]
    )
    triang_anchor = int(
        triang.triang_tree.query(
            Point(env.anchor_point),
            predicate="intersects",
        )[0]
    )

    # Find lifted copies of triangles with goal and robot, and list of copies of goal
    triang_goal_lift = [
        idx
        for idx, tri in enumerate(triang.vertices_dual_lift)
        if tri[0] == triang_goal and triang.entanglement_vertices_dual_lift[idx] is True
    ]
    triang_robot_lift = [
        idx
        for idx, tri in enumerate(triang.vertices_dual_lift)
        if ((tri[0] == triang_robot) and (tri[1] == signature))
    ][0]
    if triang.entanglement_vertices_dual_lift[triang_robot_lift] is not True:
        raise ValueError("Robot location is not in entanglement-free region")
    triang_anchor_lift = [
        idx
        for idx, tri in enumerate(triang.vertices_dual_lift)
        if ((tri[0] == triang_anchor) and (tri[1] == []))
    ][0]

    # Find geodesics in different homotopy classes
    path_length_list: list[float] = []
    path_list: list[list[np.ndarray]] = []
    comp_time: list[float] = []
    adj: dict[int, list[int]] = defaultdict(list)  # adjacency in ent-free dual graph
    for a, b in triang.edges_dual_lift:
        if (triang.entanglement_vertices_dual_lift[a] is True) and (
            triang.entanglement_vertices_dual_lift[b] is True
        ):
            adj[a].append(b)
            adj[b].append(a)
    for goal_lift in triang_goal_lift:
        t_init = time.process_time()
        stack = [(triang_robot_lift, [triang_robot_lift])]  # (node, path_to_node)
        visited = set()
        while stack:
            node, path = stack.pop()
            if node == goal_lift:
                break  # goal is reached
            if node in visited:
                continue
            visited.add(node)
            for neighbor in adj[node]:
                if neighbor not in visited:
                    stack.append((neighbor, path + [neighbor]))
        path = [triang.vertices_dual_lift[idx][0] for idx in path]
        path = triang.homotopic_shortest_path(
            alpha=path,
            p_init=robot,
            p_end=goal,
        )
        length = curves.measure_length(path)
        if length > length_max:
            raise ValueError(f"Tether is too long {length}>{length_max}")
        t_search = time.process_time() - t_init
        path_length_list.append(length)
        path_list.append(path)
        comp_time.append(t_search)
    description = f"N_{definition}"

    # Print stats + write them to file
    if len(comp_time) > 0:
        comp_time_mean = np.mean(comp_time)
        comp_time_std = np.std(comp_time)
        comp_time_max = np.max(comp_time)
    else:
        comp_time_mean = np.inf
        comp_time_std = np.inf
        comp_time_max = np.inf
    if len(path_length_list) > 0:
        path_length_mean = np.mean(path_length_list)
        path_length_std = np.std(path_length_list)
        path_length_max = np.max(path_length_list)
    else:
        path_length_mean = np.inf
        path_length_std = np.inf
        path_length_max = np.inf
    print(f"\nPath planning: {description}")
    print(
        f"Time stats: mean {comp_time_mean:.6f}, "
        f"std {comp_time_std:.6f}, "
        f"max {comp_time_max:.6f}"
    )
    print(
        f"Length stats: mean {path_length_mean:.6f}, "
        f"std {path_length_std:.6f}, "
        f"max {path_length_max:.6f}"
    )
    for i, (l, t) in list(enumerate(zip(path_length_list, comp_time))):
        print(f"\t#{i}\t length: {l:.2f}, time: {t:.6f}")
    with open("results/path-planning.txt", "a", encoding="utf-8") as f:
        f.write(f"\nPath planning: {description}\n")
        f.write(
            f"Time stats: mean {comp_time_mean:.6f}, "
            f"std {comp_time_std:.6f}, "
            f"max {comp_time_max:.6f}\n"
        )
        f.write(
            f"Length stats: mean {path_length_mean:.6f}, "
            f"std {path_length_std:.6f}, "
            f"max {path_length_max:.6f}\n"
        )
        for i, (l, t) in list(enumerate(zip(path_length_list, comp_time))):
            f.write(f"\t#{i}\t length: {l:.2f}, time: {t:.6f}\n")
    f.close()

    # Generate and save plot
    fig: plt.Figure
    ax: plt.Axes
    fig, ax = plot.plot_env(
        env,
        show_tether=True,
        show_robot=True,
        show_anchor=True,
        show_goal=True,
        show_legend=False,
        show_generators=False,
        show_curves_labels=False,
        show_robot_anchor_labels=False,
        show_generators_labels=False,
        show_obstacles_labels=False,
        show_axes_labels=False,
        figsize=[4, 4],
    )
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.plot(goal[0], goal[1], color="green", marker="o", markersize=2, zorder=10)
    for i, path in enumerate(path_list):
        ax.plot(path[:, 0], path[:, 1], color=colors[i % 10], zorder=8, linewidth=1)
    fig.savefig(f"results/{env_name}-pp-{description}.png", dpi=1200, format="png")

    # Perform path planning on h-augmented graphs
    # NOTE: similarly to planning on simplicial complexes, planning on lenght-reachable
    # h-augmented graph is performed only once
    # TODO: implement graph search
    #   - find lifted goal nodes
    #   - find lifted anchor
    #   - find lifted robot location (unique due to lifted anchor)
    #   - graph search from robot to goal (BFS, DFS, A*?)
    # TODO: print data
    # TODO: generate and save plots
    # TODO: generate and save plots
