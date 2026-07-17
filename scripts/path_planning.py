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
from collections import defaultdict, deque
from collections.abc import Callable

import matplotlib.pyplot as plt
import numpy as np
from shapely import LineString, Point

from tethered_planning.env import env_2d
from tethered_planning.env.grid_graph import GridGraph
from tethered_planning.env.triangulation import Triangulation

# from tethered_planning.plan import graph_search
from tethered_planning.utils import curves, entanglement, plot
from tethered_planning.utils.settings import Settings

# Script settings
SELECT_TETHER_MANUALLY: bool = True
env_name = "env_4"  # NOTE: currently only env_4 is supported in this script
goal = np.array([1.0, 8.5])  # NOTE: change to select different goal
goal = np.array([2.0, 6.5])
length_max = 12  # current available options: {10, 12, 15}

# Select experiments
experiments: list[list[str]] = []
if env_name == "env_4" and length_max == 10:
    experiments = [
        ["results/entanglement_free_model/comparison-25.pkl", "none"],
        ["results/entanglement_free_model/comparison-25.pkl", "convex_hull"],
        ["results/entanglement_free_model/comparison-26.pkl", "linear_homotopy"],
        ["results/entanglement_free_model/comparison-27.pkl", "local_visibility"],
    ]
elif env_name == "env_4" and length_max == 12:
    experiments = [
        ["results/entanglement_free_model/comparison-28.pkl", "none"],
        ["results/entanglement_free_model/comparison-28.pkl", "convex_hull"],
        ["results/entanglement_free_model/comparison-29.pkl", "linear_homotopy"],
        ["results/entanglement_free_model/comparison-30.pkl", "local_visibility"],
    ]
elif env_name == "env_4" and length_max == 15:
    experiments = [
        ["results/entanglement_free_model/comparison-31.pkl", "none"],
        ["results/entanglement_free_model/comparison-31.pkl", "convex_hull"],
        ["results/entanglement_free_model/comparison-32.pkl", "linear_homotopy"],
        ["results/entanglement_free_model/comparison-33.pkl", "local_visibility"],
    ]

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
for filename, definition in experiments:

    # Load data
    objects: list = []
    with open(filename, "rb") as openfile:
        while True:
            try:
                objects.append(pickle.load(openfile))
            except EOFError:
                break
    data: dict = objects[0]
    if definition == "none":  # 1st loop
        settings = data["settings"]
        env = data["env"]
        env.goal_vertices = goal
        if SELECT_TETHER_MANUALLY is True:
            tether = curves.generate_curve(
                env,
                init_point=env.anchor_point,
                check_self_intersection=True,
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
                    [4.11379658, 4.0020141],
                    [3.26787513, 4.33769721],
                    [2.5427996, 4.36455186],
                    [2.40852635, 4.78079893],
                    [2.18026183, 5.19704599],
                    [2.13997986, 5.5327291],
                ]
            )
        signature = curves.compute_signature(tether, env, simplify=True)
        length_curve = curves.measure_length(tether)
        if length_curve > length_max:
            raise ValueError(f"Tether is too long {length_curve}>{length_max}")
        robot = tether[-1]
        anchor = env.anchor_point

        # Update env
        env.robot_initial_pos = robot
        env.tether_state = tether
        env.tether_configuration = LineString(tether)
        env.tether_length = length_max
        env.goal_vertices = goal

    # Sanity check: verify that config is not entangled
    entanglement_function: Callable
    if definition == "convex_hull":
        entanglement_function = entanglement.convex_hull
    elif definition == "linear_homotopy":
        entanglement_function = entanglement.linear_homotopy
    elif definition == "local_visibility_homotopy":
        entanglement_function = entanglement.local_visibility_homotopy
    if definition == "none":
        pass  # no need to evaluate the entanglement function
    elif entanglement_function(tether, env) is not True:
        raise ValueError(
            f"Tether configuration is entangled w.r.t. definiton {definition}"
        )

    # Load data
    # NOTE: the _entanglement data contain both R and N data structures
    triang: Triangulation = data["triangulation_entanglement"]
    graph: GridGraph = data["graph_2_entanglement"]
    triang: Triangulation
    graph: GridGraph

    ####################################################################################
    # Perform path planning on simplicial complex model
    path_length_list: list[float] = []
    path_list: list[list[np.ndarray]] = []
    comp_time: list[float] = []

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
    if definition == "none":
        triang_goal_lift = [
            idx
            for idx, tri in enumerate(triang.vertices_dual_lift)
            if tri[0] == triang_goal
        ]
    else:
        triang_goal_lift = [
            idx
            for idx, tri in enumerate(triang.vertices_dual_lift)
            if tri[0] == triang_goal
            and triang.entanglement_vertices_dual_lift[idx] is True
        ]
    triang_anchor_lift = [
        idx
        for idx, tri in enumerate(triang.vertices_dual_lift)
        if ((tri[0] == triang_anchor) and (tri[1] == []))
    ][0]
    triang_robot_lift = [
        idx
        for idx, tri in enumerate(triang.vertices_dual_lift)
        if ((tri[0] == triang_robot) and (tri[1] == signature))
    ][0]

    # Build adjacency dictionary (data structure used in graph search)
    adj: dict[int, list[int]] = defaultdict(list)
    if definition == "none":
        # Build adjacency dictionary in lifted dual graph
        for a, b in triang.edges_dual_lift:
            adj[a].append(b)
            adj[b].append(a)
    else:
        # Build adjacency dictionary in entanglement-free lifted dual graph
        for a, b in triang.edges_dual_lift:
            if (triang.entanglement_vertices_dual_lift[a] is True) and (
                triang.entanglement_vertices_dual_lift[b] is True
            ):
                adj[a].append(b)
                adj[b].append(a)

    # Perform path planning (DFS graph search)
    for goal_lift in triang_goal_lift:
        t_init = time.process_time()
        sol_found = False
        stack = [(triang_robot_lift, [triang_robot_lift])]  # (node, path_to_node)
        visited = set()
        while stack:
            node, path = stack.pop()
            if node == goal_lift:
                sol_found = True
                break  # goal is reached
            if node in visited:
                continue
            visited.add(node)
            for neighbor in adj[node]:
                if neighbor not in visited:
                    stack.append((neighbor, path + [neighbor]))
        if sol_found is True:
            path = [triang.vertices_dual_lift[idx][0] for idx in path]
            path = triang.homotopic_shortest_path(
                alpha=path,
                p_init=robot,
                p_end=goal,
            )
            t_search = time.process_time() - t_init
            comp_time.append(t_search)
            length = curves.measure_length(path)
            path_length_list.append(length)
            path_list.append(path)
        else:
            pass  # open queue got emptied without finding a solution # CHECKME

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
    print(f"\nPath planning: (def: {definition} | simplicial complex model)")
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
        f.write(f"\nPath planning: (def: {definition} | simplicial complex model)\n")
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
    fig.savefig(f"results/{env_name}-pp-{definition}-sc.png", dpi=1200, format="png")

    ####################################################################################
    # Perform path planning on h-augmented graphs
    path_length_list: list[float] = []
    path_list: list[list[np.ndarray]] = []
    comp_time: list[float] = []

    # Find relevant points in base graph
    node_goal_idx = np.argmin(np.sum((graph.vertices - goal) ** 2, axis=1))
    node_anchor_idx = np.argmin(np.sum((graph.vertices - anchor) ** 2, axis=1))
    node_robot_idx = np.argmin(np.sum((graph.vertices - robot) ** 2, axis=1))

    #  Find lifted copies of relevant points
    if definition == "none":
        node_goal_lift_idx = [
            idx
            for idx, node in enumerate(graph.vertices_lift)
            if node[0] == node_goal_idx
        ]
    else:
        node_goal_lift_idx = [
            idx
            for idx, node in enumerate(graph.vertices_lift)
            if node[0] == node_goal_idx
            and graph.entanglement_vertices_lift[idx] is True
        ]
    graph_anchor_lift_idx = [
        idx
        for idx, node in enumerate(graph.vertices_lift)
        if ((node[0] == node_anchor_idx) and (node[1] == []))
    ][0]
    node_robot_lift_idx = [
        idx
        for idx, node in enumerate(graph.vertices_lift)
        if ((node[0] == node_robot_idx) and (node[1] == signature))
    ][0]

    # Build adjacency dictionary (data structure used in graph search)
    adj: dict[int, list[int]] = defaultdict(list)
    if definition == "none":
        # Build adjacency dictionary in lifted dual graph
        for a, b in graph.edges_lift:
            adj[a].append(b)
            adj[b].append(a)
    else:
        # Build adjacency dictionary in entanglement-free lifted dual graph
        for a, b in graph.edges_lift:
            if (graph.entanglement_vertices_lift[a] is True) and (
                graph.entanglement_vertices_lift[b] is True
            ):
                adj[a].append(b)
                adj[b].append(a)

    # Perform path planning (BFS graph search)
    for node_goal_lift in node_goal_lift_idx:
        t_init = time.process_time()
        sol_found = False
        queue = deque(
            [(node_robot_lift_idx, [node_robot_lift_idx])]
        )  # (node_idx, idx_path_to_node)
        visited = set([node_robot_lift_idx])
        while queue:
            node, path = queue.popleft()
            if node == node_goal_lift:
                sol_found = True
                break
            for neighbor in adj[node]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, path + [neighbor]))
        if sol_found is True:
            path = np.array(
                [graph.vertices[graph.vertices_lift[idx][0]] for idx in path]
            )
            t_search = time.process_time() - t_init
            comp_time.append(t_search)
            length = curves.measure_length(path)
            path_length_list.append(length)
            path_list.append(path)
        else:
            pass  # open list got emptied without finding a valid solution # CHECKME

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
    print(f"\nPath planning: (def: {definition} | h-augmented graph)")
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
        f.write(f"\nPath planning: (def: {definition} | h-augmented graph)\n")
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
    fig.savefig(f"results/{env_name}-pp-{definition}-graph.png", dpi=1200, format="png")
