"""
Performs path planning between the anchor point and a random location in the
environment, and measures the computation time required to perform the graph search
on the simplicial complex model and on the h-augmented graph.
"""

import datetime
import os
import pickle
import time
from collections import defaultdict, deque

import matplotlib.pyplot as plt
import numpy as np
from shapely import Point

from tethered_planning.env import env_2d
from tethered_planning.env.grid_graph import GridGraph
from tethered_planning.env.triangulation import Triangulation
from tethered_planning.utils import curves, plot
from tethered_planning.utils.settings import Settings

# Script settings
SELECT_TETHER_MANUALLY: bool = True
env_name = "env_5"
length_max = 15.0  # available options: {10, 12.5, 15}
n_experiments_tot: int = 20

########################################################################################

# Move to script directory
abspath = os.path.abspath(__file__)
dir_name = os.path.dirname(abspath)
os.chdir(dir_name)

# Load data structures
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

# Reset log file
with open("results/path-planning-timing.csv", "w", encoding="utf-8") as f:
    print(f"[Started at: {datetime.datetime.now()}]")
    print(f"\nPath planning env {env_name} l={length_max}:")
    f.write(f"Path planning env {env_name} l={length_max}:\n")
    f.write(
        "n_sc, t_avg_sc, t_std_sc, t_max_sc, l_avg_sc, l_std_sc, l_max_sc, "
        "n_50, t_avg_50, t_std_50, t_max_50, l_avg_50, l_std_50, l_max_50, "
        "n_25, t_avg_25, t_std_25, t_max_25, l_avg_25, l_std_25, l_max_25 \n"
    )

    # Load data
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
    anchor: np.ndarray = env.anchor_point

    # Load data
    triang: Triangulation = data["triangulation_entanglement"]
    graph_50: GridGraph = data["graph_1_entanglement"]
    graph_25: GridGraph = data["graph_2_entanglement"]
    triang: Triangulation
    graph_50: GridGraph
    graph_25: GridGraph

    # Plot env
    fig: plt.Figure
    ax: plt.Axes
    fig, ax = plot.plot_env(
        env,
        show_tether=False,
        show_robot=False,
        show_anchor=True,
        show_goal=False,
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

    # Iterate over goals
    n_experiments: int = 0
    while n_experiments < n_experiments_tot:

        # Sample goal
        goal: np.ndarray = np.random.uniform(low=0.0, high=10.0, size=(1, 2))[0]
        if env.is_valid_point(goal[0], goal[1], invalid_boundary=True):
            n_experiments = n_experiments + 1  # goal is ok, perform path planning
        else:
            continue  # skip iteration and sample new point
        print(f"\n[# {n_experiments}] Goal: {goal}")

        # Initialize data structures
        path_length_sc: list[float] = []
        comp_time_sc: list[float] = []
        path_length_hag_50: list[float] = []
        comp_time_hag_50: list[float] = []
        path_length_hag_25: list[float] = []
        comp_time_hag_25: list[float] = []

        # Motion planning on simplicial complex model ##################################
        # Find triangles with goal, robot, and anchor in base triangulation
        triang_goal = int(
            triang.triang_tree.query(
                Point(goal),
                predicate="intersects",
            )[0]
        )
        triang_anchor = int(
            triang.triang_tree.query(
                Point(env.anchor_point),
                predicate="intersects",
            )[0]
        )
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

        # Build adjacency dictionary (data structure used in graph search)
        adj: dict[int, list[int]] = defaultdict(list)
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
            stack = [(triang_anchor_lift, [triang_anchor_lift])]  # (node, path_to_node)
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
                    p_init=anchor,
                    p_end=goal,
                )
                t_search = time.process_time() - t_init
                length = curves.measure_length(path)
                comp_time_sc.append(t_search)
                path_length_sc.append(length)
            else:
                pass  # no solution exists

        # Print stats + write them to file
        if len(comp_time_sc) > 0:
            comp_time_mean_sc = np.mean(comp_time_sc)
            comp_time_std_sc = np.std(comp_time_sc)
            comp_time_max_sc = np.max(comp_time_sc)
        else:
            comp_time_mean_sc = np.inf
            comp_time_std_sc = np.inf
            comp_time_max_sc = np.inf
        if len(path_length_sc) > 0:
            path_length_mean_sc = np.mean(path_length_sc)
            path_length_std_sc = np.std(path_length_sc)
            path_length_max_sc = np.max(path_length_sc)
        else:
            path_length_mean_sc = np.inf
            path_length_std_sc = np.inf
            path_length_max_sc = np.inf
        print("Simplicial complex model")
        print(f"\tNumber of paths: {len(comp_time_sc)}")
        print(
            f"\tTime stats: mean {comp_time_mean_sc:.6f}, "
            f"std {comp_time_std_sc:.6f}, "
            f"max {comp_time_max_sc:.6f}"
        )
        print(
            f"\tLength stats: mean {path_length_mean_sc:.6f}, "
            f"std {path_length_std_sc:.6f}, "
            f"max {path_length_max_sc:.6f}"
        )

        # Path planning on h-augmented graphs with resolution 0.50 #####################
        # Find relevant points in base graph
        node_goal_idx = np.argmin(np.sum((graph_50.vertices - goal) ** 2, axis=1))
        node_anchor_idx = np.argmin(np.sum((graph_50.vertices - anchor) ** 2, axis=1))

        #  Find lifted copies of relevant points
        node_goal_lift_idx = [
            idx
            for idx, node in enumerate(graph_50.vertices_lift)
            if node[0] == node_goal_idx
            and graph_50.entanglement_vertices_lift[idx] is True
        ]
        graph_anchor_lift_idx = [
            idx
            for idx, node in enumerate(graph_50.vertices_lift)
            if ((node[0] == node_anchor_idx) and (node[1] == []))
        ][0]

        # Build adjacency dictionary (data structure used in graph search)
        adj: dict[int, list[int]] = defaultdict(list)
        for a, b in graph_50.edges_lift:
            if (graph_50.entanglement_vertices_lift[a] is True) and (
                graph_50.entanglement_vertices_lift[b] is True
            ):
                adj[a].append(b)
                adj[b].append(a)

        # Perform path planning (BFS graph search)
        for node_goal_lift in node_goal_lift_idx:
            t_init = time.process_time()
            sol_found = False
            queue = deque(
                [(graph_anchor_lift_idx, [graph_anchor_lift_idx])]
            )  # (node_idx, idx_path_to_node)
            visited = set([graph_anchor_lift_idx])
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
                    [graph_50.vertices[graph_50.vertices_lift[idx][0]] for idx in path]
                )
                t_search = time.process_time() - t_init
                comp_time_hag_50.append(t_search)
                path_length_hag_50.append(curves.measure_length(path))
            else:
                pass  # open list got emptied without finding a valid solution

        # Print stats + write them to file
        if len(comp_time_hag_50) > 0:
            comp_time_mean_hag_50 = np.mean(comp_time_hag_50)
            comp_time_std_hag_50 = np.std(comp_time_hag_50)
            comp_time_max_hag_50 = np.max(comp_time_hag_50)
        else:
            comp_time_mean_hag_50 = np.inf
            comp_time_std_hag_50 = np.inf
            comp_time_max_hag_50 = np.inf
        if len(path_length_hag_50) > 0:
            path_length_mean_hag_50 = np.mean(path_length_hag_50)
            path_length_std_hag_50 = np.std(path_length_hag_50)
            path_length_max_hag_50 = np.max(path_length_hag_50)
        else:
            path_length_mean_hag_50 = np.inf
            path_length_std_hag_50 = np.inf
            path_length_max_hag_50 = np.inf
        print("Homotopy augmented graph 0.50")
        print(f"\tNumber of paths: {len(comp_time_hag_50)}")
        print(
            f"\tTime stats: mean {comp_time_mean_hag_50:.6f}, "
            f"std {comp_time_std_hag_50:.6f}, "
            f"max {comp_time_max_hag_50:.6f}"
        )
        print(
            f"\tLength stats: mean {path_length_mean_hag_50:.6f}, "
            f"std {path_length_std_hag_50:.6f}, "
            f"max {path_length_max_hag_50:.6f}"
        )

        # Path planning on h-augmented graphs with resolution 0.25 #####################
        # Find relevant points in base graph
        node_goal_idx = np.argmin(np.sum((graph_25.vertices - goal) ** 2, axis=1))
        node_anchor_idx = np.argmin(np.sum((graph_25.vertices - anchor) ** 2, axis=1))

        #  Find lifted copies of relevant points
        node_goal_lift_idx = [
            idx
            for idx, node in enumerate(graph_25.vertices_lift)
            if node[0] == node_goal_idx
            and graph_25.entanglement_vertices_lift[idx] is True
        ]
        graph_anchor_lift_idx = [
            idx
            for idx, node in enumerate(graph_25.vertices_lift)
            if ((node[0] == node_anchor_idx) and (node[1] == []))
        ][0]

        # Build adjacency dictionary (data structure used in graph search)
        adj: dict[int, list[int]] = defaultdict(list)
        for a, b in graph_25.edges_lift:
            if (graph_25.entanglement_vertices_lift[a] is True) and (
                graph_25.entanglement_vertices_lift[b] is True
            ):
                adj[a].append(b)
                adj[b].append(a)

        # Perform path planning (BFS graph search)
        for node_goal_lift in node_goal_lift_idx:
            t_init = time.process_time()
            sol_found = False
            queue = deque(
                [(graph_anchor_lift_idx, [graph_anchor_lift_idx])]
            )  # (node_idx, idx_path_to_node)
            visited = set([graph_anchor_lift_idx])
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
                    [graph_25.vertices[graph_25.vertices_lift[idx][0]] for idx in path]
                )
                t_search = time.process_time() - t_init
                comp_time_hag_25.append(t_search)
                path_length_hag_25.append(curves.measure_length(path))
            else:
                pass  # open list got emptied without finding a valid solution

        # Print stats + write them to file
        if len(comp_time_hag_25) > 0:
            comp_time_mean_hag_25 = np.mean(comp_time_hag_25)
            comp_time_std_hag_25 = np.std(comp_time_hag_25)
            comp_time_max_hag_25 = np.max(comp_time_hag_25)
        else:
            comp_time_mean_hag_25 = np.inf
            comp_time_std_hag_25 = np.inf
            comp_time_max_hag_25 = np.inf
        if len(path_length_hag_25) > 0:
            path_length_mean_hag_25 = np.mean(path_length_hag_25)
            path_length_std_hag_25 = np.std(path_length_hag_25)
            path_length_max_hag_25 = np.max(path_length_hag_25)
        else:
            path_length_mean_hag_25 = np.inf
            path_length_std_hag_25 = np.inf
            path_length_max_hag_25 = np.inf
        print("Homotopy augmented graph 0.25")
        print(f"\tNumber of paths: {len(comp_time_hag_25)}")
        print(
            f"\tTime stats: mean {comp_time_mean_hag_25:.6f}, "
            f"std {comp_time_std_hag_25:.6f}, "
            f"max {comp_time_max_hag_25:.6f}"
        )
        print(
            f"\tLength stats: mean {path_length_mean_hag_25:.6f}, "
            f"std {path_length_std_hag_25:.6f}, "
            f"max {path_length_max_hag_25:.6f}"
        )

        # Update csv
        f.write(
            f"{len(comp_time_sc)}, "
            f"{comp_time_mean_sc:.4f}, "
            f"{comp_time_std_sc:.4f}, "
            f"{comp_time_max_sc:.4f}, "
            f"{path_length_mean_sc:.4f}, "
            f"{path_length_std_sc:.4f}, "
            f"{path_length_max_sc:.4f}, "
            f"{len(comp_time_hag_50)}, "
            f"{comp_time_mean_hag_50:.4f}, "
            f"{comp_time_std_hag_50:.4f}, "
            f"{comp_time_max_hag_50:.4f}, "
            f"{path_length_mean_hag_50:.4f}, "
            f"{path_length_std_hag_50:.4f}, "
            f"{path_length_max_hag_50:.4f}, "
            f"{len(comp_time_hag_25)}, "
            f"{comp_time_mean_hag_25:.4f}, "
            f"{comp_time_std_hag_25:.4f}, "
            f"{comp_time_max_hag_25:.4f}, "
            f"{path_length_mean_hag_25:.4f}, "
            f"{path_length_std_hag_25:.4f}, "
            f"{path_length_max_hag_25:.4f} \n"
        )

f.close()

# Print summary
data_csv = np.loadtxt(
    "results/path-planning-timing.csv",
    delimiter=",",
    skiprows=2,
)  # Load full csv data
n_rows, n_cols = data_csv.shape
print()
arr_sc = [data_csv[i][1] for i in range(n_rows)]
print(
    f"time sc: {np.nanmean(np.where(np.isinf(arr_sc), np.nan, arr_sc)):.4f} "
    f"+ {np.nanstd(np.where(np.isinf(arr_sc), np.nan, arr_sc)):.4f}"
)
print(
    f"time hag_50: {np.nanmean([data_csv[i][8] for i in range(n_rows)]):.4f} "
    f"+ {np.nanstd([data_csv[i][8] for i in range(n_rows)]):.4f}"
)
print(
    f"time hag_25: {np.nanmean([data_csv[i][15] for i in range(n_rows)]):.4f}"
    f"{np.nanstd([data_csv[i][15] for i in range(n_rows)]):.4f}"
)
