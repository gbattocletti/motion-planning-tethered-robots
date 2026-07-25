"""
Generates a plot showing a disentanglement path computed from an entangled tether
configuration to a non entangled one. The script performs the path planning operation
for the three different entanglement definitions, and superimposes the resulting paths
on the same plot.
"""

import os
import pickle
from collections import defaultdict, deque

import matplotlib.pyplot as plt
import numpy as np
from shapely import Point

from tethered_planning.env import env_2d
from tethered_planning.env.triangulation import Triangulation
from tethered_planning.utils import curves, plot, plot_triangulation
from tethered_planning.utils.settings import Settings

# Script settings
SELECT_TETHER_MANUALLY: bool = True
env_name = "env_5"
length = 15.0

match (env_name, length):
    case ("env_3b", 10.0):
        experiments = [
            ["results/entanglement_free_model/comparison-19.pkl", "convex_hull"],
            ["results/entanglement_free_model/comparison-21.pkl", "local_visibility"],
        ]
    case ("env_3b", 12.5):
        experiments = [
            ["results/entanglement_free_model/comparison-22.pkl", "convex_hull"],
            ["results/entanglement_free_model/comparison-24.pkl", "local_visibility"],
        ]
    case ("env_3b", 15.0):
        experiments = [
            ["results/entanglement_free_model/comparison-25.pkl", "convex_hull"],
            ["results/entanglement_free_model/comparison-27.pkl", "local_visibility"],
        ]
    case ("env_3", 10.0):
        experiments = [
            ["results/entanglement_free_model/comparison-28.pkl", "convex_hull"],
            ["results/entanglement_free_model/comparison-30.pkl", "local_visibility"],
        ]
    case ("env_3", 12.5):
        experiments = [
            ["results/entanglement_free_model/comparison-31.pkl", "convex_hull"],
            ["results/entanglement_free_model/comparison-33.pkl", "local_visibility"],
        ]
    case ("env_3", 15.0):
        experiments = [
            ["results/entanglement_free_model/comparison-34.pkl", "convex_hull"],
            ["results/entanglement_free_model/comparison-36.pkl", "local_visibility"],
        ]
    case ("env_4", 10.0):
        experiments = [
            ["results/entanglement_free_model/comparison-37.pkl", "convex_hull"],
            ["results/entanglement_free_model/comparison-39.pkl", "local_visibility"],
        ]
    case ("env_4", 12.5):
        experiments = [
            ["results/entanglement_free_model/comparison-40.pkl", "convex_hull"],
            ["results/entanglement_free_model/comparison-42.pkl", "local_visibility"],
        ]
    case ("env_4", 15.0):
        experiments = [
            ["results/entanglement_free_model/comparison-43.pkl", "convex_hull"],
            ["results/entanglement_free_model/comparison-45.pkl", "local_visibility"],
        ]
    case ("env_5", 10.0):
        experiments = [
            ["results/entanglement_free_model/comparison-46.pkl", "convex_hull"],
            ["results/entanglement_free_model/comparison-48.pkl", "local_visibility"],
        ]
    case ("env_5", 12.5):
        experiments = [
            ["results/entanglement_free_model/comparison-49.pkl", "convex_hull"],
            ["results/entanglement_free_model/comparison-51.pkl", "local_visibility"],
        ]
    case ("env_5", 15.0):
        experiments = [
            ["results/entanglement_free_model/comparison-52.pkl", "convex_hull"],
            ["results/entanglement_free_model/comparison-54.pkl", "local_visibility"],
        ]
    case _:
        raise ValueError("Invalid combination of env and tether length.")

# Move to script directory
abspath = os.path.abspath(__file__)
dir_name = os.path.dirname(abspath)
os.chdir(dir_name)

# initialize useful data structures
settings: Settings
env: env_2d.Env2D
anchor: np.ndarray
robot: np.ndarray
goal: np.ndarray
tether: np.ndarray
fig_dfs: plt.Figure
ax_dfs: plt.Axes
fig_bfs: plt.Figure
ax_bfs: plt.Axes
colors = [
    "#006C74",
    "#720000",
]

# Iterate over entanglement definitions
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
        anchor = env.anchor_point
        if SELECT_TETHER_MANUALLY is True:
            tether = curves.generate_curve(
                env,
                init_point=env.anchor_point,
                check_self_intersection=False,
                max_points=50,
                show_goal=False,
                output_type="numpy",
            )
            print(tether)
        else:
            tether = np.array(
                [
                    [5.0, 4.5],
                    [4.78516281, 4.80765358],
                    [4.54347096, 5.10305472],
                    [4.1809332, 5.35817388],
                    [3.57670359, 5.49244713],
                    [3.18731118, 5.50587445],
                    [2.66364552, 5.35817388],
                    [2.19368916, 5.39845586],
                    [1.37462236, 5.45216516],
                    [0.89123867, 5.61329305],
                    [0.54212823, 5.94897617],
                    [0.46156428, 6.52635112],
                    [0.62269218, 7.06344411],
                    [0.93152064, 7.4394092],
                    [1.36119503, 7.68110104],
                    [2.05941591, 7.80194696],
                    [2.69050017, 7.66767372],
                    [3.46928499, 7.66767372],
                    [4.16750587, 7.61396442],
                    [5.21483719, 7.64081907],
                    [5.64451158, 7.88251091],
                    [6.10104062, 8.11077543],
                    [6.55756965, 8.40617657],
                    [7.20208124, 8.50016784],
                    [7.63175562, 8.35246727],
                    [8.07485733, 8.0973481],
                    [8.16884861, 7.78851964],
                ]
            )
        if curves.measure_length(tether) > length:
            raise ValueError(
                f"Curve length {curves.measure_length(tether)} is larger than the "
                f"maximum tether length {length}."
            )
        signature = curves.compute_signature(tether, env, simplify=True)
        robot = tether[-1]
        env.robot_initial_pos = robot
        env.tether_state = tether
        # NOTE: no goal is selected as objective is to reach N_bar from R_bar

        # Initialize plots
        fig_dfs, ax_dfs = plot.plot_env(
            env,
            show_tether=True,
            tether=tether,
            show_robot=True,
            show_anchor=True,
            show_goal=False,
            show_legend=False,
            show_generators=False,
            show_curves_labels=False,
            show_generators_labels=False,
            show_robot_anchor_labels=False,
            show_obstacles_labels=False,
            show_axes_labels=False,
            figsize=[4, 4],
        )
        ax_dfs.set_xlabel("")
        ax_dfs.set_ylabel("")
        ax_dfs.set_xticklabels([])
        ax_dfs.set_yticklabels([])
        fig_bfs, ax_bfs = plot.plot_env(
            env,
            show_tether=True,
            tether=tether,
            show_robot=True,
            show_anchor=True,
            show_goal=False,
            show_legend=False,
            show_generators=False,
            show_curves_labels=False,
            show_generators_labels=False,
            show_robot_anchor_labels=False,
            show_obstacles_labels=False,
            show_axes_labels=False,
            figsize=[4, 4],
        )
        ax_bfs.set_xlabel("")
        ax_bfs.set_ylabel("")
        ax_bfs.set_xticklabels([])
        ax_bfs.set_yticklabels([])

    # RUN PATH PLANNING (ONCE PER ENTANGLEMENT DEFINITION)
    # Definition-specific data
    triang: Triangulation = data["triangulation_entanglement"]  # entang free model
    unique_sign_list = plot_triangulation.get_unique_signatures(triang)
    unique_sign_n = len(unique_sign_list)

    # Sanity check: does robot lie in entanglement-free region
    triang_robot = int(
        triang.triang_tree.query(
            Point(robot),
            predicate="intersects",
        )[0]
    )
    triangs_not_entangled = [
        tri
        for tri, is_not_entangled in zip(
            triang.vertices_dual_lift, triang.entanglement_vertices_dual_lift
        )
        if is_not_entangled is True
    ]
    triang_robot_lift = [
        idx
        for idx, tri in enumerate(triangs_not_entangled)
        if ((tri[0] == triang_robot) and (tri[1] == signature))
    ]
    if not triang_robot_lift == []:
        raise ValueError(
            "The initial tether configuration is not entangled w.r.t. the entanglement "
            f"definition {definition}."
        )

    # Find lifted location of robot
    if signature not in unique_sign_list:
        raise ValueError("Robot location is not in simplicial complex.")
    else:
        triang_robot_lift = [
            idx
            for idx, tri in enumerate(triang.vertices_dual_lift)
            if ((tri[0] == triang_robot) and (tri[1] == signature))
        ]
        if len(triang_robot_lift) > 1:
            raise ValueError("Robot lifts to multiple triangles.")
        else:
            triang_robot_lift = triang_robot_lift[0]

    # Initialization of graph search
    adj: dict[int, list[int]] = defaultdict(list)
    for a, b in triang.edges_dual_lift:
        adj[a].append(b)
        adj[b].append(a)

    # Perform DFS path planning on triangulations to go from R to N
    print("Running DFS...")
    stack = [(triang_robot_lift, [triang_robot_lift])]  # (node, path_to_node)
    visited = set()
    while stack:
        node, path_indexes = stack.pop()
        if triang.entanglement_vertices_dual_lift[node] is True:
            break  # entanglement-free region reached
        if node in visited:
            continue
        visited.add(node)
        for neighbor in adj[node]:
            if neighbor not in visited:
                stack.append((neighbor, path_indexes + [neighbor]))
    path_indexes = [triang.vertices_dual_lift[idx][0] for idx in path_indexes]
    path_geom = triang.homotopic_shortest_path(
        alpha=path_indexes,
        p_init=robot,
        p_end=triang.vertices_dual[path_indexes[-1]],
    )
    ax_dfs.plot(
        path_geom[:, 0],
        path_geom[:, 1],
        color=colors[idx],
        zorder=8,
        linewidth=1,
    )

    # Perform BFS path planning on triangulations to go from R to N
    print("Running BFS...")
    queue = deque([(triang_robot_lift, [triang_robot_lift])])  # (node, path_to_node)
    visited = set([triang_robot_lift])
    while queue:
        node, path_indexes = queue.popleft()
        if triang.entanglement_vertices_dual_lift[node] is True:
            break  # entanglement-free region reached
        for neighbor in adj[node]:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, path_indexes + [neighbor]))
    path_indexes = [triang.vertices_dual_lift[idx][0] for idx in path_indexes]
    path_geom = triang.homotopic_shortest_path(
        alpha=path_indexes,
        p_init=robot,
        p_end=triang.vertices_dual[path_indexes[-1]],
    )
    ax_bfs.plot(
        path_geom[:, 0],
        path_geom[:, 1],
        color=colors[idx],
        zorder=8,
        linewidth=1,
    )

# Save figures
fig_dfs.savefig(
    f"results/{env_name}-disentanglement-dfs.png",
    dpi=1200,
    format="png",
)
fig_bfs.savefig(
    f"results/{env_name}-disentanglement-bfs.png",
    dpi=1200,
    format="png",
)
