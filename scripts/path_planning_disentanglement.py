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
env_name = "env_4"
length = 15.0
experiments = [
    ["results/entanglement_free_model/comparison-34.pkl", "convex_hull"],
    # ["results/entanglement_free_model/comparison-35.pkl", "linear_homotopy"],
    # ["results/entanglement_free_model/comparison-36.pkl", "local_visibility"],
]
SELECT_TETHER_MANUALLY: bool = False

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
    "#755000",
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
                check_self_intersection=True,
                show_goal=False,
                output_type="numpy",
            )
            print(tether)
        else:
            tether = np.array(
                [
                    [4.5, 3.5],
                    [4.22121517, 4.06915072],
                    [4.38234307, 4.75394428],
                    [4.36891574, 5.7475663],
                    [4.1272239, 6.31151393],
                    [3.4155757, 6.53977845],
                    [2.67707284, 6.13695871],
                    [2.435381, 5.30446459],
                    [2.42195368, 4.49882511],
                    [1.69687815, 4.49882511],
                    [0.83752937, 4.75394428],
                    [0.6361195, 5.5864384],
                    [0.94494797, 6.25780463],
                    [0.75696542, 7.03658946],
                    [0.71668345, 7.84222894],
                    [0.93152064, 8.32561262],
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
            show_obstacles_labels=True,
            show_axes_labels=False,
            figsize=[8, 8],
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
            show_obstacles_labels=True,
            show_axes_labels=False,
            figsize=[8, 8],
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
