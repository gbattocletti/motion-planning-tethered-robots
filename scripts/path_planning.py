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

import os
import pickle
import time

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from packaging import version
from shapely import Point

from tethered_planning.env import env_2d
from tethered_planning.env.grid_graph import GridGraph
from tethered_planning.env.triangulation import Triangulation

# from tethered_planning.plan import graph_search
from tethered_planning.utils import curves, plot
from tethered_planning.utils.colors import CmdColors
from tethered_planning.utils.settings import Settings

# Script settings
SAVE_PNG = True
SAVE_PGF = False
SAVE_PDF = False
env_filename = "env_4.yaml"
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

# Iterate over experiments and perform path planning
settings: Settings
env: env_2d.Env2D
anchor: np.ndarray
robot: np.ndarray
goal: np.ndarray
tether: np.ndarray
for idx, [filename, definition] in list(enumerate(experiments)):
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
        tether = curves.generate_curve(
            env,
            init_point=env.anchor_point,  # CHECKME
            check_self_intersection=True,
            show_goal=False,
            output_type="numpy",
        )
        print(tether)  # TODO: copy good one and hardcode the tether config
        robot = tether[-1]  # CHECKME check dimension
        goal = np.array([1.0, 8.3])  # TODO  select manually

    # Entanglement definition-specific data
    triang_R: Triangulation = data["triangulation"]  # length reachable model
    triang_N: Triangulation = data["triangulation_entanglement"]  # entang free model
    graph_R: GridGraph = data["graph_2"]
    graph_N: GridGraph = data["graph_2_entanglement"]

    # Perform path planning on triangulations
    # NOTE: planning on triang_R is performed only once as the simplicial complex
    # triang_R is the same independently from the entanglement definition, as the
    # definition is ignored when building them.
    triangulations: list[str, Triangulation] = ["N", triang_N]
    label: str
    triang: Triangulation
    if idx == 0:
        triangulations.insert(0, ["R", triang_R])
    for label, triang in triangulations:

        # Find triangles with goal, robot, and anchor
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

        # Find list of triangles that are lifted copies of the one with goal and robot
        triang_goal_lift = [
            idx
            for idx, tri in enumerate(triang.vertices_dual_lift)
            if tri[0] == triang_goal
        ]
        triang_robot_lift = [
            idx
            for idx, tri in enumerate(triang.vertices_dual_lift)
            if ((tri[0] == triang_robot) and (tri[1] == [-1]))
        ][0]
        triang_anchor_lift = [
            idx
            for idx, tri in enumerate(triang.vertices_dual_lift)
            if ((tri[0] == triang_anchor) and (tri[1] == []))
        ][0]

        # Find geodesics in different homotopy classes
        path_length_list: list[float] = []
        path_list: list[list[np.ndarray]] = []
        comp_time: list[float] = []
        for goal_lift in triang_goal_lift:
            t_init = time.process_time()
            length, path = triang.geodesic_distance(
                p1=goal,
                s1=triang.vertices_dual_lift[goal_lift][1],
                p2=robot,
                s2=[-1],
                search_algorithm="dfs",
            )
            t_search = time.process_time() - t_init
            path_length_list.append(length)
            path_list.append(path)
            comp_time.append(t_search)
        if label == "R":
            description = "R"
        else:
            description = f"N_{definition}"
        print(f"Path planning: {description}")
        print(
            f"Time stats: mean {np.mean(comp_time):.6f}, "
            f"std {np.std(comp_time):.6f}, "
            f"max {np.max(comp_time):.6f}"
        )
        print(
            f"Length stats: mean {np.mean(path_length_list):.6f}, "
            f"std {np.std(path_length_list):.6f}, "
            f"max {np.max(path_length_list):.6f}"
        )
        for i, (l, t) in list(enumerate(zip(path_length_list, comp_time))):
            print(f"\t#{i}\t length: {l:.2f}, time: {t:.6f}")

        # Generate and save plot
        fig: plt.Figure
        ax: plt.Axes
        fig, ax = plot.plot_env(
            env,
            show_tether=False,
            show_robot=True,
            show_anchor=True,
            show_goal=False,
            show_legend=False,
            show_generators=False,
            show_curves_labels=False,
            show_generators_labels=False,
            show_obstacles_labels=True,
            show_axes_labels=False,
            figsize=[4, 4],
        )
        ax.plot(goal[0], goal[1], color="green", marker="o", markersize=4, zorder=10)
        ax.text(0.7, 8.8, "$x_\\mathrm{{{g}}}$", fontsize=8, zorder=10)
        for path in path_list:
            ax.plot(path[:, 0], path[:, 1], color="blue", zorder=8, linewidth=1)
        fig.savefig(f"results/{description}.png", dpi=900, format="png")
        fig.savefig(f"results/{description}.svg")

    # Perform path planning on h-augmented graphs
    # NOTE: similarly to planning on simplicial complexes, planning on lenght-reachable
    # h-augmented graph is performed only once
    graphs: list[str, GridGraph] = ["N", graph_N]
    label: str
    graph: GridGraph
    if idx == 0:
        graphs.insert(0, ["R", graph_R])
    for label, graph in graphs:
        pass
        # TODO: implement graph search
        #   - find lifted goal nodes
        #   - find lifted anchor
        #   - find lifted robot location (unique due to lifted anchor)
        #   - graph search from robot to goal (BFS, DFS, A*?)
        # TODO: print data
        # TODO: generate and save plots
