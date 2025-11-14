"""
Generates the following plots
- an atlas for the environment defined in env_3.yaml where all the layers are shown.
- a plot showing a path planning scenario in multiple homotopy classes between the robot
and the goal are evaluated in parallel. The robot's paths and resulting tether
configurations are shown.

Some elements in this script, such as the label and some points that are added to the
plots, are tailored to the specific environment defined in env_3.yaml. If the
environment is changed, these elements may need to be adjusted accordingly.
"""

import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from packaging import version
from shapely import Point

from tethered_planning.env import env_2d
from tethered_planning.env.triangulation import Triangulation
from tethered_planning.plan import graph_search
from tethered_planning.utils import plot, plot_triangulation
from tethered_planning.utils.colors import CmdColors
from tethered_planning.utils.settings import Settings

# Script settings
SAVE_PNG = True
SAVE_PGF = True
SHOW_PLOTS = False
filename = "env_3.yaml"  # set to None to open file dialog and select manually

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

# Create settings and env objects
settings = Settings(create_sim_folder=False)
settings.env_name = filename
env = env_2d.Env2D(settings)

# Generate triangulation and lift it
triang = Triangulation(env)
triang.triangulate()
triang.max_lifted_triangles = 1000  # high value to avoid cutting off layers
triang.set_max_dist(12.0)  # max tether length
triang.lift_triangulation()

# Plot triangulation
fig, _ = plot.plot_graph(
    triang.vertices,
    triang.edges,
    env,
    nodes_dual=triang.vertices_dual,
    edges_dual=triang.edges_dual,
    show_dual_graph=True,
    label_nodes=False,
    label_triangles=True,
    show_generators=False,
    show_generators_labels=False,
    show_legend=False,
    show_obstacles_labels=True,
    show_anchor=True,
    show_robot=False,
    show_tether=False,
    figsize=[4.5, 4.5],
)
if SHOW_PLOTS is True:
    plt.show(block=False)

# Initial conditions for path planning
robot = [2, 2]
tether = np.array(
    [
        [5.0, 4.5],
        [5.0, 4.4],
        [5.0, 4.3],
        [5.0, 4.2],
        [4.8, 4.1],
        [4.5, 4.0],
        [4.3, 3.8],
        [4.0, 3.7],
        [3.8, 3.6],
        [3.5, 3.5],
        [3.2, 3.2],
        [3.0, 3.1],
        [2.8, 3.0],
        [2.6, 2.8],
        [2.5, 2.6],
        [2.4, 2.5],
        [2.3, 2.4],
        [2.2, 2.3],
        [2.1, 2.2],
        [2.0, 2.1],
        [2.0, 2.0],
    ]
)
goal = np.array([1.0, 8.3])
env.robot_initial_pos = robot
env.tether_state = tether

# Generate 2D plot of the triangulation layers - row 1
fig, ax = plot_triangulation.plot_2d(
    triang,
    env,
    max_cols=5,
    custom_sign_order=[
        [],
        [6],
        [-4],
        [-1],
    ],
    add_env_subplot=True,
    show_obstacles=False,
    figsize=[18, 4.3],
)

# Manually update env subplot to show robot, goal, tether, and anchor
plot.plot_env(
    env,
    show_generators=True,
    show_generators_labels=True,
    show_anchor=True,
    show_tether=True,
    tether=tether,
    show_robot=True,
    show_goal=False,
    show_legend=False,
    show_axes_labels=False,
    target_ax=ax[0],
)
ax[0].plot(
    goal[0],
    goal[1],
    color="green",
    marker="o",
    markersize=4,
    zorder=10,
)
ax[0].text(
    0.6,
    8.8,
    "$x_\\mathrm{{{g}}}$",
    fontsize=8,
    zorder=10,
)
ax[0].text(
    0.8,
    -1,
    "env with $n=7, m=6$",
    fontsize=8,
    zorder=10,
)

# Manually add lifted points in the atlas
x_text = 0.7
y_text = 8.8
ax[4].plot(
    goal[0],
    goal[1],
    color="green",
    marker="o",
    markersize=4,
    zorder=10,
)
ax[4].text(
    x_text,
    y_text,
    "$\\tilde{{{x}}}_{{{\\mathrm{{{g}}}, 2}}}$",
    fontsize=8,
    zorder=10,
)
ax[3].plot(
    goal[0],
    goal[1],
    color="green",
    marker="o",
    markersize=4,
    zorder=10,
)
ax[3].text(
    x_text,
    y_text,
    "$\\tilde{{{x}}}_{{{\\mathrm{{{g}}}, 5}}}$",
    fontsize=8,
    zorder=10,
)

# Save the atlas figure
if SAVE_PNG is True:
    fig.savefig(
        "results/env3-atlas-1.png",
        dpi=300,
        format="png",
    )
if SAVE_PGF is True:
    fig.savefig(
        "results/env3-atlas-1.pgf",
        format="pgf",
    )


fig, ax = plot_triangulation.plot_2d(
    triang,
    env,
    max_cols=5,
    custom_sign_order=[
        [-1, 4],
        [-3, -2],
        [-4, 1],
        [-4, -2],
        [-6, -3, -2],
    ],
    add_env_subplot=False,
    show_obstacles=False,
    start_idx_cmap=4,
    figsize=[18, 4.3],
)
ax[3].plot(
    goal[0],
    goal[1],
    color="green",
    marker="o",
    markersize=4,
    zorder=10,
)
ax[3].text(
    x_text,
    y_text,
    "$\\tilde{{{x}}}_{{{\\mathrm{{{g}}}, 1}}}$",
    fontsize=8,
    zorder=10,
)
ax[4].plot(
    goal[0],
    goal[1],
    color="green",
    marker="o",
    markersize=4,
    zorder=10,
)
ax[4].text(
    x_text,
    y_text,
    "$\\tilde{{{x}}}_{{{\\mathrm{{{g}}}, 3}}}$",
    fontsize=8,
    zorder=10,
)
ax[1].plot(
    goal[0],
    goal[1],
    color="green",
    marker="o",
    markersize=4,
    zorder=10,
)
ax[1].text(
    x_text,
    y_text,
    "$\\tilde{{{x}}}_{{{\\mathrm{{{g}}}, 4}}}$",
    fontsize=8,
    zorder=10,
)

# Save the atlas figure
if SAVE_PNG is True:
    fig.savefig(
        "results/env3-atlas-2.png",
        dpi=300,
        format="png",
    )
if SAVE_PGF is True:
    fig.savefig(
        "results/env3-atlas-2.pgf",
        format="pgf",
    )

########################################################################################
# Perform path planning
alpha_tether = [28, 16, 30, 13, 1, 18]

# Find index of triangles containing the goal and robot in the base triangulation
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
# Find list of triangles that are lifted copies of the one containing the goal and robot
triang_goal_lift = [
    idx for idx, tri in enumerate(triang.vertices_dual_lift) if tri[0] == triang_goal
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
tether_list: list[list[np.ndarray]] = []
for idx, goal_lift in enumerate(triang_goal_lift):

    # Compute geodesic path to travel
    length, path = triang.geodesic_distance(
        p1=goal,
        s1=triang.vertices_dual_lift[goal_lift][1],
        p2=robot,
        s2=[-1],
        search_algorithm="dfs",
    )
    path_length_list.append(length)
    path_list.append(path)

    # Compute representative alpha-path of the path (to compute resulting tether config)
    alpha_tether_after_motion = graph_search.bfs(
        triang.vertices_dual_lift,
        triang.edges_dual_lift,
        triang_anchor_lift,
        goal_lift,
    )
    alpha_tether_after_motion = [
        triang.vertices_dual_lift[idx][0] for idx in alpha_tether_after_motion
    ]

    # Compute geodesic tether configuration after motion
    tether_after_motion = triang.homotopic_shortest_path(
        alpha_tether_after_motion,
        p_init=env.anchor_point,
        p_end=goal,
    )

    # # Append to list
    tether_list.append(tether_after_motion)

# Initialize paths plot
fig: plt.Figure
axs: np.ndarray[plt.Axes]
fig, axs = plt.subplots(
    1,
    5,
    figsize=np.array([18, 4]) / 2.54,
    constrained_layout=True,
)

# Add plots of geodesics
for idx, path in enumerate(path_list):
    if idx >= 5:
        break  # limit to first 5 paths
    plot.plot_env(
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
        target_ax=axs[idx],
    )
    axs[idx].plot(
        path[:, 0],
        path[:, 1],
        color="blue",
        zorder=8,
        linewidth=1,
    )
    axs[idx].plot(
        tether_list[idx][:, 0],
        tether_list[idx][:, 1],
        color="black",
        zorder=7,
        linewidth=1.2,
    )
    axs[idx].plot(
        goal[0],
        goal[1],
        color="green",
        marker="o",
        markersize=4,
        zorder=10,
    )
    axs[idx].text(
        0.7,
        8.8,
        "$x_\\mathrm{{{g}}}$",
        fontsize=8,
        zorder=10,
    )
    axs[idx].text(
        1.5,
        -1,
        "$\\mathrm{len}(\\beta_{%d})=%.2f$" % (idx + 1, path_length_list[idx]),
        fontsize=8,
        zorder=10,
    )  # add text label below plot

# Add curve labels in the plot
axs[0].text(
    3.7,
    3,
    "$\\beta_{{{1}}}$",
    fontsize=8,
    zorder=10,
    color="blue",
)
axs[0].text(
    3.8,
    6,
    "$\\gamma_{{{1}}}'$",
    fontsize=8,
    zorder=10,
    color="black",
)
axs[1].text(
    0.6,
    3,
    "$\\beta_{{{2}}}$",
    fontsize=8,
    zorder=10,
    color="blue",
)
axs[1].text(
    3.2,
    3.2,
    "$\\gamma_{{{2}}}'$",
    fontsize=8,
    zorder=10,
    color="black",
)
axs[2].text(
    3.7,
    3,
    "$\\beta_{{{3}}}$",
    fontsize=8,
    zorder=10,
    color="blue",
)
axs[2].text(
    6.5,
    5.8,
    "$\\gamma_{{{3}}}'$",
    fontsize=8,
    zorder=10,
    color="black",
)
axs[3].text(
    3.7,
    3,
    "$\\beta_{{{4}}}$",
    fontsize=8,
    zorder=10,
    color="blue",
)
axs[3].text(
    5.2,
    5.8,
    "$\\gamma_{{{4}}}'$",
    fontsize=8,
    zorder=10,
    color="black",
)
axs[4].text(
    3.7,
    3,
    "$\\beta_{{{5}}}$",
    fontsize=8,
    zorder=10,
    color="blue",
)
axs[4].text(
    3.7,
    5.4,
    "$\\gamma_{{{5}}}'$",
    fontsize=8,
    zorder=10,
    color="black",
)

# Save the path planning figure
if SAVE_PNG is True:
    fig.savefig(
        "results/env3-path-planning.png",
        dpi=300,
        format="png",
    )
if SAVE_PGF is True:
    fig.savefig(
        "results/env3-path-planning.pgf",
        format="pgf",
    )

if SHOW_PLOTS is True:
    plt.show()
