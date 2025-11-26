"""
Generates the following plots
- plot of the 2D environment including the anchor point but not the robot;
- the plot of the environment triangulation, including both primal and dual graphs;
- the plot of the homotopy-augmented graph.
- a plot showing a path planning scenario in which all the homotopy classes to a given
  point are found and visualized.
"""

import os

import matplotlib as mpl
import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import numpy as np
from packaging import version

from tethered_planning.env import env_2d
from tethered_planning.env.triangulation import Triangulation
from tethered_planning.utils import plot, plot_triangulation
from tethered_planning.utils.colors import CmdColors, CustomColors
from tethered_planning.utils.settings import Settings

labels_font_size = 8
tick_labels_font_size = 8
mpl.rcParams.update(
    {
        "pgf.texsystem": "xelatex",  # or any other engine you want to use
        "text.usetex": True,  # use TeX for all texts
        "font.family": "serif",
        "font.size": labels_font_size,
        "axes.labelsize": labels_font_size,
        "legend.fontsize": labels_font_size,
        "xtick.labelsize": tick_labels_font_size,
        "ytick.labelsize": tick_labels_font_size,
        "pgf.rcfonts": False,
        "pgf.preamble": "\\usepackage[T1]{fontenc}",  # extra preamble for LaTeX
    }
)

# Script settings
SAVE_PNG = True
SAVE_PGF = True
SHOW_PLOTS = False
filename = "env_2b.yaml"  # set to None to open file dialog and select manually

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

# Manually define tether for visualization
robot = [7, 1]
env.anchor_point = np.array([4.0, 5.2])
tether = np.array(
    [
        [4.0, 5.2],
        [4.2, 4.5],
        [4.3, 4.0],
        [4.4, 3.5],
        [4.5, 3.0],
        [4.7, 2.5],
        [5.0, 2.0],
        [5.5, 1.5],
        [6.0, 1.2],
        [6.5, 1.1],
        [7.0, 1.0],
    ]
)
env.robot_initial_pos = robot
env.tether_state = tether

# Plot and save the environment
fig, _ = plot.plot_env(
    env,
    show_tether=True,
    show_robot=True,
    show_anchor=True,
    tether=tether,
    show_goal=False,
    show_legend=False,
    show_generators=True,
    show_generators_labels=True,
    show_obstacles_labels=True,
    figsize=[4.5, 4.5],
)
if SAVE_PNG is True:
    fig.savefig(
        "results/env2.png",
        dpi=300,
        format="png",
        bbox_inches="tight",
        pad_inches=0.01,
    )
if SAVE_PGF is True:
    fig.savefig(
        "results/env2.pgf",
        format="pgf",
        bbox_inches="tight",
        pad_inches=0.01,
    )

# Generate triangulation
triang = Triangulation(env)
triang.triangulate()

# Plot and save the triangulation
fig, _ = plot.plot_graph(
    triang.vertices,
    triang.edges,
    env,
    nodes_dual=triang.vertices_dual,
    edges_dual=triang.edges_dual,
    show_dual_graph=True,
    label_nodes=False,
    label_triangles=False,
    show_generators_labels=True,
    show_legend=False,
    show_obstacles_labels=True,
    show_anchor=True,
    show_robot=False,
    show_tether=False,
    figsize=[4.5, 4.5],
)
if SAVE_PNG is True:
    fig.savefig(
        "results/env2-triang.png",
        dpi=300,
        format="png",
        bbox_inches="tight",
        pad_inches=0.01,
    )
if SAVE_PGF is True:
    fig.savefig(
        "results/env2-triang.pgf",
        format="pgf",
        bbox_inches="tight",
        pad_inches=0.01,
    )

# Lift the triangulation
settings.env_name = "env_2.yaml"  # symmetric env to correct 3D perspective
env = env_2d.Env2D(settings)
triang = Triangulation(env)
triang.triangulate()
triang.set_max_dist(15.0)
triang.set_max_triangles(100)
pov = [15, 35, 2]
order = [[2], [1], [], [-1], [-2], [-2, -1]]
cmap = CustomColors.layers_cmap[0:7]
triang.lift_triangulation()

# Plot and save the lifted triangulation
fig, _ = plot_triangulation.plot_3d(
    triang,
    env,
    connect_layers=False,
    multi_layer_triangles=True,
    custom_sign_order=order,
    layers_colormap=cmap,
    pov=pov,
    figsize=[4.5, 4.5],
)
if SAVE_PNG is True:
    fig.savefig(
        "results/env2-lift.png",
        dpi=300,
        format="png",
    )
if SAVE_PGF is True:
    fig.savefig(
        "results/env2-lift.pgf",
        format="pgf",
    )

# Plot and save the geodesic
fig, ax = plot_triangulation.plot_3d(
    triang,
    env,
    connect_layers=False,
    multi_layer_triangles=True,
    custom_sign_order=order,
    layers_colormap=cmap,
    pov=pov,
    figsize=[4.5, 4.5],
)

# Add points, curves, and labels
ax.plot(
    env.anchor_point[0],
    env.anchor_point[1],
    2,
    marker=".",
    markersize=6,
    color="red",
    zorder=15,
)
txt = ax.text(
    env.anchor_point[0] - 0.5,
    env.anchor_point[1] - 2.0,
    2,
    "$\\tilde{x}_\\mathrm{a}$",
    fontsize=10,
    ha="center",
    va="center",
    color="black",
    zorder=15,
)
txt.set_path_effects(
    [
        path_effects.Stroke(linewidth=1.5, foreground="white"),
        path_effects.Normal(),
    ]
)
ax.plot(
    5,
    7,
    4.25,
    marker=".",
    markersize=6,
    color="blue",
    zorder=15,
)
txt = ax.text(
    5,
    8.5,
    4.5,
    "$\\tilde{x}_\\mathrm{g}$",
    fontsize=10,
    ha="center",
    va="center",
    color="black",
    zorder=15,
)
txt.set_path_effects(
    [
        path_effects.Stroke(linewidth=1.5, foreground="white"),
        path_effects.Normal(),
    ]
)
ax.plot(
    [5, 8, 8, 6, 5],
    [5, 6, 8, 8, 7],
    [2, 2, 2, 4, 4.25],
    color="blue",
    linewidth=1.5,
    zorder=12,
)
txt = ax.text(
    5.5,
    8,
    3,
    "$\\tilde{{\\beta}}$",
    fontsize=10,
    ha="center",
    va="center",
    color="black",
    zorder=12,
)
txt.set_path_effects(
    [
        path_effects.Stroke(linewidth=1.5, foreground="white"),
        path_effects.Normal(),
    ]
)

if SAVE_PNG is True:
    fig.savefig(
        "results/env2-paths.png",
        dpi=300,
        format="png",
    )
if SAVE_PGF is True:
    fig.savefig(
        "results/env2-paths.pgf",
        format="pgf",
    )

if SHOW_PLOTS is True:
    plt.show()
