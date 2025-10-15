"""
Generates the following plots
- plot of the 2D environment including the anchor point but not the robot;
- the plot of the environment triangulation, including both primal and dual graphs;
- the plot of the homotopy-augmented graph.
"""

import os
import tkinter as tk
from tkinter import filedialog

import matplotlib as mpl
from packaging import version

from tethered_planning.env import env_2d
from tethered_planning.env.triangulation import Triangulation
from tethered_planning.utils import plot, plot_triangulation
from tethered_planning.utils.colors import CmdColors, CustomColors
from tethered_planning.utils.settings import Settings

# Script settings
SAVE_PNG = True
SAVE_PGF = True
# filename = None
filename = "env_1.yaml"  # set to None to open file dialog

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

# Select environment to use
if "filename" not in globals() or filename is None:
    root = tk.Tk()
    root.withdraw()
    try:
        filename = filedialog.askopenfilename(initialdir=dir_name).split("/")[-1]
    except FileNotFoundError:
        print("File was not found, using default environment.")
        filename = "env_1.yaml"
env_name = filename.replace("_", "-")
env_name = env_name.replace(".yaml", "")
base_name = "results/" + env_name

# Create settings and env objects
settings = Settings(create_sim_folder=False)
settings.env_name = filename
env = env_2d.Env2D(settings)

# Plot and save the environment
fig, _ = plot.plot_env(
    env,
    show_tether=False,
    show_robot=False,
    show_anchor=True,
    show_goal=False,
    show_legend=False,
    show_generators=True,
    show_generators_labels=True,
    show_obstacles_labels=True,
    figsize=[4, 4],
)
if SAVE_PNG is True:
    fig.savefig(
        base_name + ".png",
        dpi=300,
        format="png",
        bbox_inches="tight",
        pad_inches=0.01,
    )
if SAVE_PGF is True:
    fig.savefig(
        base_name + ".pgf",
        dpi=300,
        format="pgf",
        bbox_inches="tight",
        pad_inches=0.01,
    )

# Generate triangulation
triang = Triangulation(env)
triang.set_max_dist(1000.0)
triang.set_max_triangles(40)
triang.triangulate()
triang.lift_triangulation()

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
    figsize=[4, 4],
)
if SAVE_PNG is True:
    fig.savefig(
        base_name + "-triang.png",
        dpi=300,
        format="png",
        bbox_inches="tight",
        pad_inches=0.01,
    )
if SAVE_PGF is True:
    fig.savefig(
        base_name + "-triang.pgf",
        dpi=300,
        format="pgf",
        bbox_inches="tight",
        pad_inches=0.01,
    )

# Plot and save the lifted triangulation
pov: list[float]  # point of view for 3D plot [elevation, azimuth, roll]
order: list[list[int]]  # custom order for layers plotting
cmap: list[str]  # custom colormap for layers plotting
match settings.env_name:
    case "env_1.yaml":
        pov = [15, 35, 2]
        order = [[1, 1, 1], [1, 1], [1], [], [-1], [-1, -1]]
        cmap = CustomColors.layers_cmap[0:6]
    case "env_2.yaml":
        pov = [15, 35, 2]
        order = [[1, 2], [2], [1], [], [-1], [-2]]
        cmap = CustomColors.layers_cmap[0:6]
    case "env_3.yaml":
        pov = [15, 35, 2]
        order = [[1, 2], [2], [1], [], [-1], [-2], [-2, -1]]
        cmap = CustomColors.layers_cmap[0:7]
fig, _ = plot_triangulation.plot_3d(
    triang,
    env,
    connect_layers=False,
    multi_layer_triangles=True,
    custom_sign_order=order,
    layers_colormap=cmap,
    pov=pov,
    figsize=[6, 6],
)
if SAVE_PNG is True:
    fig.savefig(
        base_name + "-lift.png",
        dpi=300,
        format="png",
    )
if SAVE_PGF is True:
    fig.savefig(
        base_name + "-lift.pgf",
        dpi=300,
        format="pgf",
    )
