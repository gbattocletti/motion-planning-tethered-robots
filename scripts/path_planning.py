"""
Generates a plot showing a path planning scenario in which all the homotopy classes
between the robot and the goal are evaluated in parallel.
"""

import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from packaging import version
from shapely import Point

from tethered_planning.env import env_2d
from tethered_planning.env.triangulation import Triangulation
from tethered_planning.utils import plot
from tethered_planning.utils.colors import CmdColors
from tethered_planning.utils.settings import Settings

# Script settings
SAVE_PNG = True
SAVE_PGF = True
SHOW_PLOTS = True
filename = "env_2.yaml"  # set to None to open file dialog and select manually

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

# Find env name and base name for saving
env_name = filename.replace("_", "-")
env_name = env_name.replace(".yaml", "")
base_name = "results/" + env_name

# Create settings and env objects
settings = Settings(create_sim_folder=False)
settings.env_name = filename
env = env_2d.Env2D(settings)

# Generate triangulation
triang = Triangulation(env)
triang.triangulate()

# Perform path planning
robot = []  # TODO
tether = []  # TODO -- only for visualization and initial condition (lift)
goal = np.array([1.0, 3.0])
triang_goal = int(
    triang.triang_tree.query(
        Point(goal),
        predicate="intersects",
    )[0]
)  # index of triangle containing the goal in the base triangulation
triang_goal_lift = [
    idx for idx, tri in enumerate(triang.vertices_dual_lift) if tri[0] == triang_goal
]  # list of lifted triangles containing the goal

length_list: list[float] = []
geodesic_list: list[list[np.ndarray]] = []
for idx, goal_lift in enumerate(triang_goal_lift):

    # Find geodesics in different homotopy classes
    length, geodesic = triang.geodesic_distance(
        p1=goal,
        s1=triang.vertices_dual_lift[goal_lift][1],
        p2=env.anchor_point,
        s2=[],
        search_algorithm="dfs",
    )
    length_list.append(length)
    geodesic_list.append(geodesic)

# Plot and save the geodesic
fig, _ = plot.plot_env(
    env,
    show_tether=False,
    show_robot=False,
    show_anchor=True,
    show_goal=False,
    show_legend=False,
    show_generators=True,
    points=[goal],  # goal point
    curves=geodesic_list,  # geodesics
    show_curves_labels=True,
    show_generators_labels=True,
    show_obstacles_labels=True,
    figsize=[4.5, 4.5],
)

if SAVE_PNG is True:
    fig.savefig(
        base_name + "-paths.png",
        dpi=300,
        format="png",
    )
if SAVE_PGF is True:
    fig.savefig(
        base_name + "-paths.pgf",
        format="pgf",
    )

if SHOW_PLOTS is True:
    plt.show()
