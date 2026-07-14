"""
Generates a plot showing a disentanglement path computed from an entangled tether
configuration to a non entangled one. The script performs the path planning operation
for the three different entanglement definitions, and superimposes the resulting paths
on the same plot.
"""

import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
from shapely import Point

from tethered_planning.env import env_2d
from tethered_planning.env.triangulation import Triangulation
from tethered_planning.utils import curves, plot
from tethered_planning.utils.settings import Settings

# Script settings
env_filename = "env_4.yaml"
experiments = [
    ["results/entanglement_free_model/comparison-31.pkl", "convex_hull"],
    ["results/entanglement_free_model/comparison-32.pkl", "linear_homotopy"],
    ["results/entanglement_free_model/comparison-33.pkl", "local_visibility"],
]
SELECT_TETHER_MANUALLY: bool = True

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
fig: plt.Figure
ax: plt.Axes
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
            print(tether)  # TEMP (to remove)
        else:
            pass  # TODO: hardcode the tether config
        signature = curves.compute_signature(tether, env, simplify=True)
        robot = tether[-1]
        # NOTE: no goal is selected as objective is to reach N_bar from R_bar

        # Initialize plot
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
            figsize=[8, 8],
        )

    # Definition-specific data
    triang_R: Triangulation = data["triangulation"]  # length reachable model
    triang_N: Triangulation = data["triangulation_entanglement"]  # entang free model

    # Sanity check that initial config is entangled
    triang_robot = int(
        triang_R.triang_tree.query(
            Point(robot),
            predicate="intersects",
        )[0]
    )
    triang_robot_lift = [
        idx
        for idx, tri in enumerate(triang_R.vertices_dual_lift)
        if ((tri[0] == triang_robot) and (tri[1] == signature))
    ]
    if not triang_robot_lift == []:
        raise ValueError(
            "The initial tether configuration is not entangled w.r.t. the entanglement "
            f"definition {definition}."
        )

    # Perform path planning on triangulations to go from R to N
    # Find triangles with goal, robot, and anchor
    triang_robot = int(
        triang_R.triang_tree.query(
            Point(robot),
            predicate="intersects",
        )[0]
    )
    triang_anchor = int(
        triang_R.triang_tree.query(
            Point(env.anchor_point),
            predicate="intersects",
        )[0]
    )

    # Find lifted copies of goal and robot
    triang_robot_lift = [
        idx
        for idx, tri in enumerate(triang_R.vertices_dual_lift)
        if ((tri[0] == triang_robot) and (tri[1] == signature))
    ][0]
    triang_anchor_lift = [
        idx
        for idx, tri in enumerate(triang_R.vertices_dual_lift)
        if ((tri[0] == triang_anchor) and (tri[1] == []))
    ][0]

    # Find geodesics in different homotopy classes
    # TODO: implement graph search with stopping when a node in N is reached
    path = np.zeros([10, 2])  # TEMP
    ax.plot(path[:, 0], path[:, 1], color=colors[idx], zorder=8, linewidth=1)

# Save figures
fig.savefig("results/disentanglement.png", dpi=900, format="png")
fig.savefig("results/disentanglement.svg")
fig.savefig("results/disentanglement.pdf")
