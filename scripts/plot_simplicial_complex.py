"""
Generates the following plots
- plot of the 2D environment including the anchor point but not the robot;
- the plot of the environment triangulation, including both primal and dual graphs;
- the plot of the homotopy-augmented graph.
- a plot showing a path planning scenario in which all the homotopy classes to a given
  point are found and visualized.
"""

import os

import matplotlib.pyplot as plt

from tethered_planning.env import env_2d
from tethered_planning.env.triangulation import Triangulation
from tethered_planning.utils import plot_triangulation
from tethered_planning.utils.colors import CustomColors
from tethered_planning.utils.settings import Settings

filename = "env_3.yaml"  # set to None to open file dialog and select manually

# Move to script directory
abspath = os.path.abspath(__file__)
dir_name = os.path.dirname(abspath)
os.chdir(dir_name)

# Create settings and env objects
settings = Settings(create_sim_folder=False)
settings.env_name = filename
env = env_2d.Env2D(settings)

# Generate triangulation
triang = Triangulation(env)
triang.triangulate()
triang.set_max_dist(8.0)
triang.set_max_triangles(200)
pov = [15, 35, 2]
order = [[2], [1], [], [-1], [-2], [-2, -1]]  # TODO: update depending on triangulation
if len(order) > len(CustomColors.layers_cmap):
    cmap = []
    for i in range(len(order) / len(CustomColors.layers_cmap)):
        cmap.append(CustomColors.layers_cmap)
    cmap.append(
        CustomColors.layers_cmap[0 : len(order) % len(CustomColors.layers_cmap)]
    )
cmap = CustomColors.layers_cmap[0 : len(order)]
triang.lift_triangulation()

# Generate plot
fig, ax = plot_triangulation.plot_3d(
    triang,
    env,
    connect_layers=False,
    multi_layer_triangles=True,
    # custom_sign_order=order,
    # layers_colormap=cmap,
    pov=pov,
    figsize=[6, 6],
)
ax.set_proj_type("ortho")
ax.set_box_aspect((1, 1, 1))
ax.plot(
    env.anchor_point[0],
    env.anchor_point[1],
    2,
    marker=".",
    markersize=6,
    color="red",
    zorder=15,
)

# Save + show plot
fig.savefig("results/env2-simplicial-complex.png", dpi=900, format="png")
fig.savefig("results/env2-simplicial-complex.svg")
plt.show()
