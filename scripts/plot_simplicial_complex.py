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
from tethered_planning.utils import plot, plot_triangulation
from tethered_planning.utils.colors import CustomColors
from tethered_planning.utils.settings import Settings

env_name = "env_3"  # set to None to open file dialog and select manually

# Move to script directory
abspath = os.path.abspath(__file__)
dir_name = os.path.dirname(abspath)
os.chdir(dir_name)

# Create settings and env objects
settings = Settings(create_sim_folder=False)
settings.env_name = f"{env_name}.yaml"
env = env_2d.Env2D(settings)
# fig, _ = plot.plot_env(
#     env,
#     show_tether=False,
#     show_robot=False,
#     show_anchor=True,
#     show_goal=False,
#     show_legend=False,
#     show_generators=True,
#     show_generators_labels=True,
#     show_obstacles_labels=True,
#     figsize=[15, 15],
# )
# fig.savefig(f"results/{env_name}.png", dpi=900, format="png", bbox_inches="tight")

# Generate triangulation
triang = Triangulation(env)
triang.triangulate()
triang.set_max_dist(10.0)
triang.set_max_triangles(200)
triang.lift_triangulation()
print(
    f"Triangulation completed with {len(triang.entanglement_triangles_lift)} triangles."
)

# Visualization specifications
signatures = [list(s) for s in set(tuple(v[1]) for v in triang.vertices_dual_lift)]
print(f"Signatures (#{len(signatures)}): {signatures}.")
order = [
    [-6, -4],
    [-6],
    [-4, 1, -4],
    [-4, 1],
    [-4, -2],
    [-4],
    [-4, 2],
    # [-4, 2, 4],
    # [-5],
    # [1],
    [],
    [5],
    [6],
    [-3, -2],
    [-3],
    [-1],
    [-1, 4],
    # [-1, 4, -1],
    [-1, 2],
    [-1, -2],
]
cmap = CustomColors.layers_cmap[::-1] + CustomColors.layers_cmap

# Generate plot
fig, ax = plot_triangulation.plot_3d(
    triang,
    env,
    connect_layers=False,
    multi_layer_triangles=True,
    custom_sign_order=order,
    layers_colormap=cmap,
    # show_layer_area=False,
    show_obstacles=True,
    pov=[25, -70, 0],
    figsize=[7.5, 6.5],
)
ax.set_proj_type("ortho")
ax.set_box_aspect((1, 1, 0.7))
# ax.set_xlabel("")
# ax.set_ylabel("")
# ax.set_zlabel("")
# ax.set_xticklabels([])
# ax.set_yticklabels([])
# ax.set_zticklabels([])

# Save + show plot
fig.savefig("results/env2-simplicial-complex.png", dpi=900, format="png")
fig.savefig("results/env2-simplicial-complex.svg")
plt.show()
