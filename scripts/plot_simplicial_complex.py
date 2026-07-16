"""
Generates the following plots for env_3b (note: script is tailored to this env)
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

env_name = "env_3b"  # set to None to open file dialog and select manually
show_figures = False

# Move to script directory
abspath = os.path.abspath(__file__)
dir_name = os.path.dirname(abspath)
os.chdir(dir_name)

# Create settings and env objects
settings = Settings(create_sim_folder=False)
settings.env_name = f"{env_name}.yaml"
env = env_2d.Env2D(settings)
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
    figsize=[15, 15],
)
fig.savefig(f"results/{env_name}.png", dpi=900, format="png", bbox_inches="tight")

# Generate triangulation
triang = Triangulation(env)
triang.triangulate()
triang.set_max_dist(11.0)
triang.set_max_triangles(200)
triang.set_entanglement_definition("convex_hull")
triang.lift_triangulation(check_entanglement=True)
triang.entanglement_triangles_lift[21] = False  # HACK bugfix triangulation

# Visualization specifications
cmap = CustomColors.layers_cmap[::-1] + CustomColors.layers_cmap[1:]
signatures = [list(s) for s in set(tuple(v[1]) for v in triang.vertices_dual_lift)]
print(f"Signatures (#{len(signatures)}): {signatures}.")

# Custom signatures order for plotting simplicial complex model of env_3b
order = [
    [2, 3, -2],
    [2, 3, 3],
    [2, 3],
    [2, 2],
    [2, -3, -2],
    [2, -3],
    [2],
    [],
    [-1],
    [4],
    [4, 3],
    [-2],
]
order = order[::-1]  # flip list (cosmetic only)

# Generate plots
# Matplotlib plot -- used only to select the ordering of the signature layers
fig, ax = plot_triangulation.plot_3d(
    triang,
    env,
    connect_layers=False,
    multi_layer_triangles=True,
    custom_sign_order=order,
    layers_colormap=cmap,
    show_layer_area=False,
    show_obstacles=False,
    pov=[25, -70, 0],
    figsize=[7.5, 6.5],
)
ax.set_proj_type("ortho")
ax.set_box_aspect((1, 1, 0.7))
ax.set_xlabel("")
ax.set_ylabel("")
ax.set_zlabel("")
ax.set_xticklabels([])
ax.set_yticklabels([])
ax.set_zticklabels([])

# Pyplot plot - length-reachable simplicial complex R
fig_R = plot_triangulation.plot_3d_plotly(
    triang,
    env,
    custom_sign_order=order,
    layers_colormap=cmap,
    show_obstacles=True,
    show_layer_area=False,
    pov=[25, -85, 2],
)

# Pyplot plot - entanglement-free simplicial complex N
fig_N = plot_triangulation.plot_3d_plotly(
    triang,
    env,
    custom_sign_order=order,
    layers_colormap=cmap,
    show_obstacles=True,
    show_layer_area=False,
    plot_entanglement_free_simplices=True,
    pov=[25, -85, 2],
)

# Save plots
fig_R.write_image(
    f"results/{env_name}-sc-R.png",
    width=300,
    height=300,
    scale=10,
)
fig_N.write_image(
    f"results/{env_name}-sc-N.png",
    width=300,
    height=300,
    scale=10,
)

# Show plots
if show_figures is True:
    plt.show()
    fig_R.show()
    fig_N.show()
