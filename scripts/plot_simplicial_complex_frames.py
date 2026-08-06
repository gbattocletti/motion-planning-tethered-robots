"""
Generate a sequence of frames increasingly showing the simplicial complex model of a
2D environment. Each frame shows the simplicial complex with one more triangle added.
"""

import os

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
    figsize=[6, 6],
)
fig.savefig(f"results/{env_name}.png", dpi=900, format="png", bbox_inches="tight")

# Generate triangulation
triang = Triangulation(env)
triang.triangulate()
triang.set_max_dist(11.0)
triang.set_max_triangles(1000)
triang.lift_triangulation()
n_triangs = len(triang.triangles_lift)
print(f"Triangulation completed with {n_triangs} triangles.")

# Visualization specifications
cmap = CustomColors.layers_cmap[::-1] + CustomColors.layers_cmap[1:]
signatures = [list(s) for s in set(tuple(v[1]) for v in triang.vertices_dual_lift)]
print(f"Signatures (#{len(signatures)}): {signatures}.")

# Custom signatures order for plotting simplicial complex model of env_3b
order = [
    [4],
    [4, 3],
    [-1],
    [],
    [2],
    [2, -3],
    [2, -3, -2],
    [2, 3],
    [2, 3, -2],
]
order = order[::-1]  # flip list (cosmetic only)

# Generate frames
vertices_dual_all = triang.vertices_dual_lift.copy()
for idx in range(n_triangs):

    triang_sub = triang
    triang_sub.vertices_dual_lift = vertices_dual_all[: idx + 1]

    # Generate plot
    fig_R = plot_triangulation.plot_3d_plotly(
        triang_sub,
        env,
        custom_sign_order=order,
        layers_colormap=cmap,
        show_obstacles=True,
        show_layer_area=False,
        pov=[25, -85, 2],
    )

    # Save plots
    fig_R.write_image(
        f"results/simplicial_complex_frames/{env_name}-{idx}.png",
        width=300,
        height=300,
        scale=10,
    )
