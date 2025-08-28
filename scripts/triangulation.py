"""
Triangulate the environment and lift it to obtain a simplicial complex representing the
workspace of a tethered robot.
"""

import os

import matplotlib.pyplot as plt
import shapely
from shapely.plotting import plot_points, plot_polygon

from tethered_planning.env import env_2d
from tethered_planning.plan.triangulation import Triangulation
from tethered_planning.utils import io, plot
from tethered_planning.utils.settings import Settings

# Simulation parameters
env_name = "env_1.yaml"


########################################################################################

# Move to script directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Load settings and env
settings = Settings()
settings.env_name = env_name
env = env_2d.Env2D(settings)

# Generate triangulation
triang = Triangulation(env)
triang.triangulate()

# Generate figure
# TODO: add second plot and plot the data from the numpy data structures for validation
fig, ax = plot.plot_env(env, settings)
plot_polygon(
    triang.triang,
    ax=ax,
    facecolor=(0.0, 0.0, 0.0, 0.0),
    edgecolor=(1.0, 0, 0.01, 0.8),
    linewidth=1,
    add_points=False,
    zorder=4,
)  # plot triangles
for p in triang.vertices_dual:
    plot_points(
        shapely.Point(p),
        color="coral",
    )  # plot centroids (for dual graph)
plt.show()

# Save figure
fig_env_name = "env"
fig_triang_name = "triang"
io.save_figure(fig, settings, fig_env_name, "png")
