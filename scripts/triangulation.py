"""
Triangulate the environment and lift it to obtain a simplicial complex representing the
workspace of a tethered robot.
"""

import os

import matplotlib.pyplot as plt
import shapely
from shapely.plotting import plot_line, plot_points, plot_polygon

from tethered_planning.env import env_2d
from tethered_planning.env.triangulation import Triangulation
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

# Generate figure from shapely objects
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
for p in triang.triang.geoms:
    plot_points(
        shapely.Point(p.centroid),
        color="coral",
    )  # plot centroids (for dual graph)

# Generate figure from extracted data
fig, ax = plot.plot_env(env, settings)
for p in triang.vertices:
    plot_points(
        shapely.Point(p),
        color="blue",
    )  # plot primary vertices
for p in triang.vertices_dual:
    plot_points(
        shapely.Point(p),
        color="coral",
    )  # plot centroids (for dual graph)
for e in triang.edges:
    plot_line(
        shapely.LineString([triang.vertices[int(e[0])], triang.vertices[int(e[1])]]),
        color="blue",
    )  # plot edges
for e in triang.edges_dual:
    plot_line(
        shapely.LineString(
            [triang.vertices_dual[int(e[0])], triang.vertices_dual[int(e[1])]]
        ),
        color="orange",
    )  # plot edges dual
plt.show()

# Save figure
fig_env_name = "env"
fig_triang_name = "triang"
io.save_figure(fig, settings, fig_env_name, "png")
