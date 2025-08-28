"""
Triangulate the environment and lift it to obtain a simplicial complex representing the
workspace of a tethered robot.
"""

import os

import matplotlib.pyplot as plt
import shapely
from shapely import LineString, Point, Polygon, STRtree
from shapely.plotting import plot_points, plot_polygon

from tethered_planning.env import env_2d
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
triang = shapely.constrained_delaunay_triangles(env.free_workspace)
triang_tree = STRtree(triang.geoms)  # for geometry lookup
root_idx = triang_tree.query(Point(env.anchor_point), predicate="intersects")


def edges(triangle: Polygon):
    return list(
        map(
            LineString, zip(triangle.exterior.coords[:-1], triangle.exterior.coords[1:])
        )
    )


# Build simplcial complex (overlapped manifold)
queue: list[int] = [root_idx]
n_max: int = 100  # max number of triangles
n: int = 0
while queue:
    idx: int = queue.pop(0)[0]  # pop 1st element from queue
    n += 1  # increase triangles counter

    # iterate over edges to expand toward adjacent triangles
    for edge in edges(triang_tree.geometries[idx]):
        idx_list: list[int] = triang_tree.query(edge, predicate="covered_by")
        idx_list.remove(idx)  # remove current triangle
        if idx_list:
            queue.append(idx_list[0])  # append new triangle to list

        # TODO: do something to avoid previous triangle to be added again at the next
        #       time step!

    # Termination condition TODO: swap with
    if n >= n_max:
        break


# Generate figure
fig, ax = plot.plot_env(env, settings)
plot_polygon(
    shapely.MultiPolygon(triang),
    ax=ax,
    facecolor=(0.0, 0.0, 0.0, 0.0),
    edgecolor=(1.0, 0, 0.01, 0.8),
    linewidth=1,
    add_points=False,
    zorder=4,
)  # plot triangles
for t in triang_tree.geometries:
    plot_points(
        t.centroid,
        color="coral",
    )  # plot centroids (for dual graph)
plt.show()

# Save figure
fig_env_name = "env"
fig_triang_name = "triang"
io.save_figure(fig, settings, fig_env_name, "png")
