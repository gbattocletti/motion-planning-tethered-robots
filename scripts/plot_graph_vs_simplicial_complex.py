"""
Generates a plot showing the h-augmented graph superimposed to the simplicial complex
model. The plot is organized as a grid of 2D plots, each representing one layer of the
simplicial complex model.
"""

import os
import pickle

import matplotlib.pyplot as plt
import numpy as np

from tethered_planning.env import env_2d
from tethered_planning.env.grid_graph import GridGraph
from tethered_planning.env.triangulation import Triangulation
from tethered_planning.utils import plot_triangulation
from tethered_planning.utils.settings import Settings

# Script settings
filename: str = "results/entanglement_free_model/comparison-31.pkl"

# Move to script directory
abspath = os.path.abspath(__file__)
dir_name = os.path.dirname(abspath)
os.chdir(dir_name)

# Load data
objects: list = []
with open(filename, "rb") as openfile:
    while True:
        try:
            objects.append(pickle.load(openfile))
        except EOFError:
            break
data: dict = objects[0]
settings: Settings = data["settings"]
env: env_2d.Env2D = data["env"]
length: float = env.tether_length
anchor: np.ndarray = env.anchor_point
triang_R: Triangulation = data["triangulation"]  # length reachable model
graph_R: GridGraph = data["graph_2"]

# Define list of layers to visualize
sign_order = [
    [],  # TODO: get list of signatures from triang_R and fill this list
]

# Plot simplicial complex
fig: plt.Figure
ax: plt.Axes
fig, ax = plot_triangulation.plot_2d(
    triang_R,
    env,
    custom_sign_order=sign_order,
    max_cols=5,
    add_env_subplot=False,
    show_obstacles=True,
    figsize=[18, 18],
)

# Plot grid graph
# TODO

plt.show()
