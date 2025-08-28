"""
Plots a 2D environment and (optionally) the initial conditions of the tethered robot.
"""

import os
import tkinter as tk
from tkinter import filedialog

from tethered_planning.env import env_2d
from tethered_planning.utils import io, plot
from tethered_planning.utils.settings import Settings

# Move to script directory
abspath = os.path.abspath(__file__)
dir_name = os.path.dirname(abspath)
os.chdir(dir_name)

# Select environment to use
root = tk.Tk()
root.withdraw()
try:
    filename = filedialog.askopenfilename(initialdir=dir_name).split("/")[-1]
except FileNotFoundError:
    print("File was not found, using default environment.")
    filename = "env_1.yaml"

# Load settings and env
settings = Settings()
settings.env_name = filename
env = env_2d.Env2D(settings)

# Plot the environment
fig, _ = plot.plot_env(env, settings)
# TODO: add option to plot tether and robot

# Save the figure
fig_name = filename.replace("_", "-")
io.save_figure(fig, settings, fig_name, "png")
