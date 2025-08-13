import os

from tethered_planning.env import env_2d
from tethered_planning.utils import io, plot
from tethered_planning.utils.settings import Settings

# Move to script directory
abspath = os.path.abspath(__file__)
dir_name = os.path.dirname(abspath)
os.chdir(dir_name)

# Select environment to use
env_name = input("Enter the name of the environment: ")
env_path = f"data/{env_name}.yaml"

# Load settings and env
settings = Settings("settings_env_plot")  # "plot_and_save_env.yaml"
settings.env_name = env_name
env = env_2d.Env2D(settings)

# Plot the environment and save the figure
fig, _ = plot.plot_env(env, settings)
fig_name = env_name.replace("_", "-")
io.save_figure(fig, settings, fig_name, "png")
