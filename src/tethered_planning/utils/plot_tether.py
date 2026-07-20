import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from tethered_planning.env import env_2d
from tethered_planning.utils import plot

# Matplotlib settings
labels_font_size = 8
tick_labels_font_size = 8
mpl.rcParams.update(
    {
        "pgf.texsystem": "xelatex",  # or any other engine you want to use
        "text.usetex": True,  # use TeX for all texts
        "font.family": "serif",
        "font.size": labels_font_size,
        "axes.labelsize": labels_font_size,
        "legend.fontsize": labels_font_size,
        "xtick.labelsize": tick_labels_font_size,
        "ytick.labelsize": tick_labels_font_size,
        "pgf.rcfonts": False,
        "pgf.preamble": "\\usepackage[T1]{fontenc}",  # extra preamble for LaTeX
    }
)


def plot_tether(
    env: env_2d.Env2D,
    tether_init: np.ndarray,
    tether_final: np.ndarray | None = None,
    tether_snapshots: list[np.ndarray] = [],
    cmap: list[str] | None = None,
    show_plot: bool = False,
) -> tuple[plt.Figure, plt.Axes]:

    # Select colormap
    if cmap is None or len(cmap < len(tether_snapshots)):
        cmap_cont = mpl.colormaps["viridis"]
        cmap = cmap_cont(np.linspace(0, 1, len(tether_snapshots)))

    fig, ax = plot.plot_env(
        env,
        show_anchor=False,
        show_robot=False,
        show_tether=False,
        show_goal=False,
        show_legend=False,
        show_generators=False,
        show_generators_labels=False,
        show_robot_anchor_labels=False,
        show_axes_labels=False,
    )

    # TODO: add tethers to plot

    # Save and show plot
    if show_plot is True:
        plt.show()
