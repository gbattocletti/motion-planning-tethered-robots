"""
Generates a plot comparing the area coverage of the the h-augmented graph vs the
simplicial complex model for different numbers of obstacles and different values of the
tether length.
"""

import os

import matplotlib.pyplot as plt
import numpy as np

# Move to script directory
abspath = os.path.abspath(__file__)
dir_name = os.path.dirname(abspath)
os.chdir(dir_name)

# Load data
data = np.loadtxt(
    "results/conservativeness_reduction/comparison_results.csv",
    delimiter=",",
    skiprows=1,
)
n_rows, n_cols = data.shape

# Define colormap
cmap = {
    "10.0": [
        "#FF4747",
        "#E70000",
        "#8F0000",
        "#4B0000",
    ],
    "12.5": [
        "#FFC340",
        "#FFAE00",
        "#B97F00",
        "#6B4900",
    ],
    "15.0": [
        "#3C7AFF",
        "#0051FF",
        "#003097",
        "#001646",
    ],
}

# Define plot groups
n = [1, 2, 5, 7, 12, 15]
m = [1, 2, 4, 6, 8, 10]
lengths = [10.0, 12.5, 15.0]
methods_indexes = [7, 9, 3, 5]  # columns with area data

# Define styles
styles = [
    "--o",  # H 0.5
    "--s",  # H 0.25
    "-^",  # R
    "-D",  # R'
]


# Generate figure
fig: plt.Figure
ax: plt.Axes
fig, ax = plt.subplots(
    1,
    1,
    figsize=np.array([5, 3]) / 2.54,
    sharex=True,
)

for length_idx, length in enumerate(lengths):
    # ax = axes[length_idx]  % for 3-rows plot
    for idx, col in enumerate(methods_indexes):
        if col == 5:
            area = [
                data[i][col] + data[i][3] for i in range(n_rows) if data[i][1] == length
            ]
        else:
            area = [data[i][col] for i in range(n_rows) if data[i][1] == length]
        ax.plot(
            m,
            area,
            styles[idx],
            linewidth=0.7,
            markersize=2,
            markeredgewidth=0.7,
            markerfacecolor="none" if col in [7, 9] else cmap[str(length)][idx],
            color=cmap[str(length)][idx],
        )
    ax.set_yscale("log")
    ax.set_xlim([0.7, 10.3])
    ax.grid(True, which="major", linestyle=":", color="gray", linewidth=0.5, zorder=1)
    ax.grid(True, which="minor", linestyle=":", color="gray", linewidth=0.3, zorder=1)
    ax.minorticks_on()
    ax.set_xticklabels([])
    ax.set_yticklabels([])


fig.savefig("results/conservativeness_reduction.png", dpi=1200, format="png")
# plt.show()
