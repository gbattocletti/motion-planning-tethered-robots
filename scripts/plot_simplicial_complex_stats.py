"""
Generates a plot comparing the computation time and memory usage of the the h-augmented
graph vs the simplicial complex model for different numbers of obstacles and different
values of the tether length.
"""

import os

import matplotlib.pyplot as plt
import numpy as np

# Move to script directory
abspath = os.path.abspath(__file__)
dir_name = os.path.dirname(abspath)
os.chdir(dir_name)

# Define colormap
cmap_1 = [
    "#E70000",
    "#FFAE00",
    "#0051FF",
    "#008F5A",
]

cmap_2 = [
    "#AA0000",
    "#CA8A00",
    "#0039B4",
    "#007244",
]
color_hag = cmap_2
color_scm = cmap_1

# Copy data from tables
# Hidden: data with also m=10
# lengths = [10.0, 12.5, 15.0, 20.0]
# n = [1, 2, 5, 7, 12, 15]
# n_20 = [1, 2, 7, 12]
# m = [1, 2, 4, 6, 8, 10]
# m_20 = [1, 2, 6, 8]
# time_scm_10 = [0.01, 0.01, 0.08, 0.24, 0.81, 1.30]
# time_h50_10 = [0.35, 0.57, 1.28, 2.97, 3.38, 6.26]
# time_h25_10 = [2.10, 3.41, 7.36, 20.02, 20.34, 38.04]
# time_scm_12 = [0.01, 0.01, 0.19, 0.70, 2.84, 6.10]
# time_h50_12 = [0.52, 1.02, 2.80, 9.14, 12.48, 29.30]
# time_h25_12 = [3.24, 6.56, 18.31, 78.87, 112.49, 390.95]
# time_scm_15 = [0.01, 0.04, 0.40, 2.05, 10.80, 29.92]
# time_h50_15 = [1.25, 2.70, 9.09, 43.86, 75.67, 270.27]
# time_h25_15 = [4.82, 11.75, 49.33, 747.40, 1666.67, 7660.32]
# time_scm_20 = [0.02, 0.11, 26.31, 555.86]
# time_h50_20 = [1.48, 4.15, 2414.15, 9704.11]
# time_h25_20 = [10.26, 32.73, 46258.24, 171031.74]
# size_scm_10 = [4, 18, 69, 157, 449, 637]
# size_h50_10 = [493, 679, 1211, 2240, 2288, 3471]
# size_h25_10 = [1942, 2655, 4660, 8935, 8881, 13135]
# size_scm_12 = [10, 18, 143, 398, 1314, 2449]
# size_h50_12 = [722, 1080, 2340, 5443, 6548, 11216]
# size_h25_12 = [2807, 4327, 9014, 21743, 25362, 42704]
# size_scm_15 = [14, 42, 269, 995, 4034, 9004]
# size_h50_15 = [990, 1724, 4385, 13178, 18176, 36016]
# size_h25_15 = [3845, 6777, 16884, 53389, 70953, 139238]
# size_scm_20 = [26, 82, 6881, 39646]
# size_h50_20 = [1553, 3104, 77723, 148246]
# size_h25_20 = [5913, 12039, 318673, 586482]
lengths = [10.0, 12.5, 15.0, 20.0]
n = [1, 2, 5, 7, 12]
n_20 = [1, 2, 7, 12]
m = [1, 2, 4, 6, 8]
m_20 = [1, 2, 6, 8]
time_scm_10 = [0.01, 0.01, 0.08, 0.24, 0.81]
time_h50_10 = [0.35, 0.57, 1.28, 2.97, 3.38]
time_h25_10 = [2.10, 3.41, 7.36, 20.02, 20.34]
time_scm_12 = [0.01, 0.01, 0.19, 0.70, 2.84]
time_h50_12 = [0.52, 1.02, 2.80, 9.14, 12.48]
time_h25_12 = [3.24, 6.56, 18.31, 78.87, 112.49]
time_scm_15 = [0.01, 0.04, 0.40, 2.05, 10.80]
time_h50_15 = [1.25, 2.70, 9.09, 43.86, 75.67]
time_h25_15 = [4.82, 11.75, 49.33, 747.40, 1666.67]
time_scm_20 = [0.02, 0.11, 26.31, 555.86]
time_h50_20 = [1.48, 4.15, 2414.15, 9704.11]
time_h25_20 = [10.26, 32.73, 46258.24, 171031.74]
size_scm_10 = [4, 18, 69, 157, 449]
size_h50_10 = [493, 679, 1211, 2240, 2288]
size_h25_10 = [1942, 2655, 4660, 8935, 8881]
size_scm_12 = [10, 18, 143, 398, 1314]
size_h50_12 = [722, 1080, 2340, 5443, 6548]
size_h25_12 = [2807, 4327, 9014, 21743, 25362]
size_scm_15 = [14, 42, 269, 995, 4034]
size_h50_15 = [990, 1724, 4385, 13178, 18176]
size_h25_15 = [3845, 6777, 16884, 53389, 70953]
size_scm_20 = [26, 82, 6881, 39646]
size_h50_20 = [1553, 3104, 77723, 148246]
size_h25_20 = [5913, 12039, 318673, 586482]

# Generate figures
fig: plt.Figure
ax: plt.Axes
fig, ax = plt.subplots(
    1,
    1,
    figsize=np.array([8, 8]) / 2.54,
)
linewidth_hag = 1.3
markersize_hag = 4
markeredgewidth_hag = 0.8
linewidth_scm = 1.4
markersize_scm = 4.5
markeredgewidth_scm = 0.8
style_h50 = ":^"
style_h25 = "--o"
style_scm = "-D"

# Time only graph
ax.plot(
    m,
    time_h50_10,
    style_h50,
    linewidth=linewidth_hag,
    markersize=markersize_hag,
    markeredgewidth=markeredgewidth_hag,
    markerfacecolor=color_hag[0],
    color=color_hag[0],
)
ax.plot(
    m,
    time_h25_10,
    style_h25,
    linewidth=linewidth_hag,
    markersize=markersize_hag,
    markeredgewidth=markeredgewidth_hag,
    markerfacecolor=color_hag[0],
    color=color_hag[0],
)
ax.plot(
    m,
    time_h50_12,
    style_h50,
    linewidth=linewidth_hag,
    markersize=markersize_hag,
    markeredgewidth=markeredgewidth_hag,
    markerfacecolor=color_hag[1],
    color=color_hag[1],
)
ax.plot(
    m,
    time_h25_12,
    style_h25,
    linewidth=linewidth_hag,
    markersize=markersize_hag,
    markeredgewidth=markeredgewidth_hag,
    markerfacecolor=color_hag[1],
    color=color_hag[1],
)
ax.plot(
    m,
    time_h50_15,
    style_h50,
    linewidth=linewidth_hag,
    markersize=markersize_hag,
    markeredgewidth=markeredgewidth_hag,
    markerfacecolor=color_hag[2],
    color=color_hag[2],
)
ax.plot(
    m,
    time_h25_15,
    style_h25,
    linewidth=linewidth_hag,
    markersize=markersize_hag,
    markeredgewidth=markeredgewidth_hag,
    markerfacecolor=color_hag[2],
    color=color_hag[2],
)
ax.plot(
    m_20,
    time_h50_20,
    style_h50,
    linewidth=linewidth_hag,
    markersize=markersize_hag,
    markeredgewidth=markeredgewidth_hag,
    markerfacecolor=color_hag[3],
    color=color_hag[3],
)
ax.plot(
    m_20,
    time_h25_20,
    style_h25,
    linewidth=linewidth_hag,
    markersize=markersize_hag,
    markeredgewidth=markeredgewidth_hag,
    markerfacecolor=color_hag[3],
    color=color_hag[3],
)
ax.set_yscale("log")
ax.set_xlim([0.7, 8.3])
ax.set_ylim([5e-3, 5e5])
ax.grid(True, which="major", linestyle=":", color="gray", linewidth=0.5, zorder=1)
ax.grid(True, which="minor", linestyle=":", color="gray", linewidth=0.3, zorder=1)
ax.minorticks_on()
ax.set_xlabel("$m$")
ax.set_ylabel("$t$ [s]")
fig.savefig("results/time_graph.png", dpi=1200, format="png", bbox_inches="tight")

# Time both
ax.plot(
    m,
    time_scm_10,
    style_scm,
    linewidth=linewidth_scm,
    markersize=markersize_scm,
    markeredgewidth=markeredgewidth_scm,
    markerfacecolor=color_scm[0],
    color=color_scm[0],
)
ax.plot(
    m,
    time_scm_12,
    style_scm,
    linewidth=linewidth_scm,
    markersize=markersize_scm,
    markeredgewidth=markeredgewidth_scm,
    markerfacecolor=color_scm[1],
    color=color_scm[1],
)
ax.plot(
    m,
    time_scm_15,
    style_scm,
    linewidth=linewidth_scm,
    markersize=markersize_scm,
    markeredgewidth=markeredgewidth_scm,
    markerfacecolor=color_scm[2],
    color=color_scm[2],
)
ax.plot(
    m_20,
    time_scm_20,
    style_scm,
    linewidth=linewidth_scm,
    markersize=markersize_scm,
    markeredgewidth=markeredgewidth_scm,
    markerfacecolor=color_scm[3],
    color=color_scm[3],
)
fig.savefig("results/time_both.png", dpi=1200, format="png", bbox_inches="tight")


########################################################################################
fig: plt.Figure
ax: plt.Axes
fig, ax = plt.subplots(
    1,
    1,
    figsize=np.array([8, 8]) / 2.54,
)

# Size only graph
ax.plot(
    m,
    size_h50_10,
    style_h50,
    linewidth=linewidth_hag,
    markersize=markersize_hag,
    markeredgewidth=markeredgewidth_hag,
    markerfacecolor=color_hag[0],
    color=color_hag[0],
)
ax.plot(
    m,
    size_h25_10,
    style_h25,
    linewidth=linewidth_hag,
    markersize=markersize_hag,
    markeredgewidth=markeredgewidth_hag,
    markerfacecolor=color_hag[0],
    color=color_hag[0],
)
ax.plot(
    m,
    size_h50_12,
    style_h50,
    linewidth=linewidth_hag,
    markersize=markersize_hag,
    markeredgewidth=markeredgewidth_hag,
    markerfacecolor=color_hag[1],
    color=color_hag[1],
)
ax.plot(
    m,
    size_h25_12,
    style_h25,
    linewidth=linewidth_hag,
    markersize=markersize_hag,
    markeredgewidth=markeredgewidth_hag,
    markerfacecolor=color_hag[1],
    color=color_hag[1],
)
ax.plot(
    m,
    size_h50_15,
    style_h50,
    linewidth=linewidth_hag,
    markersize=markersize_hag,
    markeredgewidth=markeredgewidth_hag,
    markerfacecolor=color_hag[2],
    color=color_hag[2],
)
ax.plot(
    m,
    size_h25_15,
    style_h25,
    linewidth=linewidth_hag,
    markersize=markersize_hag,
    markeredgewidth=markeredgewidth_hag,
    markerfacecolor=color_hag[2],
    color=color_hag[2],
)
ax.plot(
    m_20,
    size_h50_20,
    style_h50,
    linewidth=linewidth_hag,
    markersize=markersize_hag,
    markeredgewidth=markeredgewidth_hag,
    markerfacecolor=color_hag[3],
    color=color_hag[3],
)
ax.plot(
    m_20,
    size_h25_20,
    style_h25,
    linewidth=linewidth_hag,
    markersize=markersize_hag,
    markeredgewidth=markeredgewidth_hag,
    markerfacecolor=color_hag[3],
    color=color_hag[3],
)
ax.set_yscale("log")
ax.set_ylim([1e0, 1e6])
ax.set_xlim([0.7, 8.3])
ax.grid(True, which="major", linestyle=":", color="gray", linewidth=0.5, zorder=1)
ax.grid(True, which="minor", linestyle=":", color="gray", linewidth=0.3, zorder=1)
ax.minorticks_on()
ax.set_xlabel("$m$")
ax.set_ylabel("size (nodes/simplices)")
fig.savefig("results/size_graph.png", dpi=1200, format="png", bbox_inches="tight")

# Size both
ax.plot(
    m,
    size_scm_10,
    style_scm,
    linewidth=linewidth_scm,
    markersize=markersize_scm,
    markeredgewidth=markeredgewidth_scm,
    markerfacecolor=color_scm[0],
    color=color_scm[0],
)
ax.plot(
    m,
    size_scm_12,
    style_scm,
    linewidth=linewidth_scm,
    markersize=markersize_scm,
    markeredgewidth=markeredgewidth_scm,
    markerfacecolor=color_scm[1],
    color=color_scm[1],
)
ax.plot(
    m,
    size_scm_15,
    style_scm,
    linewidth=linewidth_scm,
    markersize=markersize_scm,
    markeredgewidth=markeredgewidth_scm,
    markerfacecolor=color_scm[2],
    color=color_scm[2],
)
ax.plot(
    m_20,
    size_scm_20,
    style_scm,
    linewidth=linewidth_scm,
    markersize=markersize_scm,
    markeredgewidth=markeredgewidth_scm,
    markerfacecolor=color_scm[3],
    color=color_scm[3],
)
fig.savefig("results/size_both.png", dpi=1200, format="png", bbox_inches="tight")
