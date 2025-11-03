"""
Generates the lifted simplicial complex and homotopy augmented graph while measuring
time and graph statistics. Returns a table with the measured statistics.
"""

import os
import pickle
import timeit
from datetime import datetime
from pickletools import optimize

import numpy as np

from tethered_planning.env import env_2d
from tethered_planning.env.grid_graph import GridGraph
from tethered_planning.env.triangulation import Triangulation
from tethered_planning.utils.settings import Settings

# Evaluation cases
# Each case is defined as a tuple (env_filename, obstacles m, tether length l), where:
# - env_filename: str, name of the environment file
# - m: int, number of obstacles in the environment that give rise to multiple homotopy
#      classes (manually inserted from visual inspection)
# - l: float, tether length
eval_cases = [
    ("env_1.yaml", 1, 10.0),
    ("env_1.yaml", 1, 15.0),
    ("env_1.yaml", 1, 20.0),
    ("env_2.yaml", 2, 10.0),
    ("env_2.yaml", 2, 15.0),
    ("env_2.yaml", 2, 20.0),
    ("env_3.yaml", 6, 10.0),
    ("env_3.yaml", 6, 15.0),
    ("env_3.yaml", 6, 20.0),
    ("env_4.yaml", 8, 10.0),
    ("env_4.yaml", 8, 15.0),
    ("env_4.yaml", 8, 20.0),
]

# Move to script directory
abspath = os.path.abspath(__file__)
dir_name = os.path.dirname(abspath)
os.chdir(dir_name)

# Initialize settings
settings = Settings(create_sim_folder=False)
n_runs = 1  # number of runs for time averaging

# Initialize results table
# Results table columns:
#   - text index (int)
#   - num obstacles m (int)
#   - tether length (float)
#   - num of nodes in simplicial complex (primal lifted graph) (int)
#   - num of triangles in simplicial complex (nodes in dual lifted graph) (int)
#   - computation time for simplicial complex (float, seconds)
#   - num of nodes in homotopy augmented graph (int)
#   - computation time for homotopy augmented graph (float, seconds)
results_table = np.zeros((len(eval_cases), 8), dtype=float)


# Loop over evaluation cases
for idx, case in enumerate(eval_cases):
    print(
        f"Running evaluation {idx+1} [{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]"
    )

    # Unpack case
    env_name, m, l = case
    results_table[idx, 0] = idx + 1  # text index
    results_table[idx, 1] = m  # num obstacles
    results_table[idx, 2] = l  # tether length

    # Generate env for the test case
    settings.env_name = env_name  # change environment
    env = env_2d.Env2D(settings)

    # Create triangulation
    triang = Triangulation(env)
    triang.DEBUG = True  # Enable debug info
    triang.triangulate()
    triang.max_lifted_triangles = 10_000
    triang.set_max_dist(l)  # max tether length
    t = (
        timeit.timeit(
            lambda triang=triang: triang.lift_triangulation(),
            number=n_runs,
        )
        / n_runs
    )
    results_table[idx, 3] = (
        len(triang.vertices_dual_lift * 3) - len(triang.edges_dual_lift) * 2
    )  # nodes simplicial complex (vertices of triangles, or primal lifted graph)
    results_table[idx, 4] = len(
        triang.vertices_dual_lift
    )  # triangles simplicial complex
    results_table[idx, 5] = t  # time simplicial complex

    # Create homotopy augmented graph
    graph = GridGraph(env)
    graph.DEBUG = True  # Enable debug info
    graph.set_max_dist(l)
    graph.n_max = 200_000  # increase max number of nodes for this test
    graph.set_grid_resolution(0.5, 0.5)
    t = (
        timeit.timeit(
            lambda graph=graph: graph.build_homotopy_augmented_graph(),
            number=n_runs,
        )
        / n_runs
    )
    results_table[idx, 6] = len(graph.vertices_lift)  # nodes homotopy augmented graph
    results_table[idx, 7] = t  # time homotopy augmented graph

    # Print intermediate results
    print(
        f"Completed evaluation {idx+1} [{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]"
    )

    # Save pickle files of the generated data structures for later analysis
    filename = f"results/comparison-{idx+1}.pkl"
    data = {
        "settings": settings,
        "runs": n_runs,
        "env": env,
        "triangulation": triang,
        "graph": graph,
        "results": results_table,
    }
    pickled = pickle.dumps(data)  # dump data dictionary in pickle file
    optimized = optimize(pickled)  # optimize the pickle file
    with open(filename, "wb") as f:
        f.write(optimized)

    # Print results table and save to CSV file (updated after each eval case)
    with open("results/comparison_results.csv", "w", encoding="utf-8") as f:
        for i, row in enumerate(results_table):
            # index
            print(f"#{int(row[0])}: ", end="")
            f.write(f"{int(row[0])},")

            # num obstacles (m)
            print(f"\t{int(row[1])}", end="")
            f.write(f"{int(row[1])},")

            # tether length (l)
            print(f"\t{row[2]:.1f}", end="")
            f.write(f"{row[2]:.1f},")

            # num triangles simplicial complex
            print(f"\t{int(row[3])}", end="")
            f.write(f"{int(row[3])},")

            # num nodes simplicial complex
            print(f"\t{int(row[4])}", end="")
            f.write(f"{int(row[4])},")

            # time simplicial complex
            print(f"\t{row[5]:.2f} s", end="")
            f.write(f"{row[5]:.2f},")

            # num nodes homotopy augmented graph
            print(f"\t{int(row[6])}", end="")
            f.write(f"{int(row[6])},")

            # time homotopy augmented graph
            print(f"\t{row[7]:.2f} s")
            f.write(f"{row[7]:.2f}\n")
