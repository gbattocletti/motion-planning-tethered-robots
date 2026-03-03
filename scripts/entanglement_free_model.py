"""
Generates the simplicial complex model of the entanglement-free workspace and an
entanglement-free homotopy augmented graph, and compares their computation time and
memory occupance.
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
from tethered_planning.utils.colors import CmdColors
from tethered_planning.utils.settings import Settings

# Evaluation cases
# Each case is defined as a tuple (env, m, l, def), where:
# - env: str, name of the environment file
# - m: int, number of obstacles in the environment that give rise to multiple homotopy
#      classes (currently manually inserted from visual inspection)
# - l: float, tether length
# - def: entanglement_definition: str, definition of entanglement to use to determine
#      the entanglement of the curves in the graphs.
eval_cases = [
    ("env_1.yaml", 1, 12.0, "convex_hull"),
    ("env_1.yaml", 1, 12.0, "linear_homotopy"),
    ("env_1.yaml", 1, 12.0, "local_visibility_homotopy"),
    ("env_2.yaml", 2, 12.0, "convex_hull"),
    ("env_2.yaml", 2, 12.0, "linear_homotopy"),
    ("env_2.yaml", 2, 12.0, "local_visibility_homotopy"),
]

# Move to script directory
abspath = os.path.abspath(__file__)
dir_name = os.path.dirname(abspath)
os.chdir(dir_name)

# Initialize settings
settings = Settings(create_sim_folder=True)
n_runs = 5  # number of runs for time averaging

# Initialize results table
# Results table columns:
#   - env name (str)
#   - num obstacles m (int)
#   - tether length (float)
#   - entanglement definition (str)
#   - num of triangles in base triangulation (int)
#   - num of triangles in simplicial complex (nodes in dual lifted graph) (int)
#   - computation time for simplicial complex (float, seconds)
#   - % of entanglement-admissible area in simplicial complex (float, 0-100)
#   - num of nodes in homotopy augmented graph (int)
#   - computation time for homotopy augmented graph (float, seconds)
#   - % of entanglement-admissible nodes in homotopy augmented graph (float, 0-100)
results_table = np.zeros((len(eval_cases), 11), dtype=float)

# Loop over evaluation cases
for idx, case in enumerate(eval_cases):
    print(
        f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Running "
        f"evaluation {idx+1} (env {case[0][4]} l={case[2]})"
    )

    # Unpack case
    env_name, m, l, ent_def = case
    results_table[idx, 0] = int(env_name[4])  # env name
    results_table[idx, 1] = m  # num obstacles
    results_table[idx, 2] = l  # tether length
    match ent_def:
        case "convex_hull":
            results_table[idx, 3] = 1
        case "linear_homotopy":
            results_table[idx, 3] = 2
        case "local_visibility_homotopy":
            results_table[idx, 3] = 3

    # Generate env for the test case
    settings.env_name = env_name  # change environment
    env = env_2d.Env2D(settings)

    # Compute base triangulation
    triang = Triangulation(env)
    triang.INFO = True  # Enable info
    triang.DEBUG = False  # Disable verbose debug prints
    triang.triangulate()
    results_table[idx, 4] = triang.triangles.shape[0]

    # Define settings for lifted simplicial complex
    triang.max_lifted_triangles = 100_000
    triang.set_max_dist(l)  # max tether length
    triang.set_entanglement_definition(ent_def)  # set entanglement definition

    # Create lifted simplicial complex
    print(f"{CmdColors.WARNING}[Triang]{CmdColors.ENDC} Running triangulation.")
    t = (
        timeit.timeit(
            lambda triang=triang: triang.lift_triangulation(
                check_entanglement=True,
            ),
            number=n_runs,
        )
        / n_runs
    )
    results_table[idx, 5] = len(triang.triangles_lift)  # triangs in simplicial complex
    results_table[idx, 6] = t  # time simplicial complex
    results_table[idx, 7] = 0  # TODO

    # Create homotopy augmented graph
    graph = GridGraph(env)
    graph.INFO = True  # Enable info
    graph.DEBUG = False  # Disable verbose debug prints
    graph.set_max_dist(l)
    graph.n_max = 1_000_000  # increase max number of nodes for this test
    graph.set_entanglement_definition(ent_def)  # set entanglement definition

    # Medium resolution
    print(
        f"{CmdColors.WARNING}[GridGraph]{CmdColors.ENDC} Running graph with "
        "resolution 1."
    )
    graph.set_grid_resolution(1, 1)
    t = (
        timeit.timeit(
            lambda graph=graph: graph.build_homotopy_augmented_graph(
                check_entanglement=True,
            ),
            number=n_runs,
        )
        / n_runs
    )
    results_table[idx, 8] = len(graph.vertices_lift)  # nodes homotopy augmented graph
    results_table[idx, 9] = t  # time homotopy augmented graph
    results_table[idx, 10] = 0  # TODO

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
        # Print header
        print("env\tm\tl\t|T_2|\t|T'_2|\tt\t|G'|.5\tt.5\t|G'|.25\tt.25")
        for i, row in enumerate(results_table):
            # index
            print(f"{str(int(row[0]))}", end="")
            f.write(f"{str(int(row[0]))},")

            # num obstacles (m)
            print(f"\t{int(row[1])}", end="")
            f.write(f"{int(row[1])},")

            # tether length (l)
            print(f"\t{row[2]:.1f}", end="")
            f.write(f"{row[2]:.1f},")

            # number of triangles in the base triangulation
            print(f"\t{int(row[3])}", end="")
            f.write(f"{int(row[3])},")

            # num triangles simplicial complex
            print(f"\t{int(row[4])}", end="")
            f.write(f"{int(row[4])},")

            # time simplicial complex
            print(f"\t{row[5]:.2f}", end="")
            f.write(f"{row[5]:.2f},")

            # homotopy augmented graph with resolution 0.5
            print(f"\t{int(row[6])}", end="")
            f.write(f"{int(row[6])},")
            print(f"\t{row[7]:.2f}", end="")
            f.write(f"{row[7]:.2f}")

            # homotopy augmented graph with resolution 0.1
            print(f"\t{int(row[8])}", end="")
            f.write(f"{int(row[8])},")
            print(f"\t{row[9]:.2f}")
            f.write(f"{row[9]:.2f}\n")
