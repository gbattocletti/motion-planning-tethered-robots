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
    ("env_1.yaml", 1, 10.0, "convex_hull"),
    ("env_1.yaml", 1, 10.0, "linear_homotopy"),
    ("env_1.yaml", 1, 10.0, "local_visibility_homotopy"),
    ("env_1.yaml", 1, 12.0, "convex_hull"),
    ("env_1.yaml", 1, 12.0, "linear_homotopy"),
    ("env_1.yaml", 1, 12.0, "local_visibility_homotopy"),
    ("env_1.yaml", 1, 15.0, "convex_hull"),
    ("env_1.yaml", 1, 15.0, "linear_homotopy"),
    ("env_1.yaml", 1, 15.0, "local_visibility_homotopy"),
    ("env_2.yaml", 2, 10.0, "convex_hull"),
    ("env_2.yaml", 2, 10.0, "linear_homotopy"),
    ("env_2.yaml", 2, 10.0, "local_visibility_homotopy"),
    ("env_2.yaml", 2, 12.0, "convex_hull"),
    ("env_2.yaml", 2, 12.0, "linear_homotopy"),
    ("env_2.yaml", 2, 12.0, "local_visibility_homotopy"),
    ("env_2.yaml", 2, 15.0, "convex_hull"),
    ("env_2.yaml", 2, 15.0, "linear_homotopy"),
    ("env_2.yaml", 2, 15.0, "local_visibility_homotopy"),
    ("env_3.yaml", 6, 10.0, "convex_hull"),
    ("env_3.yaml", 6, 10.0, "linear_homotopy"),
    ("env_3.yaml", 6, 10.0, "local_visibility_homotopy"),
    ("env_3.yaml", 6, 12.0, "convex_hull"),
    ("env_3.yaml", 6, 12.0, "linear_homotopy"),
    ("env_3.yaml", 6, 12.0, "local_visibility_homotopy"),
    ("env_3.yaml", 6, 15.0, "convex_hull"),
    ("env_3.yaml", 6, 15.0, "linear_homotopy"),
    ("env_3.yaml", 6, 15.0, "local_visibility_homotopy"),
    ("env_4.yaml", 8, 10.0, "convex_hull"),
    ("env_4.yaml", 8, 10.0, "linear_homotopy"),
    ("env_4.yaml", 8, 10.0, "local_visibility_homotopy"),
    ("env_4.yaml", 8, 12.0, "convex_hull"),
    ("env_4.yaml", 8, 12.0, "linear_homotopy"),
    ("env_4.yaml", 8, 12.0, "local_visibility_homotopy"),
    ("env_4.yaml", 8, 15.0, "convex_hull"),
    ("env_4.yaml", 8, 15.0, "linear_homotopy"),
    ("env_4.yaml", 8, 15.0, "local_visibility_homotopy"),
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
# The last three columns are repeated for multiple grid resolutions (1, 0.5)
results_table = np.zeros((len(eval_cases), 14), dtype=float)

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

    ####################################################################################
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

    # Compute entanglement-admissible area ratio in simplicial complex
    area_reachable: float = 0
    area_entanglement_admissible: float = 0
    for tri, ent in zip(triang.triangles_lift, triang.entanglement_triangles_lift):
        i1, i2, i3 = tri  # triangle indexes
        [x1, y1] = triang.vertices[triang.vertices_lift[i1][0]]  # coords vertex 1
        [x2, y2] = triang.vertices[triang.vertices_lift[i2][0]]  # coords vertex 2
        [x3, y3] = triang.vertices[triang.vertices_lift[i3][0]]  # coords vertex 3
        area = 0.5 * abs(x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))  # tri area
        area_reachable += area
        if ent is True:
            area_entanglement_admissible += area
        else:
            pass
    area_ratio_sc = area_entanglement_admissible / area_reachable * 100

    # Store results for simplicial complex
    results_table[idx, 5] = len(triang.triangles_lift)  # triangs in simplicial complex
    results_table[idx, 6] = t  # time simplicial complex
    results_table[idx, 7] = area_ratio_sc

    ####################################################################################
    # Coarse resolution
    print(
        f"{CmdColors.WARNING}[GridGraph]{CmdColors.ENDC} Running graph with "
        "resolution 1.0."
    )

    # Create graph
    graph_1 = GridGraph(env)
    graph_1.INFO = True  # Enable info
    graph_1.DEBUG = False  # Disable verbose debug prints
    graph_1.set_max_dist(l)
    graph_1.n_max = 1_000_000  # increase max number of nodes for this test
    graph_1.set_entanglement_definition(ent_def)  # set entanglement definition
    graph_1.set_grid_resolution(1.0, 1.0)

    # Compute homotopy augmented graph with entanglement check
    t_1 = (
        timeit.timeit(
            lambda graph_1=graph_1: graph_1.build_homotopy_augmented_graph(
                check_entanglement=True,
            ),
            number=n_runs,
        )
        / n_runs
    )

    # Compute entanglement-admissible node ratio in homotopy augmented graph
    n_reachable: int = len(graph_1.vertices_lift)
    n_entanglement_admissible: int = sum(graph_1.entanglement_vertices_lift)
    area_ratio_hag = n_entanglement_admissible / n_reachable * 100

    # Store results for homotopy augmented graph
    results_table[idx, 8] = len(graph_1.vertices_lift)  # nodes homotopy augmented graph
    results_table[idx, 9] = t_1  # time homotopy augmented graph
    results_table[idx, 10] = area_ratio_hag  # entanglement-admissible node ratio

    ####################################################################################
    # Medium resolution
    print(
        f"{CmdColors.WARNING}[GridGraph]{CmdColors.ENDC} Running graph with "
        "resolution 0.5."
    )

    # Create graph
    graph_2 = GridGraph(env)
    graph_2.INFO = True  # Enable info
    graph_2.DEBUG = False  # Disable verbose debug prints
    graph_2.set_max_dist(l)
    graph_2.n_max = 1_000_000  # increase max number of nodes for this test
    graph_2.set_entanglement_definition(ent_def)  # set entanglement definition
    graph_2.set_grid_resolution(0.5, 0.5)

    # Compute homotopy augmented graph with entanglement check
    t_2 = (
        timeit.timeit(
            lambda graph_2=graph_2: graph_2.build_homotopy_augmented_graph(
                check_entanglement=True,
            ),
            number=n_runs,
        )
        / n_runs
    )

    # Compute entanglement-admissible node ratio in homotopy augmented graph
    n_reachable: int = len(graph_2.vertices_lift)
    n_entanglement_admissible: int = sum(graph_2.entanglement_vertices_lift)
    area_ratio_hag = n_entanglement_admissible / n_reachable * 100

    results_table[idx, 11] = len(graph_2.vertices_lift)  # nodes h-augmented graph
    results_table[idx, 12] = t_2  # time homotopy augmented graph
    results_table[idx, 13] = area_ratio_hag  # entanglement-admissible node ratio

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
        "graph_coarse": graph_1,
        "graph_medium": graph_2,
        "results": results_table,
    }
    pickled = pickle.dumps(data)  # dump data dictionary in pickle file
    optimized = optimize(pickled)  # optimize the pickle file
    with open(filename, "wb") as f:
        f.write(optimized)

    # Print results table and save to CSV file (updated after each eval case)
    with open("results/comparison_results.csv", "w", encoding="utf-8") as f:
        # Print header
        print("env\tm\tl\tdef\t|T_2|\t|T'_2|\tt\t% T'\t|G'|\tt\t% G\t|G'|\tt\t% G")
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

            # entanglement definition (def)
            print(f"\t{int(row[3])}", end="")
            f.write(f"{int(row[3])},")

            # number of triangles in the base triangulation
            print(f"\t{int(row[4])}", end="")
            f.write(f"{int(row[4])},")

            # num triangles simplicial complex
            print(f"\t{int(row[5])}", end="")
            f.write(f"{int(row[5])},")

            # time simplicial complex
            print(f"\t{row[6]:.2f}", end="")
            f.write(f"{row[6]:.2f},")

            # entanglement-admissible area in simplicial complex
            print(f"\t{row[7]:.2f}", end="")
            f.write(f"{row[7]:.2f},")

            # number of nodes in homotopy augmented graph
            print(f"\t{int(row[8])}", end="")
            f.write(f"{int(row[8])},")

            # time homotopy augmented graph
            print(f"\t{row[9]:.2f}", end="")
            f.write(f"{row[9]:.2f}")

            # entanglement-admissible area in homotopy augmented graph
            print(f"\t{row[10]:.2f}", end="")
            f.write(f"{row[10]:.2f}")

            # number of nodes in homotopy augmented graph
            print(f"\t{int(row[11])}", end="")
            f.write(f"{int(row[11])},")

            # time homotopy augmented graph
            print(f"\t{row[12]:.2f}", end="")
            f.write(f"{row[12]:.2f}")

            # entanglement-admissible area in homotopy augmented graph
            print(f"\t{row[13]:.2f}")
            f.write(f"{row[13]:.2f}\n")
