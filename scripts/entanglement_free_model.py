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
settings = Settings(create_sim_folder=False)
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
#   - num of triangles in simplicial complex with entanglement check (int)
#   - computation time for simplicial complex with entanglement check (float, seconds)
#   - % of entanglement-admissible area in simplicial complex (float, 0-100)
#   - num of nodes in homotopy augmented graph (int)
#   - computation time for homotopy augmented graph (float, seconds)
#   - num of nodes in homotopy augmented graph with entanglement check (int)
#   - computation time for homotopy augmented graph with entanglement check (float)
#   - % of entanglement-admissible nodes in homotopy augmented graph (float, 0-100)
# The last three columns are repeated for multiple grid resolutions (1, 0.5, 0.2)
results_table = np.zeros((len(eval_cases), 25), dtype=float)

# Loop over evaluation cases
for idx, case in enumerate(eval_cases):
    print(
        f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Running "
        f"evaluation {idx+1} (env {case[0][4]} l={case[2]})"
    )

    # Unpack case
    env_name, m, l, ent_def = case

    # Store info in table
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
    # Compute base triangulation + lifted triangulation
    print(f"{CmdColors.WARNING}[Triang]{CmdColors.ENDC} Running triangulation.")

    # Create triangulation
    triang = Triangulation(env)
    triang.INFO = True  # Enable info
    triang.DEBUG = False  # Disable verbose debug prints
    triang.triangulate()

    # Store results in table (only for the first eval case)
    results_table[idx, 4] = triang.triangles.shape[0]

    # Define settings for lifted simplicial complex
    triang.max_lifted_triangles = 100_000
    triang.set_max_dist(l)  # max tether length

    # Create lifted simplicial complex
    t = (
        timeit.timeit(
            lambda triang=triang: triang.lift_triangulation(
                check_entanglement=False,
            ),
            number=n_runs,
        )
        / n_runs
    )

    # Store results in table
    results_table[idx, 5] = len(triang.triangles_lift)  # nr of triangs
    results_table[idx, 6] = t  # time simplicial complex

    ####################################################################################
    # Compute lifted triangulation with entanglement check
    print(
        f"{CmdColors.WARNING}[Triang]{CmdColors.ENDC} Running triangulation "
        "with entanglement check."
    )

    # Create triangulation
    triang_ent = Triangulation(env)
    triang_ent.INFO = True  # Enable info
    triang_ent.DEBUG = False  # Disable verbose debug prints
    triang_ent.triangulate()

    # Define settings for lifted simplicial complex
    triang_ent.max_lifted_triangles = 100_000
    triang_ent.set_max_dist(l)  # max tether length
    triang_ent.set_entanglement_definition(ent_def)  # set entanglement def

    # Create lifted simplicial complex
    t = (
        timeit.timeit(
            lambda triang_ent=triang_ent: triang_ent.lift_triangulation(
                check_entanglement=True,
            ),
            number=n_runs,
        )
        / n_runs
    )

    # Compute entanglement-admissible area ratio in simplicial complex
    area_reachable: float = 0
    area_entanglement_admissible: float = 0
    for tri, ent in zip(
        triang_ent.triangles_lift,
        triang_ent.entanglement_triangles_lift,
    ):
        i1, i2, i3 = tri  # triangle indexes
        [x1, y1] = triang_ent.vertices[
            triang_ent.vertices_lift[i1][0]
        ]  # coords vertex 1
        [x2, y2] = triang_ent.vertices[
            triang_ent.vertices_lift[i2][0]
        ]  # coords vertex 2
        [x3, y3] = triang_ent.vertices[
            triang_ent.vertices_lift[i3][0]
        ]  # coords vertex 3
        area = 0.5 * abs(x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))  # tri area
        area_reachable += area
        if ent is True:
            area_entanglement_admissible += area
        else:
            pass
    area_ratio_sc = area_entanglement_admissible / area_reachable * 100

    # Store results in table
    results_table[idx, 7] = sum(triang_ent.entanglement_triangles_lift)  # nr of triangs
    results_table[idx, 8] = t  # time simplicial complex
    results_table[idx, 9] = area_ratio_sc

    ####################################################################################
    # Homotopy augmented graph with coarse resolution
    print(
        f"{CmdColors.WARNING}[GridGraph]{CmdColors.ENDC} Running graph with "
        "resolution 1.0."
    )

    # Create graph
    graph_1 = GridGraph(env)
    graph_1.INFO = True  # Enable info
    graph_1.DEBUG = False  # Disable verbose debug prints
    graph_1.set_max_dist(l)
    graph_1.n_max = 1_000_000
    graph_1.set_grid_resolution(1.0, 1.0)

    # Compute homotopy augmented graph with entanglement check
    t_1 = (
        timeit.timeit(
            lambda graph_1=graph_1: graph_1.build_homotopy_augmented_graph(
                check_entanglement=False,
            ),
            number=n_runs,
        )
        / n_runs
    )

    # Store results in table
    results_table[idx, 10] = len(graph_1.vertices_lift)  # nodes h augmented graph
    results_table[idx, 11] = t_1  # time homotopy augmented graph

    ####################################################################################
    # Homotopy augmented graph with coarse resolution and entanglement check
    print(
        f"{CmdColors.WARNING}[GridGraph]{CmdColors.ENDC} Running graph with "
        "resolution 1.0 and entanglement check."
    )

    # Create graph
    graph_1_ent = GridGraph(env)
    graph_1_ent.INFO = True  # Enable info
    graph_1_ent.DEBUG = False  # Disable verbose debug prints
    graph_1_ent.set_max_dist(l)
    graph_1_ent.n_max = 1_000_000
    graph_1_ent.set_grid_resolution(1.0, 1.0)
    graph_1_ent.set_entanglement_definition(ent_def)  # set entanglement definition

    # Compute homotopy augmented graph with entanglement check
    t_1_ent = (
        timeit.timeit(
            lambda graph_1_ent=graph_1_ent: graph_1_ent.build_homotopy_augmented_graph(
                check_entanglement=True,
            ),
            number=n_runs,
        )
        / n_runs
    )

    # Compute entanglement-admissible node ratio in homotopy augmented graph
    n_reachable: int = len(graph_1_ent.vertices_lift)
    n_entanglement_admissible: int = sum(graph_1_ent.entanglement_vertices_lift)
    area_ratio_hag = n_entanglement_admissible / n_reachable * 100

    # Store results in table
    results_table[idx, 12] = n_entanglement_admissible  # nodes h augmented graph
    results_table[idx, 13] = t_1_ent  # time homotopy augmented graph
    results_table[idx, 14] = area_ratio_hag  # entanglement-admissible node ratio

    ####################################################################################
    # Homotopy augmented graph with medium resolution
    print(
        f"{CmdColors.WARNING}[GridGraph]{CmdColors.ENDC} Running graph with "
        "resolution 0.5."
    )

    # Create graph
    graph_2 = GridGraph(env)
    graph_2.INFO = True  # Enable info
    graph_2.DEBUG = False  # Disable verbose debug prints
    graph_2.set_max_dist(l)
    graph_2.n_max = 1_000_000
    graph_2.set_grid_resolution(0.5, 0.5)

    # Compute homotopy augmented graph with entanglement check
    t_2 = (
        timeit.timeit(
            lambda graph_2=graph_2: graph_2.build_homotopy_augmented_graph(
                check_entanglement=False,
            ),
            number=n_runs,
        )
        / n_runs
    )

    # Store results in table
    results_table[idx, 15] = len(graph_2.vertices_lift)  # nodes h-augmented graph
    results_table[idx, 16] = t_2  # time homotopy augmented graph

    ####################################################################################
    # Homotopy augmented graph with medium resolution and entanglement check
    print(
        f"{CmdColors.WARNING}[GridGraph]{CmdColors.ENDC} Running graph with "
        "resolution 0.5 and entanglement check."
    )

    # Create graph
    graph_2_ent = GridGraph(env)
    graph_2_ent.INFO = True  # Enable info
    graph_2_ent.DEBUG = False  # Disable verbose debug prints
    graph_2_ent.set_max_dist(l)
    graph_2_ent.n_max = 1_000_000
    graph_2_ent.set_grid_resolution(0.5, 0.5)
    graph_2_ent.set_entanglement_definition(ent_def)  # set entanglement definition

    # Compute homotopy augmented graph with entanglement check
    t_2_ent = (
        timeit.timeit(
            lambda graph_2_ent=graph_2_ent: graph_2_ent.build_homotopy_augmented_graph(
                check_entanglement=True,
            ),
            number=n_runs,
        )
        / n_runs
    )

    # Compute entanglement-admissible node ratio in homotopy augmented graph
    n_reachable: int = len(graph_2_ent.vertices_lift)
    n_entanglement_admissible: int = sum(graph_2_ent.entanglement_vertices_lift)
    area_ratio_hag = n_entanglement_admissible / n_reachable * 100

    # Store results in table
    results_table[idx, 17] = n_entanglement_admissible  # nodes h-augmented graph
    results_table[idx, 18] = t_2_ent  # time homotopy augmented graph
    results_table[idx, 19] = area_ratio_hag  # entanglement-admissible node ratio

    ####################################################################################
    # Homotopy augmented graph with fine resolution
    print(
        f"{CmdColors.WARNING}[GridGraph]{CmdColors.ENDC} Running graph with "
        "resolution 0.2."
    )

    # Create graph
    graph_3 = GridGraph(env)
    graph_3.INFO = True  # Enable info
    graph_3.DEBUG = False  # Disable verbose debug prints
    graph_3.set_max_dist(l)
    graph_3.n_max = 5_000_000
    graph_3.set_grid_resolution(0.2, 0.2)

    # Compute homotopy augmented graph with entanglement check
    t_3 = (
        timeit.timeit(
            lambda graph_3=graph_3: graph_3.build_homotopy_augmented_graph(
                check_entanglement=False,
            ),
            number=n_runs,
        )
        / n_runs
    )

    # Store results in table
    results_table[idx, 20] = len(graph_3.vertices_lift)  # nodes h-augmented graph
    results_table[idx, 21] = t_3  # time homotopy augmented graph

    ####################################################################################
    # Homotopy augmented graph with fine resolution and entanglement check
    print(
        f"{CmdColors.WARNING}[GridGraph]{CmdColors.ENDC} Running graph with "
        "resolution 0.2 and entanglement check."
    )

    # Create graph
    graph_3_ent = GridGraph(env)
    graph_3_ent.INFO = True  # Enable info
    graph_3_ent.DEBUG = False  # Disable verbose debug prints
    graph_3_ent.set_max_dist(l)
    graph_3_ent.n_max = 5_000_000  # increase max number of nodes for this test
    graph_3_ent.set_grid_resolution(0.2, 0.2)
    graph_3_ent.set_entanglement_definition(ent_def)  # set entanglement definition

    # Compute homotopy augmented graph with entanglement check
    t_3_ent = (
        timeit.timeit(
            lambda graph_3_ent=graph_3_ent: graph_3_ent.build_homotopy_augmented_graph(
                check_entanglement=True,
            ),
            number=n_runs,
        )
        / n_runs
    )

    # Compute entanglement-admissible node ratio in homotopy augmented graph
    n_reachable: int = len(graph_3_ent.vertices_lift)
    n_entanglement_admissible: int = sum(graph_3_ent.entanglement_vertices_lift)
    area_ratio_hag = n_entanglement_admissible / n_reachable * 100

    # Store results in table
    results_table[idx, 22] = n_entanglement_admissible  # nodes h-augmented graph
    results_table[idx, 23] = t_3_ent  # time homotopy augmented graph
    results_table[idx, 24] = area_ratio_hag  # entanglement-admissible node ratio

    ####################################################################################
    # Save iteration rdesults in pickle file for later analysis
    # A separate pkl file is saved for each evaluation case, containing all the data
    # structures (simplicial complexes and graphs) created during that iteration.
    filename = f"results/comparison-{idx+1}.pkl"
    data = {
        "settings": settings,
        "runs": n_runs,
        "env": env,
        "triangulation": triang,
        "triangulation_entanglement": triang_ent,
        "graph_coarse": graph_1,
        "graph_coarse_entanglement": graph_1_ent,
        "graph_medium": graph_2,
        "graph_medium_entanglement": graph_2_ent,
        "graph_fine": graph_3,
        "graph_fine_entanglement": graph_3_ent,
        "results": results_table,
    }
    pickled = pickle.dumps(data)  # dump data dictionary in pickle file
    optimized = optimize(pickled)  # optimize the pickle file
    with open(filename, "wb") as f:
        f.write(optimized)

    ####################################################################################
    # Print & save intermediate results
    print(
        f"Completed evaluation {idx+1} [{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]"
    )

    # Print results table and save to CSV file (updated after each eval case)
    with open("results/comparison_results.csv", "w", encoding="utf-8") as f:
        # Print header
        print(
            "env\tm\tl\tdef\t|T|\t|R|\tt\t|N|\tt\t% N"
            "\t|G|\tt\t|G'|\tt\t% G'\t|G|\tt\t|G'|\tt\t% G'\t|G|\tt\t|G'|\tt\t% G'"
        )
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

            # num triangles simplicial complex with entanglement check
            print(f"\t{int(row[7])}", end="")
            f.write(f"{int(row[7])},")

            # time simplicial complex with entanglement check
            print(f"\t{row[8]:.2f}", end="")
            f.write(f"{row[8]:.2f},")

            # entanglement-admissible area in simplicial complex with entanglement check
            print(f"\t{row[9]:.2f}", end="")
            f.write(f"{row[9]:.2f},")

            # RESOLUTION 1
            # number of nodes in homotopy augmented graph
            print(f"\t{int(row[10])}", end="")
            f.write(f"{int(row[10])},")

            # time homotopy augmented graph
            print(f"\t{row[11]:.2f}", end="")
            f.write(f"{row[11]:.2f}")

            # number of nodes in homotopy augmented graph with entanglement check
            print(f"\t{int(row[12])}", end="")
            f.write(f"{int(row[12])},")

            # time homotopy augmented graph with entanglement check
            print(f"\t{row[13]:.2f}", end="")
            f.write(f"{row[13]:.2f}")

            # entanglement-admissible area in homotopy augmented graph
            print(f"\t{row[14]:.2f}", end="")
            f.write(f"{row[14]:.2f}")

            # RESOLUTION 0.5
            # number of nodes in homotopy augmented graph
            print(f"\t{int(row[15])}", end="")
            f.write(f"{int(row[15])},")

            # time homotopy augmented graph
            print(f"\t{row[16]:.2f}", end="")
            f.write(f"{row[16]:.2f}")

            # number of nodes in homotopy augmented graph with entanglement check
            print(f"\t{int(row[17])}", end="")
            f.write(f"{int(row[17])},")

            # time homotopy augmented graph with entanglement check
            print(f"\t{row[18]:.2f}", end="")
            f.write(f"{row[18]:.2f}")

            # entanglement-admissible area in homotopy augmented graph
            print(f"\t{row[19]:.2f}", end="")
            f.write(f"{row[19]:.2f}")

            # RESOLUTION 0.2
            # number of nodes in homotopy augmented graph
            print(f"\t{int(row[20])}", end="")
            f.write(f"{int(row[20])},")

            # time homotopy augmented graph
            print(f"\t{row[21]:.2f}", end="")
            f.write(f"{row[21]:.2f}")

            # number of nodes in homotopy augmented graph with entanglement check
            print(f"\t{int(row[22])}", end="")
            f.write(f"{int(row[22])},")

            # time homotopy augmented graph with entanglement check
            print(f"\t{row[23]:.2f}", end="")
            f.write(f"{row[23]:.2f}")

            # entanglement-admissible area in homotopy augmented graph
            print(f"\t{row[24]:.2f}")
            f.write(f"{row[24]:.2f}\n")
