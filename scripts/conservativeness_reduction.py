"""
Generates a simplicial complex with conservativeness reduction to evaluate the benefits
of the addition of additional triangles. The script evaluates multiple environments and
tether lengths. The resulting simplicial complexes (with and without the extra
triangles) are compared h-augmented graphs with different grid resolutions, and
particularly with the one having the finest respolution which is used as 'best
approximation' of the real reachable region. Each evaluation is run only once as
computation time is not of interest.
"""

import os
import pickle
from datetime import datetime
from pickletools import optimize

import numpy as np

from tethered_planning.env import env_2d
from tethered_planning.env.grid_graph import GridGraph
from tethered_planning.env.triangulation import Triangulation
from tethered_planning.utils.colors import CmdColors
from tethered_planning.utils.settings import Settings

# Evaluation cases
# Each case is defined as a tuple (env, l), where:
# - env: str, name of the environment file
# - l: float, tether length
eval_cases = [
    ("env_1.yaml", 10.0),
    ("env_1.yaml", 12.0),
    ("env_1.yaml", 15.0),
    ("env_2.yaml", 10.0),
    ("env_2.yaml", 12.0),
    ("env_2.yaml", 15.0),
    ("env_3.yaml", 10.0),
    ("env_3.yaml", 12.0),
    ("env_3.yaml", 15.0),
    ("env_4.yaml", 10.0),
    ("env_4.yaml", 12.0),
    ("env_4.yaml", 15.0),
]

# Move to script directory
abspath = os.path.abspath(__file__)
dir_name = os.path.dirname(abspath)
os.chdir(dir_name)

# Initialize settings
settings = Settings(create_sim_folder=False)

# Initialize results table
# Results table columns:
#   - env number (int)
#   - tether length (float)
#
#   - num of triangles in simplicial complex (nodes in dual lifted graph) (int)
#   - area of simplicial complex (sum of areas of triangles) (float)
#
#   - num of triangles in simplicial complex with extra triangles (int)
#   - area of simplicial complex (sum of areas of triangles) (float)
#
#   - num of nodes in homotopy augmented graph with res 1 (int)
#   - approximate area of homotopy augmented graph (number of nodes * res^2) (float)
#
#   - num of nodes in homotopy augmented graph with res 0.5 (int)
#   - approximate area of homotopy augmented graph (number of nodes * res^2) (float)
#
#   - num of nodes in homotopy augmented graph with res 0.1 (int)
#   - approximate area of homotopy augmented graph (number of nodes * res^2) (float)
results_table = np.zeros((len(eval_cases), 12), dtype=float)

# Loop over evaluation cases
for idx, case in enumerate(eval_cases):
    print(
        f"\n{CmdColors.OKBLUE}[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Running "
        f"evaluation {idx+1} (env {case[0][4]} l={case[1]}){CmdColors.ENDC}"
    )

    # Unpack case
    env_name, l = case
    results_table[idx, 0] = int(env_name[4])  # env name
    results_table[idx, 1] = l  # tether length

    # Generate env for the test case
    settings.env_name = env_name  # change environment
    env = env_2d.Env2D(settings)

    ####################################################################################
    # Compute lifted simplicial complex with conservativeness reduction
    print(
        f"{CmdColors.WARNING}[Triang]{CmdColors.ENDC} Running triangulation "
        "with conservativeness reduction."
    )
    # Create lifted simplicial complex
    triang = Triangulation(env)
    triang.INFO = True  # Enable info
    triang.DEBUG = False
    triang.triangulate()
    triang.max_lifted_triangles = 100_000
    triang.set_max_dist(l)
    triang.lift_triangulation(reduce_conservativeness=True)

    # Compute area of simplicial complex (sum of areas of triangles)
    area_sc: float = 0
    for tri in triang.triangles_lift:
        i1, i2, i3 = tri  # triangle indexes
        [x1, y1] = triang.vertices[triang.vertices_lift[i1][0]]
        [x2, y2] = triang.vertices[triang.vertices_lift[i2][0]]
        [x3, y3] = triang.vertices[triang.vertices_lift[i3][0]]
        area = 0.5 * abs(x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))
        area_sc += area

    # Compute extra area added by the extra triangles for conservativeness reduction
    area_extra: float = 0
    n_extra: int = len(triang.extra_simplices)
    for tri in triang.extra_simplices:
        [x1, y1] = tri[0][0]
        [x2, y2] = tri[0][1]
        [x3, y3] = tri[0][2]
        area = 0.5 * abs(x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))
        area_extra += area

    # Store results in table
    results_table[idx, 2] = len(triang.triangles_lift)  # nr of triangs
    results_table[idx, 3] = area_sc  # area simplicial complex
    results_table[idx, 4] = n_extra  # extra nr of triangs
    results_table[idx, 5] = area_extra  # extra area simplicial complex

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
    graph_1.build_homotopy_augmented_graph()

    # Store results in table
    results_table[idx, 6] = len(graph_1.vertices_lift)  # nodes h augmented graph
    results_table[idx, 7] = len(graph_1.vertices_lift) * 1.0 * 1.0  # approx area

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
    graph_2.build_homotopy_augmented_graph()

    # Store results in table
    results_table[idx, 8] = len(graph_2.vertices_lift)  # nodes h augmented graph
    results_table[idx, 9] = len(graph_2.vertices_lift) * 0.5 * 0.5  # approx area

    ####################################################################################
    # Homotopy augmented graph with fine resolution
    # print(
    #     f"{CmdColors.WARNING}[GridGraph]{CmdColors.ENDC} Running graph with "
    #     "resolution 0.1."
    # )

    # # Create graph
    # graph_3 = GridGraph(env)
    # graph_3.INFO = True  # Enable info
    # graph_3.DEBUG = False  # Disable verbose debug prints
    # graph_3.set_max_dist(l)
    # graph_3.n_max = 1_000_000
    # graph_3.set_grid_resolution(0.1, 0.1)
    # graph_3.build_homotopy_augmented_graph()

    # # Store results in table
    # results_table[idx, 10] = len(graph_3.vertices_lift)  # nodes h augmented graph
    # results_table[idx, 11] = len(graph_3.vertices_lift) * 0.1 * 0.1  # approx area

    ####################################################################################
    # Save iteration rdesults in pickle file for later analysis
    # A separate pkl file is saved for each evaluation case, containing all the data
    # structures (simplicial complexes and graphs) created during that iteration.
    filename = f"results/conservativeness_reduction/comparison-{idx+1}.pkl"
    data = {
        "settings": settings,
        "env": env,
        "triangulation": triang,
        "graph_coarse": graph_1,
        "graph_medium": graph_2,
        # "graph_fine": graph_3,
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
    with open(
        "results/conservativeness_reduction/comparison_results.csv",
        "w",
        encoding="utf-8",
    ) as f:
        # Print header
        print(
            "env\tl\t|R|\tA_R\t|R'|\tA_R'\t|H_1|\tA_H_1\t|H_2|\tA_H_2\t|H_3|\tA_H_3",
            end="\n",
        )
        for i, row in enumerate(results_table):
            # index
            print(f"{str(int(row[0]))}", end="")
            f.write(f"{str(int(row[0]))},")

            # tether length (l)
            print(f"\t{row[1]:.1f}", end="")
            f.write(f"{row[1]:.1f},")

            # num triangles and area of simplicial complex
            print(f"\t{int(row[2])}", end="")
            f.write(f"{int(row[2])},")
            print(f"\t{row[3]:.2f}", end="")
            f.write(f"{row[3]:.2f},")

            # number of triangles and approximate area of the extra triangles added
            # during conservativeness reduction
            print(f"\t{int(row[4])}", end="")
            f.write(f"{int(row[4])},")
            print(f"\t{row[5]:.2f}", end="")
            f.write(f"{row[5]:.2f},")

            # number of nodes and approximate area in homotopy augmented graph 1
            print(f"\t{int(row[6])}", end="")
            f.write(f"{int(row[6])},")
            print(f"\t{row[7]:.2f}", end="")
            f.write(f"{row[7]:.2f}")

            # number of nodes and approximate area in homotopy augmented graph 2
            print(f"\t{int(row[8])}", end="")
            f.write(f"{int(row[8])},")
            print(f"\t{row[9]:.2f}", end="")
            f.write(f"{row[9]:.2f},")

            # number of nodes and approximate area in homotopy augmented graph 3
            print(f"\t{int(row[10])}", end="")
            f.write(f"{int(row[10])},")
            print(f"\t{row[11]:.2f}")
            f.write(f"{row[11]:.2f}\n")
