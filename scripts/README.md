# Scripts

This folder contains a collection of python scripts that can be used to run simulations and experiments with the motion planning algorithms implemented in the repository.

## Available scripts
The following scripts are currently available:
- `graph_comparison.py` Computes both an h-augmented graph and a simplicial complex model of the configuration space of a tethered robot, and compares the resulting data structures and computation times.
- `path_planning.py` Example of path planning algorithm for tethered mobile robots. Computes all the paths between the robot and the goal point that can be traveled while respecting the tether length constraint. 
- `plot_env.py` Plots an environment, its 2D Delaunay triangulation, and the corresponding simplicial complex model of the configuration space of a tethered robot.

The scripts have been used to generate the results presented in the paper 'Efficient Computation of a Continuous Topological Model of the Configuration Space of Tethered Mobile Robots' (2025).