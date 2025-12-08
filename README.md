# Motion Planning Algorithms for Mobile Tethered Robots
Motion planning algorithms for mobile tethered robots.

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
![Python 3.11](https://img.shields.io/badge/python->=3.11-green.svg)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

# Publications
For the code used to generate the results in the paper 'Efficient Computation of a Continuous Topological Model of the Configuration Space of Tethered Mobile Robots' (2025), see https://github.com/gbattocletti/motion-planning-tethered-robots/releases/tag/paper-2025. The scripts are available in the [`/scripts/` folder](/scripts/).

# Installation
The repository is structured as a python package.
To install, clone the repository by running:
```sh
git clone https://github.com/gbattocletti/motion-planning-tethered-robots.git
cd motion-planning-tethered-robots
pip install .
```

In case you want to install also the dependencies used in the development phase you can run instead:
```sh
pip install -e .[dev]
```

## Project Structure
The repository structure is detailed below:
```
root/  
│
├── scripts/                        # folder with simulation files and experiments
│   ├── README.md                   # info on the experiments scripts
│   ├── graph_comparison.py         # computation and comparison of h-augmented graph vs simplicial complex model 
│   ├── path_planning.py            # path planning experiment with enumeration of homotopically distinct paths
│   ├── plot_env.py                 # plots environment, triangulation, and simplicial complex for a given env file
│   ├── data/
│   │   └── .yaml                   # environment files for experiments
│   └── results/
│       └── .pkl, .gif, .png...     # experiments output data (untracked)
│
├── src                             # main package (importable with -e)
│   └── tethered_robot_planning/    # python package name (use for import)
│       ├── __init__.py
│       ├── env/
│       │   ├── env_2d.py           # environment class and methods 
│       │   ├── env_default.yaml    # default environment settings file
│       │   ├── grid_graph.py       # h-augmented graph of environment
│       │   └── triangulation.yaml  # triangulation and simplicial complex of polygonal environment
│       ├── plan/
│       │   ├── graph_search.py     # search-based motion planning (A*, BFS, DFS)
│       │   ├── rrt_star.py         # RRT* motion planning
│       │   └── rrt.py              # RRT motion planning
│       └── utils/
│           └── ...                 # helper functions and plot functions
│
├── tests/
│   └── ...                         # tests to verify the working of the modules
│
├── pyproject.toml                  
├── README.md
├── LICENSE
├── .gitignore                      
└──  .pylintrc                      
```

---

## License
The repository is provided under the GNU GPLv3 License. See the LICENSE file included with this repository.


## Author
[Gianpietro Battocletti](https://www.tudelft.nl/staff/g.battocletti/), PhD Candidate at the [Delft Center for Systems and Control](https://www.tudelft.nl/en/me/about/departments/delft-center-for-systems-and-control/), [Delft University of Technology](https://www.tudelft.nl/en/).<br>
Contact information: [g.battocletti@tudelft.nl]().<br>
Copyright (c) 2025 Gianpietro Battocletti.

