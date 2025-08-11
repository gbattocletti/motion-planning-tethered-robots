# motion-planning-tethered-robots
Motion planning algorithms for mobile tethered robots. 

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
![Python 3.11](https://img.shields.io/badge/python->=3.11-green.svg)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

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
├── pyproject.toml                  # python project configuration file
├── README.md
├── LICENSE
├── .gitignore                      # git configuration
├── .pylintrc                       # custom settings for linter
├── .env                            # secret file with tokens and keys (untracked) 
│
├── scripts/                        # folder with simulation files and experiments
│   ├── data/
│   │   └── .yaml
│   ├── results/
│   │   └── .pkl, .gif, .png...     # saved simulation data (pkl files, gifs, images...)
│   ├── ...
|   └── ...
│
├── src                             # main package (importable with -e)
│   └── tethered_robot_planning/    # python package name (use for import)
│       ├── __init__.py
│       ├── control/
│       │   ├──
│       │   └──
│       ├── env/
│       │   ├──
│       │   └──
│       ├── plan/
│       │   ├──
│       │   └──
│       ├── world/
│       │   ├──
│       │   └──
│       └── utils/
│           ├──
│           └──
│
└── tests/
    └── ...                         # tests to verify the working of the modules
```

## License
The repository is provided under the GNU GPLv3 License. See the LICENSE file included with this repository.

---

## Author
[Gianpietro Battocletti](https://www.tudelft.nl/staff/g.battocletti/), PhD Candidate at the [Delft Center for Systems and Control](https://www.tudelft.nl/en/me/about/departments/delft-center-for-systems-and-control/), [Delft University of Technology](https://www.tudelft.nl/en/).

Contact information: [g.battocletti@tudelft.nl]()

Copyright (c) 2025 Gianpietro Battocletti.

---
