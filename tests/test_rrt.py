import os
import random
import unittest

import matplotlib.pyplot as plt
import numpy as np

from tethered_planning.env import env_2d
from tethered_planning.plan import rrt, rrt_star
from tethered_planning.utils import plot
from tethered_planning.utils.colors import CmdColors
from tethered_planning.utils.settings import Settings
from tethered_planning.utils.wrappers import measureStats

unittest.TestLoader.sortTestMethodsUsing = None  # run tests in order they are defined


class TestWorld(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        abspath = os.path.abspath(__file__)
        dir_name = os.path.dirname(abspath)
        os.chdir(dir_name)
        if not os.path.exists("results"):
            os.makedirs("results")
        # io.clean_folder("results")

        # Plot variables
        cls.blocking = True
        cls.wait_time = 1

    def setUp(self):
        # Initialize instance variables
        self.settings = None
        self.env = None
        self.planner = None

        # Load settings for planner
        self.settings = Settings()  # initialize settings
        self.settings.planner.max_nodes_n = 1_000
        if self.settings.fix_seed:
            random.seed(self.settings.seed)
            np.random.seed(self.settings.seed)
        self.env = env_2d.Env2D(self.settings)  # initialize 2d world

    def show_plot(self):
        if not self.blocking:
            plt.show(block=self.blocking)
            plt.pause(self.wait_time)
            plt.close()
        else:
            plt.show()  # wait on user to close plot and continue

    @measureStats
    def test_rrt(self):
        print(f"{CmdColors.OKBLUE}[TestRRT]{CmdColors.ENDC} Running test_rrt.")
        self.planner = rrt.RRT(self.env, self.settings)
        self.planner.plan()

    @measureStats
    def test_rrt_star(self):
        print(f"{CmdColors.OKBLUE}[TestRRT]{CmdColors.ENDC} Running test_rrt_star.")
        self.planner = rrt_star.RRTStar(self.env, self.settings)
        self.planner.plan()

    def test_plot_rrt_np(self):
        self.planner = rrt.RRT(self.env, self.settings)
        self.planner.plan()
        plot.plot_graph(
            self.planner.nodes[: self.planner.n_nodes + 1],
            self.planner.edges[: self.planner.n_nodes + 1],
            self.env,
            self.settings,
        )
        self.show_plot()


if __name__ == "__main__":
    unittest.main()
