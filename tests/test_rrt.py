import os
import random
import unittest

import matplotlib.pyplot as plt
import numpy as np

from tethered_planning.env import env_2d
from tethered_planning.plan import rrt, rrt_star
from tethered_planning.utils import io, plot
from tethered_planning.utils.colors import CmdColors
from tethered_planning.utils.settings import Settings

unittest.TestLoader.sortTestMethodsUsing = None  # run tests in order they are defined


class TestWorld(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        abspath = os.path.abspath(__file__)
        dir_name = os.path.dirname(abspath)
        os.chdir(dir_name)
        if not os.path.exists("results"):
            os.makedirs("results")
        io.clean_folder("results")

        # Plot variables
        cls.show = False
        cls.blocking = False
        cls.wait_time = 1

    def setUp(self):
        # Initialize instance variables
        self.settings = None
        self.world = None
        self.planner = None

        # Load settings for planner
        self.settings = Settings()  # initialize settings
        if self.settings.fix_seed:
            random.seed(self.settings.seed)
            np.random.seed(self.settings.seed)
        self.world = env_2d.Env2D(self.settings)  # initialize 2d world

    def show_plot(self):
        if self.show:
            if not self.blocking:
                plt.show(block=self.blocking)
                plt.pause(self.wait_time)
                plt.close()
            else:
                plt.show()  # wait on user to close plot and continue

    def test_rrt(self):
        print(f"{CmdColors.OKBLUE}[TestPlanners]{CmdColors.ENDC} Running test_rrt.")
        self.planner = rrt.RRT(self.world, self.settings)  # plan with rrt
        self.planner.plan()
        plot.plot_graph(self.world, self.planner.graph, self.settings)
        self.show_plot()

    def test_rrt_star(self):
        print(
            f"{CmdColors.OKBLUE}[TestPlanners]{CmdColors.ENDC} Running "
            "test_rrt_star."
        )
        self.planner = rrt_star.RRTStar(self.world, self.settings)  # plan with rrt star
        self.planner.plan()
        plot.plot_graph(self.world, self.planner.graph, self.settings)
        self.show_plot()


if __name__ == "__main__":
    unittest.main()
