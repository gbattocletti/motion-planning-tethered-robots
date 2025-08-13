import os
import random
import unittest

import numpy as np

from tethered_planning.env import env_2d
from tethered_planning.plan import rrt_star
from tethered_planning.utils import animate, io
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

    def setUp(self):
        # Initialize class settings
        self.settings = None
        self.world = None
        self.planner = None

        # Load settings
        self.settings = Settings()  # initialize settings
        if self.settings.fix_seed:
            random.seed(self.settings.seed)
            np.random.seed(self.settings.seed)
        self.world = env_2d.Env2D(self.settings)  # initialize 2d world

    def test_rrt_animation(self) -> None:
        self.planner = rrt_star.RRTStar(self.world, self.settings)  # plan with rrt star
        frames = self.planner.plan()
        anim = animate.animate(frames, self.world, self.settings)  # animate the path
        io.save_animation(anim, self.settings)  # save the animation


if __name__ == "__main__":
    unittest.main()
