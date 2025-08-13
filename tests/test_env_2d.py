import os
import unittest

import matplotlib.pyplot as plt
import numpy as np

from tethered_planning.env.env_2d import Env2D
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

        # Set blocking and wait time for plots
        cls.show = False
        cls.blocking = False
        cls.wait_time = 1

    def test_world_generation(self):
        print(
            f"{CmdColors.OKBLUE}[TestWorld]{CmdColors.ENDC} Running "
            "test_world_generation."
        )
        settings = Settings()  # default settings
        print(settings)
        world = Env2D(settings)
        if self.show:
            plot.plot_env(world, settings)
            plt.show(block=self.blocking)
            plt.pause(self.wait_time)
            plt.close()

    def test_goal_region_intersection_1(self):
        print(
            f"{CmdColors.OKBLUE}[TestWorld]{CmdColors.ENDC} Running "
            "test_goal_region_intersection_1."
        )
        settings = Settings("test_settings")
        settings.env_name = "test_env_1.yaml"
        print(settings)
        world = Env2D(settings)
        if self.show:
            plot.plot_env(world, settings)
            plt.show(block=self.blocking)
            plt.pause(self.wait_time)
            plt.close()

    def test_goal_region_intersection_2(self):
        print(
            f"{CmdColors.OKBLUE}[TestWorld]{CmdColors.ENDC} Running "
            "test_goal_region_intersection_2."
        )
        settings = Settings("test_settings")
        settings.env_name = "test_env_2.yaml"
        print(settings)
        world = Env2D(settings)
        if self.show:
            plot.plot_env(world, settings)
            plt.show(block=self.blocking)
            plt.pause(self.wait_time)
            plt.close()

    def test_goal_region_no_goal(self):
        print(
            f"{CmdColors.OKBLUE}[TestWorld]{CmdColors.ENDC} Running "
            "test_goal_region_no_goal."
        )
        settings = Settings("test_settings")
        settings.env_name = "test_env_3.yaml"
        print(settings)
        world = Env2D(settings)
        if self.show:
            plot.plot_env(world, settings)
            plt.show(block=self.blocking)
            plt.pause(self.wait_time)
            plt.close()

    def test_obstacle_intersection(self):
        print(
            f"{CmdColors.OKBLUE}[TestWorld]{CmdColors.ENDC} Running "
            "test_obstacle_intersection."
        )
        settings = Settings("test_settings")
        settings.env_name = "test_env_2.yaml"
        print(settings)
        world = Env2D(settings)
        self.assertEqual(world.is_valid_edge(np.array([2, 2]), np.array([2, 6])), True)
        self.assertEqual(world.is_valid_edge(np.array([2, 2]), np.array([6, 2])), False)
        self.assertEqual(world.is_valid_edge(np.array([2, 4]), np.array([4, 8])), True)
        self.assertEqual(world.is_valid_edge(np.array([7, 2]), np.array([8, 2])), True)
        self.assertEqual(world.is_valid_edge(np.array([2, 4]), np.array([8, 4])), False)
        self.assertEqual(world.is_valid_edge(np.array([0, 2]), np.array([2, 2])), True)


if __name__ == "__main__":
    unittest.main()
