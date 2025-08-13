import os
import unittest

import matplotlib.pyplot as plt

from tethered_planning.env.env_2d import Env2D
from tethered_planning.plan import rrt
from tethered_planning.utils import io, plot
from tethered_planning.utils.colors import CmdColors
from tethered_planning.utils.settings import Settings

unittest.TestLoader.sortTestMethodsUsing = None  # run tests in order they are defined


class TestPlot(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        abspath = os.path.abspath(__file__)
        dir_name = os.path.dirname(abspath)
        os.chdir(dir_name)
        if not os.path.exists("results"):
            os.makedirs("results")
        io.clean_folder("results")

        # plot settings
        cls.show = False
        cls.blocking = False
        cls.wait_time = 1

    def setUp(self):
        self.settings = Settings("test_settings")
        self.env_name = "test_env_4"
        self.env = Env2D(self.settings)

    def show_plot(self):
        if self.show:
            if not self.blocking:
                plt.show(block=self.blocking)
                plt.pause(self.wait_time)
                plt.close()
            else:
                plt.show()  # wait on user to close plot and continue

    def test_plot_env(self):
        print(f"{CmdColors.OKBLUE}[TestPlot]{CmdColors.ENDC} Running test_plot_env.")
        plot.plot_env(self.env, self.settings)
        self.show_plot()

    def test_plot_env_with_kwargs_1(self):
        print(
            f"{CmdColors.OKBLUE}[TestPlot]{CmdColors.ENDC} Running "
            "test_plot_env_with_kwargs_1."
        )
        plot.plot_env(
            self.env,
            self.settings,
            show_goal=False,
            show_anchor=False,
            label_generators=False,
            dummy_kwarg=True,
            show_legend=True,
        )
        self.show_plot()

    def test_plot_env_with_kwargs_2(self):
        print(
            f"{CmdColors.OKBLUE}[TestPlot]{CmdColors.ENDC} Running "
            "test_plot_env_with_kwargs_2."
        )
        plot.plot_env(
            self.env,
            self.settings,
            show_goal=True,
            show_anchor=True,
            label_generators=True,
            show_legend=True,
        )
        self.show_plot()

    def test_plot_tether(self):
        print(f"{CmdColors.OKBLUE}[TestPlot]{CmdColors.ENDC} Running test_plot_tether.")
        self.settings.plot.show_legend = True
        self.settings.plot.title = "Tether Plot"
        plot.plot_tether(self.env, self.settings, tether=self.env.tether_configuration)
        self.show_plot()

    def test_plot_graph(self):
        print(f"{CmdColors.OKBLUE}[TestPlot]{CmdColors.ENDC} Running test_plot_graph.")
        self.settings.anim.animate = True  # save graph history
        self.settings.plot.show_legend = True
        planner = rrt.RRT(self.env, self.settings)
        graph_list = planner.plan()  # plan with rrt
        graph = graph_list[-1]  # get final graph
        plot.plot_graph(self.env, graph, self.settings, show_legend=True)
        self.show_plot()


if __name__ == "__main__":
    unittest.main()
