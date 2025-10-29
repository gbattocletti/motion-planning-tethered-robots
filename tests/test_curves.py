# pylint: disable=logging-fstring-interpolation

import logging
import os
import unittest

import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import LineString

from tethered_planning.env.env_2d import Env2D
from tethered_planning.utils import curves, io, plot
from tethered_planning.utils.colors import CmdColors
from tethered_planning.utils.settings import Settings

unittest.TestLoader.sortTestMethodsUsing = None  # run tests in order they are defined


logger = logging.getLogger(__name__)


class TestCurveFcns(unittest.TestCase):

    def setUp(self):
        # test setup
        abspath = os.path.abspath(__file__)
        dir_name = os.path.dirname(abspath)
        os.chdir(dir_name)
        io.clean_folder("results")

        # load settings and create environment
        self.settings = Settings("test_settings")
        self.settings.env_name = "test_env_4"
        self.env = Env2D(self.settings)

    def test_curve_generation(self):
        # test default curve generation function (no kwargs passed)
        logging.info(
            f"{CmdColors.OKBLUE}[TestCurveFcns]{CmdColors.ENDC} Running "
            "test_curve_generation."
        )
        c = curves.generate_curve(self.env)
        logging.info(type(c))

    def test_curve_generation_output(self):
        # test default curve generation function (no kwargs passed)
        logging.info(
            f"{CmdColors.OKBLUE}[TestCurveFcns]{CmdColors.ENDC} Running "
            "test_curve_generation_output."
        )
        c = curves.generate_curve(self.env, output_type="array")
        logging.info(type(c))

    def test_curve_generation_no_collision_check(self):
        # test curve generation without collision check
        logging.info(
            f"{CmdColors.OKBLUE}[TestCurveFcns]{CmdColors.ENDC} Running "
            "test_curve_generation_no_collision_check."
        )
        curves.generate_curve(self.env, check_obs=False)

    def test_curve_generation_with_self_intersection_check(self):
        # test curve generation with self-intersection check
        logging.info(
            f"{CmdColors.OKBLUE}[TestCurveFcns]{CmdColors.ENDC} Running "
            "test_curve_generation_with_self_intersection_check."
        )
        curves.generate_curve(self.env, check_self_intersection=True)

    def test_curve_generation_from_robot(self):
        # test curve generation starting from robot position
        logging.info(
            f"{CmdColors.OKBLUE}[TestCurveFcns]{CmdColors.ENDC} Running "
            "test_curve_generation_from_robot."
        )
        curve = curves.generate_curve(
            self.env,
            init_curve=self.env.tether_configuration,
            check_self_intersection=True,
        )
        logging.info(type(curve))

    def test_multiple_curve_generation(self):
        # test the generation of two curves in the same environment
        logging.info(
            f"{CmdColors.OKBLUE}[TestCurveFcns]{CmdColors.ENDC} Running "
            "test_multiple_curve_generation."
        )
        curve_1 = curves.generate_curve(self.env)
        curve_2 = curves.generate_curve(self.env)
        plot.plot_env(
            self.env,
            curves=[curve_1, curve_2],
            show_anchor=False,
            show_generators_labels=False,
        )
        plt.show()

    def test_multiple_curve_generation_with_display(self):
        # test the generation of multiple curves in the same environment with the
        # previously generated curves being displayed in the environment and the
        # collision check with other curves set to active
        logging.info(
            f"{CmdColors.OKBLUE}[TestCurveFcns]{CmdColors.ENDC} Running "
            "test_multiple_curve_generation_with_display."
        )
        n = 3
        curves_list = []
        for idx in range(n):
            curve = curves.generate_curve(
                self.env,
                other_curves=curves_list,
                check_other_curves=True,
                title=f"Generating curve ({idx+1}/{n}). ESC to terminate.",
            )
            curves_list.append(curve)
        plot.plot_env(
            self.env,
            curves=curves_list,
            show_anchor=False,
            show_generators_labels=False,
        )
        plt.show()

    def test_signature(self):
        # test curve signature computation
        logging.info(
            f"{CmdColors.OKBLUE}[TestCurveFcns]{CmdColors.ENDC} Running "
            "test_signature."
        )
        curve = curves.generate_curve(self.env)
        # NOTE: the case in which a point lies on a generator cannot be tested via
        # manual generation of the curve and requires an ad-hoc definition of the curve.
        # curve = LineString([(10.0, 8.0), (3.20, 7.80), (5.00, 9.40)])
        signature = curves.compute_signature(curve, self.env)
        logging.info(f"Signature: {signature}")
        plot.plot_env(
            self.env,
            tether=curve,
            show_tether=True,
            show_generators_labels=True,
        )
        plt.show()
        pass

    def test_shorten_curve(self):
        logging.info(
            f"{CmdColors.OKBLUE}[TestCurveFcns]{CmdColors.ENDC} Running "
            "test_shorten_curve."
        )
        curve = curves.generate_curve(self.env)
        # The curve can also be manually defined for testing purposes
        # from shapely.geometry import LineString
        # curve = LineString([(10.0, 8.0), (3.20, 7.80), (5.00, 9.40)])
        shortened_curve = curves.shorten_curve(curve, self.env)
        plot.plot_env(self.env, tether=curve)
        plot.plot_env(self.env, tether=shortened_curve)
        plt.show()

    def test_shorten_curve_2(self):
        logging.info(
            f"{CmdColors.OKBLUE}[TestCurveFcns]{CmdColors.ENDC} Running "
            "test_shorten_curve."
        )
        self.settings.env_name = "test_env_1"  # Change env name from default case
        self.env = Env2D(self.settings)
        curve = np.array(
            [
                [1.0, 1.0],
                [1.0, 1.5],
                [1.0, 2.0],
                [1.0, 2.5],
                [1.0, 3.0],
                [1.0, 3.5],
                [1.0, 4.0],
                [1.0, 4.5],
                [1.0, 5.0],
                [1.0, 5.5],
                [1.0, 6.0],
                [1.5, 6.0],
                [2.0, 6.0],
                [2.5, 6.0],
                [3.0, 6.0],
                [3.5, 6.0],
                [4.0, 6.0],
                [4.5, 6.0],
                [5.0, 6.0],
                [5.5, 6.0],
                [6.0, 6.0],
                [6.0, 5.5],
                [6.0, 5.0],
                [6.0, 4.5],
                [6.0, 4.0],
                [6.0, 3.5],
                [6.0, 3.0],
                [6.0, 2.5],
                [6.0, 2.0],
                [6.0, 1.5],
                [6.0, 1.0],
                [6.0, 0.5],
                [6.0, 0.0],
                [5.5, 0.0],
                [5.0, 0.0],
                [4.5, 0.0],
                [4.0, 0.0],
                [3.5, 0.0],
                [3.0, 0.0],
                [2.5, 0.0],
                [2.0, 0.0],
                [1.5, 0.0],
            ]
        )
        plot.plot_env(self.env, tether=LineString(curve))
        plt.show()

        # test shortening when allowing boundary overlap in edge check
        shortened_curve = curves.shorten_curve(
            curve, self.env, allow_boundary_overlap=True
        )
        plot.plot_env(self.env, tether=LineString(shortened_curve))
        plt.show()

        # test shortening without allowing boundary overlap in edge check
        shortened_curve = curves.shorten_curve(
            curve, self.env, allow_boundary_overlap=False
        )
        plot.plot_env(self.env, tether=LineString(shortened_curve))
        plt.show()

    def test_shorten_curve_multi_iteration(self):
        logging.info(
            f"{CmdColors.OKBLUE}[TestCurveFcns]{CmdColors.ENDC} Running "
            "test_shorten_curve."
        )
        curve = curves.generate_curve(self.env)
        shortened_curve_1 = curves.shorten_curve(curve, self.env)
        shortened_curve_2 = curves.shorten_curve(curve, self.env, iterations=5)
        plot.plot_env(self.env, tether=curve)
        plot.plot_env(self.env, tether=shortened_curve_1)
        plot.plot_env(self.env, tether=shortened_curve_2)
        plt.show()

    def test_resample_curve_linear(self):
        logging.info(
            f"{CmdColors.OKBLUE}[TestCurveFcns]{CmdColors.ENDC} Running "
            "test_resample_curve_linear."
        )
        curve = curves.generate_curve(self.env)
        resampled_curve = curves.resample_curve(curve, 10, "linear")
        plot.plot_env(self.env, curves=[curve], show_curves_nodes=True)
        plot.plot_env(self.env, curves=[resampled_curve], show_curves_nodes=True)
        plt.show()

    def test_resample_curve_global(self):
        logging.info(
            f"{CmdColors.OKBLUE}[TestCurveFcns]{CmdColors.ENDC} Running "
            "test_resample_curve_global."
        )
        curve = curves.generate_curve(self.env)
        resampled_curve = curves.resample_curve(curve, 10, "global")
        plot.plot_env(self.env, curves=[curve], show_curves_nodes=True)
        plot.plot_env(self.env, curves=[resampled_curve], show_curves_nodes=True)
        plt.show()


if __name__ == "__main__":
    unittest.main()
