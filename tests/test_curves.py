import os
import unittest

import matplotlib.pyplot as plt

from tethered_planning.env.env_2d import Env2D
from tethered_planning.utils import curves, io, plot
from tethered_planning.utils.colors import CmdColors
from tethered_planning.utils.settings import Settings

unittest.TestLoader.sortTestMethodsUsing = None  # run tests in order they are defined


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
        print(
            f"{CmdColors.OKBLUE}[TestCurveFcns]{CmdColors.ENDC} Running "
            "test_curve_generation."
        )
        c = curves.generate_curve(self.env, self.settings)
        print(type(c))

    def test_curve_generation_output(self):
        # test default curve generation function (no kwargs passed)
        print(
            f"{CmdColors.OKBLUE}[TestCurveFcns]{CmdColors.ENDC} Running "
            "test_curve_generation_output."
        )
        c = curves.generate_curve(self.env, self.settings, output_type="array")
        print(type(c))

    def test_curve_generation_no_collision_check(self):
        # test curve generation without collision check
        print(
            f"{CmdColors.OKBLUE}[TestCurveFcns]{CmdColors.ENDC} Running "
            "test_curve_generation_no_collision_check."
        )
        curves.generate_curve(self.env, self.settings, check_obs=False)

    def test_curve_generation_with_self_intersection_check(self):
        # test curve generation with self-intersection check
        print(
            f"{CmdColors.OKBLUE}[TestCurveFcns]{CmdColors.ENDC} Running "
            "test_curve_generation_with_self_intersection_check."
        )
        curves.generate_curve(self.env, self.settings, check_self_intersection=True)

    def test_curve_generation_from_robot(self):
        # test curve generation starting from robot position
        print(
            f"{CmdColors.OKBLUE}[TestCurveFcns]{CmdColors.ENDC} Running "
            "test_curve_generation_from_robot."
        )
        curves.generate_curve(
            self.env,
            self.settings,
            init_curve=self.env.tether_configuration,
            check_self_intersection=True,
        )

    def test_multiple_curve_generation(self):
        # test the generation of two curves in the same environment
        print(
            f"{CmdColors.OKBLUE}[TestCurveFcns]{CmdColors.ENDC} Running "
            "test_multiple_curve_generation."
        )
        curve_1 = curves.generate_curve(self.env, self.settings)
        curve_2 = curves.generate_curve(self.env, self.settings)
        plot.plot_curves(
            self.env,
            self.settings,
            curves=[curve_1, curve_2],
            show_anchor=False,
            label_generators=False,
        )
        plt.show()

    def test_multiple_curve_generation_with_display(self):
        # test the generation of multiple curves in the same environment with the
        # previously generated curves being displayed in the environment and the
        # collision check with other curves set to active
        print(
            f"{CmdColors.OKBLUE}[TestCurveFcns]{CmdColors.ENDC} Running "
            "test_multiple_curve_generation_with_display."
        )
        n = 3
        curves_list = []
        for idx in range(n):
            curve = curves.generate_curve(
                self.env,
                self.settings,
                other_curves=curves_list,
                check_other_curves=True,
                title=f"Generating curve ({idx+1}/{n}). ESC to terminate.",
            )
            curves_list.append(curve)
        plot.plot_curves(
            self.env,
            self.settings,
            curves=curves_list,
            show_anchor=False,
            label_generators=False,
        )
        plt.show()

    def test_signature(self):
        # test curve signature computation
        print(
            f"{CmdColors.OKBLUE}[TestCurveFcns]{CmdColors.ENDC} Running "
            "test_signature."
        )
        curve = curves.generate_curve(self.env, self.settings)
        # NOTE: the case in which a point lies on a generator cannot be tested via
        # manual generation of the curve and requires an ad-hoc definition of the curve.
        # curve = LineString([(10.0, 8.0), (3.20, 7.80), (5.00, 9.40)])
        signature = curves.compute_signature(curve, self.env)
        print(f"Signature: {signature}")
        plot.plot_tether(
            self.env,
            self.settings,
            tether=curve,
            label_generators=True,
            show_robot=False,
        )
        plt.show()

    def test_shorten_curve(self):
        print(
            f"{CmdColors.OKBLUE}[TestCurveFcns]{CmdColors.ENDC} Running "
            "test_shorten_curve."
        )
        curve = curves.generate_curve(self.env, self.settings)
        # The curve can also be manually defined for testing purposes
        # from shapely.geometry import LineString
        # curve = LineString([(10.0, 8.0), (3.20, 7.80), (5.00, 9.40)])
        shortened_curve = curves.shorten_curve(curve, self.env)
        plot.plot_tether(self.env, self.settings, tether=curve, show_robot=False)
        plot.plot_tether(
            self.env, self.settings, tether=shortened_curve, show_robot=False
        )
        plt.show()

    def test_shorten_curve_multi_iteration(self):
        print(
            f"{CmdColors.OKBLUE}[TestCurveFcns]{CmdColors.ENDC} Running "
            "test_shorten_curve."
        )
        curve = curves.generate_curve(self.env, self.settings)
        shortened_curve_1 = curves.shorten_curve(curve, self.env)
        shortened_curve_2 = curves.shorten_curve(curve, self.env, iterations=5)
        plot.plot_tether(self.env, self.settings, tether=curve)
        plot.plot_tether(self.env, self.settings, tether=shortened_curve_1)
        plot.plot_tether(self.env, self.settings, tether=shortened_curve_2)
        plt.show()

    def test_resample_curve_linear(self):
        print(
            f"{CmdColors.OKBLUE}[TestCurveFcns]{CmdColors.ENDC} Running "
            "test_resample_curve_linear."
        )
        curve = curves.generate_curve(self.env, self.settings)
        resampled_curve = curves.resample_curve(curve, 10, "linear")
        plot.plot_curves(self.env, self.settings, curves=[curve], show_points=True)
        plot.plot_curves(
            self.env, self.settings, curves=[resampled_curve], show_points=True
        )
        plt.show()

    def test_resample_curve_total(self):
        print(
            f"{CmdColors.OKBLUE}[TestCurveFcns]{CmdColors.ENDC} Running "
            "test_resample_curve_total."
        )
        curve = curves.generate_curve(self.env, self.settings)
        resampled_curve = curves.resample_curve(curve, 10, "total")
        plot.plot_curves(self.env, self.settings, curves=[curve], show_points=True)
        plot.plot_curves(
            self.env, self.settings, curves=[resampled_curve], show_points=True
        )
        plt.show()


if __name__ == "__main__":
    unittest.main()
