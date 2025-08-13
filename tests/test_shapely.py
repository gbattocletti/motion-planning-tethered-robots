import os
import unittest

import matplotlib.pyplot as plt
import numpy as np
import shapely
from shapely.geometry import LineString, MultiPolygon, Point, Polygon
from shapely.plotting import plot_line, plot_points, plot_polygon

from tethered_planning.utils import io
from tethered_planning.utils.colors import CmdColors

# noinspection DuplicatedCode
unittest.TestLoader.sortTestMethodsUsing = None  # run tests in order they are defined


class TestShapely(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        abspath = os.path.abspath(__file__)
        dir_name = os.path.dirname(abspath)
        os.chdir(dir_name)
        if not os.path.exists("results"):
            os.makedirs("results")
        io.clean_folder("results")

        # Plot settings
        cls.show = False
        cls.blocking = False
        cls.wait_time = 1

    def test_contains_xy(self):
        print(
            f"{CmdColors.OKBLUE}[TestShapely]{CmdColors.ENDC} Running "
            "test_contains_xy."
        )
        polygon = Polygon(np.array([[0, 0], [0, 2], [2, 2], [2, 0], [1, 1]]))
        self.assertEqual(shapely.contains_xy(polygon, 0.5, 0.6), True)
        self.assertEqual(shapely.contains_xy(polygon, 3, 3), False)
        self.assertEqual(shapely.contains_xy(polygon, 0.1, 1), True)
        self.assertEqual(shapely.contains_xy(polygon, 1, 1), False)

    def test_plot_polygon(self):
        print(
            f"{CmdColors.OKBLUE}[TestShapely]{CmdColors.ENDC} Running "
            "test_plot_polygon."
        )
        polygon = Polygon(np.array([[0, 0], [0, 2], [2, 2], [2, 0], [1, 1]]))
        fig = plt.figure()
        ax = fig.add_subplot()
        plot_polygon(polygon, ax=ax, add_points=False)
        plot_points(polygon, ax=ax, alpha=0.7)
        if self.show:
            plt.show(block=self.blocking)
            plt.pause(self.wait_time)
            plt.close()

    def test_plot_polygon_and_line(self):
        print(
            f"{CmdColors.OKBLUE}[TestShapely]{CmdColors.ENDC} Running "
            "test_plot_polygon_and_line."
        )
        polygon = Polygon(np.array([[5.0, 2.0], [6.0, 2.0], [6.0, 8.0], [5.0, 8.0]]))
        line = LineString(np.array([[2, 2], [2, 6]]))
        fig = plt.figure()
        ax = fig.add_subplot()
        plot_polygon(polygon, ax=ax)
        plot_line(line, ax=ax)
        if self.show:
            plt.show(block=self.blocking)
            plt.pause(self.wait_time)
            plt.close()

    def test_lines_intersection(self):
        print(
            f"{CmdColors.OKBLUE}[TestShapely]{CmdColors.ENDC} Running "
            "test_lines_intersection."
        )
        self.check_lines_intersection(
            np.array([0, 0]),
            np.array([2, 2]),
            np.array([2, 0]),
            np.array([1, 2]),
            "intersect",
        )
        self.check_lines_intersection(
            np.array([0, 0]),
            np.array([2, 2]),
            np.array([0, 0]),
            np.array([0, 2]),
            "intersect",
        )
        self.check_lines_intersection(
            np.array([0, 0]),
            np.array([2, 2]),
            np.array([0, 0]),
            np.array([2, 2]),
            "coincide",
        )
        self.check_lines_intersection(
            np.array([0, 0]),
            np.array([2, 2]),
            np.array([1, 0]),
            np.array([3, 2]),
            "no_intersect",
        )
        self.check_lines_intersection(
            np.array([0, 0]),
            np.array([0, 2]),
            np.array([-1, 3]),
            np.array([1, 3]),
            "no_intersect",
        )

    def test_polygon_intersection(self):
        print(
            f"{CmdColors.OKBLUE}[TestShapely]{CmdColors.ENDC} Running "
            "test_polygon_intersection."
        )
        # Test 1
        p1 = Polygon(np.array([[0, 0], [0, 2], [2, 2], [2, 0]]))
        p2 = Polygon(np.array([[1, 0.5], [3, 0.5], [3, 1.5], [1, 1.5]]))
        self.check_polygon_intersection(p1, p2, True)

        # Test 2
        p1 = Polygon(np.array([[0, 0], [0, 2], [2, 2], [2, 0]]))
        p2 = Polygon(np.array([[1, 5], [5, 1], [6, 2], [2, 6]]))
        self.check_polygon_intersection(p1, p2, False)

        # Test 3
        p1 = Polygon(np.array([[0, 0], [0, 2], [2, 2], [2, 0]]))
        p2 = Polygon(np.array([[2, 2], [3, 2], [3, 3], [2, 3]]))
        self.check_polygon_intersection(p1, p2, True)

    def test_sample_check(self):
        print(
            f"{CmdColors.OKBLUE}[TestShapely]{CmdColors.ENDC} Running "
            "test_sample_check."
        )
        obs = self.multi_polygon()
        self.sample_check(obs, Point(1, 1), True)
        self.sample_check(obs, Point(0, 0), True)
        self.sample_check(obs, Point(2, 0), True)
        self.sample_check(obs, Point(1, 6), False)
        self.sample_check(obs, Point(5, 5), False)
        self.sample_check(obs, Point(6, 6.5), True)

    def test_sample_check_2(self):
        print(
            f"{CmdColors.OKBLUE}[TestShapely]{CmdColors.ENDC} Running "
            "test_sample_check_2."
        )
        obs = self.multi_polygon()
        self.sample_check_2(obs, 1, 1, True)
        self.sample_check_2(obs, 0, 0, True)
        self.sample_check_2(obs, 2, 0, True)
        self.sample_check_2(obs, 1, 6, False)
        self.sample_check_2(obs, 5, 5, False)
        self.sample_check_2(obs, 6, 6.5, True)

    ## AUXILIARY FUNCTIONS ####################################################

    def check_lines_intersection(self, a1, a2, b1, b2, expected_output):
        line1 = LineString([a1, a2])
        line2 = LineString([b1, b2])
        int_pt = line1.intersection(line2)
        match shapely.get_type_id(int_pt):
            case 0:  # lines intersect at 1 single point
                self.assertEqual("intersect", expected_output)
            case 1:  # lines have no intersection
                if int_pt.is_empty:
                    self.assertEqual("no_intersect", expected_output)
                else:  # lines coincide
                    self.assertEqual("coincide", expected_output)

    def check_polygon_intersection(self, p1, p2, expected_output):
        aa = shapely.intersection(p1, p2)
        if aa.is_empty:
            polygons_intersect = False
        else:
            polygons_intersect = True
        self.assertEqual(polygons_intersect, expected_output)

    def multi_polygon(self):
        p1 = Polygon(np.array([[0, 0], [5, 0], [5, 2], [3, 4], [0, 3]]))
        p2 = Polygon(np.array([[5.5, 6], [6.5, 6], [7, 7.5], [7, 8], [6, 7], [5, 7.5]]))
        p3 = Polygon(np.array([[1, 8], [2, 8], [2, 9], [1, 9]]))
        self.check_polygon_intersection(p1, p2, False)
        self.check_polygon_intersection(p1, p3, False)
        self.check_polygon_intersection(p2, p3, False)
        obs = MultiPolygon([p1, p2, p3])
        if self.show:
            plt.figure()
            plot_polygon(obs, add_points=True)
            plt.show(block=self.blocking)
            plt.pause(self.wait_time)
            plt.close()
        return obs

    def sample_check(
        self, obs: MultiPolygon | Polygon, point: Point, expected_output: bool
    ) -> None:
        """
        Check if a Point object is inside a Polygon object.

        Polygons are considered to be closed, i.e., a point on the boundary is
        considered inside the Polygon.

        Args:
            obs (MultiPolygon | Polygon): MultiPolygon or Polygon object
            point (Point): Point object
            expected_output (bool): expected function output. The test is passed if out
            matches this value

        Returns:
            None
        """
        out = obs.contains(point) or obs.boundary.contains(point)
        self.assertEqual(out, expected_output)

    def sample_check_2(
        self, obs: MultiPolygon | Polygon, x: float, y: float, expected_output: bool
    ) -> None:
        """
        Alternative sample check function (see also sample_check)

        Args:
            obs (MultiPolygon | Polygon): MultiPolygon or Polygon object
            x (float): x coordinate of the point to check
            y (float): y coordinate of the point to check
            expected_output (bool): expected function output. The test is passed if out
            matches this value

        Returns:
            None
        """
        out = shapely.contains_xy(obs, x, y) or shapely.contains_xy(obs.boundary, x, y)
        self.assertEqual(out, expected_output)


if __name__ == "__main__":
    unittest.main()
