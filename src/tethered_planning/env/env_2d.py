from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import shapely
from shapely.geometry import (
    GeometryCollection,
    LineString,
    MultiLineString,
    MultiPolygon,
    Point,
    Polygon,
)

from ..utils import io
from ..utils.colors import CmdColors
from ..utils.settings import Settings


class Env2D:

    def __init__(self, settings: Settings) -> None:
        """
        Creates an instance of the Env2D class and sets up the env parameters based on
        the valued contained in the settings object.

        Args:
            settings (Settings): Settings object containing the simulation parameters

        Returns:
            None

        Raises:
            ValueError: If the initial robot location specified in the settings
            object is invalid
        """
        # Initialize env properties
        self.env_file_path: str = None
        self.name: str = None
        self.dimension: int = None
        self.origin: np.ndarray = None
        self.size: np.ndarray = None
        self.workspace: Polygon = None
        self.obstacle_vertices: np.ndarray = None
        self.obstacle_polygons: list[Polygon] = None
        self.obstacle_region: MultiPolygon = None
        self.free_workspace: Polygon = None
        self.generators_vertices: np.ndarray = None
        self.generators_list: list[LineString] = None
        self.generators: MultiLineString = None
        self.goal_vertices: np.ndarray = None
        self.goal_polygons: list[Polygon] = None
        self.goal_region: MultiPolygon = None
        self.robot_initial_pos: np.ndarray = None
        self.robot_radius: float = None
        self.anchor_point: np.ndarray = None
        self.tether_length: float = None
        self.tether_state: np.ndarray = None
        self.tether_configuration: LineString = None

        # Load the environment data from the YAML file
        self.load_env(settings.env_name)

    def load_env(self, env_name: str = "env_default.yaml") -> None:
        """
        Load the environment data from a YAML file.

        Args:
            env_name (str): name of environment file to load (Default: env_default.yaml)

        Returns:
            None

        Raises:
            ValueError: If the dimension of the environment is not 2
        """
        # Load environment data from YAML file
        if not env_name.endswith(".yaml"):
            env_name += ".yaml"
        if env_name == "env_default.yaml":
            env_default_path = os.path.join(Path(__file__).parent, "env_default.yaml")
            env_data = io.load_yaml(env_default_path)  # load default from env folder
        else:
            self.env_file_path = f"./data/{env_name}"
            env_data = io.load_yaml(self.env_file_path)  # load from file in cwd folder

        # Load the environment properties
        self.name = env_data["env"]["name"]
        self.dimension = env_data["env"]["dimension"]
        if self.dimension != 2:
            print(
                f"{CmdColors.FAIL}[Env2D]{CmdColors.ENDC} the environment dimension "
                "is not supported by this environment class."
            )
            raise ValueError
        self.origin = np.array(env_data["env"]["origin"])
        self.size = np.array(env_data["env"]["size"])

        # Create env bounding box (assumes rectangular environment)
        self.workspace = Polygon(
            [
                self.origin,
                self.origin + np.array([0, self.size[1]]),
                self.origin + self.size,
                self.origin + np.array([self.size[0], 0]),
            ]
        )

        # Create the obstacle region
        self.obstacle_vertices = np.array(env_data["obstacles"]["vertices"])
        self.obstacle_polygons, self.obstacle_region = self.generate_obstacles()

        # Create free workspace polygon
        self.free_workspace = shapely.difference(self.workspace, self.obstacle_region)
        if not isinstance(self.free_workspace, Polygon):
            print(
                f"{CmdColors.WARNING}[Env]{CmdColors.WARNING} The environment free "
                f"workspace is not a Polygon but a {type(self.free_workspace)}. The "
                "free workspace may not be a single connected component."
            )
        if not self.free_workspace.is_simple:
            print(
                f"{CmdColors.WARNING}[Env]{CmdColors.WARNING} The environment free "
                "workspace is not simple and contains point-like self intersections."
            )

        # Create the generators
        self.generators_vertices = np.array(env_data["generators"]["vertices"])
        self.generators_list, self.generators = self.generate_generators()

        # Create the goal region
        self.goal_vertices = np.array(env_data["goal"]["vertices"])
        self.goal_polygons, self.goal_region = self.generate_goal()

        # Set the initial robot location
        self.robot_initial_pos = np.array(env_data["robot"]["initial_state"])
        self.robot_radius = env_data["robot"]["radius"]
        self.anchor_point = np.array(env_data["robot"]["anchor_point"])
        self.check_robot_initial_state()

        # Set the tether properties
        self.tether_length = env_data["tether"]["max_length"]
        self.tether_state = np.array(env_data["tether"]["initial_state"])
        self.tether_configuration = self.generate_tether_configuration()

        # Print success message
        print(
            f"{CmdColors.OKBLUE}[Env2D]{CmdColors.ENDC} environment {env_name} loaded "
            "successfully."
        )

    def generate_obstacles(
        self,
    ) -> tuple[list, GeometryCollection | Polygon | MultiPolygon]:
        """
        Generate a set of polygonal obstacles and groups them in a MultiPolygon object.

        Returns:
            tuple[list, GeometryCollection | Polygon | MultiPolygon]: A tuple containing
                a list of shapely objects with the components of the obstacle region,
                and a shapely MultiPolygon object containing set of polygonal obstacles.
        """
        obs_list: list = []
        for _, obs_vert in enumerate(self.obstacle_vertices):
            o = Polygon(obs_vert)  # create new obstacle from tuple of points
            obs_list.append(o)

        # return MultiPolygon object
        return obs_list, shapely.unary_union(obs_list)

    def generate_generators(self) -> tuple[list[LineString], MultiLineString]:
        """
        Generate a set of LineString objects representing the generators of the
        environment.

        The generators are defined as line segments connecting two points.
        Each endpoint must lie either on the boundary of two different obstacles, or on
        the boundary of the environment. Moreover, the generators must not intersect
        with each other, nor with any obstacle.

        Returns:
             tuple[list[LineString], MultiLineString]: a tuple containing:
                - a list of LineString objects representing a set of valid generators
                - a MultiLineString object corresponding to the union of all generators
        """
        gen_list: list[LineString] = []
        for i, gen_vert in enumerate(self.generators_vertices):
            g = LineString(gen_vert)  # create new generator from list of points

            # Check for intersections with the obstacle region
            if self.obstacle_region.intersects(g):
                intersection = self.obstacle_region.intersection(g)

                # Check if the intersection happpens only at the endpoints of the
                # generator. If so, ignore the intersection -- the generator is valid.
                if not g.boundary.contains(intersection):
                    print(
                        f"{CmdColors.WARNING}[Env2D]{CmdColors.ENDC} generator {i} "
                        f"intersects with an obstacle in {intersection}."
                    )
            gen_list.append(g)

        # Check for intersections between the generators
        for i, gen1 in enumerate(gen_list):
            for j, gen2 in enumerate(gen_list):
                if i != j and gen1.intersects(gen2):
                    print(
                        f"{CmdColors.WARNING}[Env2D]{CmdColors.WARNING} generator {i} "
                        f"intersects with generator {j}."
                    )

        # Return list of LineString objects
        return gen_list, shapely.unary_union(gen_list)

    def generate_goal(self) -> tuple[list, GeometryCollection | Polygon | MultiPolygon]:
        """
        Generate set of polygonal goals and group them in a Shapely MultiPolygon object.

        Returns:
            tuple[list, GeometryCollection | Polygon | MultiPolygon]: a tuple containing
                a list of shapely objects corresponding to the components of the goal
                region (normally 1 object, multiple ones in case the goal region is
                split by obstacles), and a shapely object corresponding the goal region

        Raises:
            ValueError: if the goal region fully overlaps with an obstacle
            ValueError: if the goal region is of an invalid geometry type
        """
        goal_list = []
        for _, goal_vert in enumerate(self.goal_vertices):
            g = Polygon(goal_vert)  # create new goal polygon from points

            # Check if the goal_region region intersects with the obstacle region
            if self.obstacle_region.intersects(g):
                non_overlapping_region = g.difference(self.obstacle_region)
                if non_overlapping_region.is_empty:
                    print(
                        f"{CmdColors.FAIL}[Env2D]{CmdColors.ENDC} the goal region "
                        "does not contain any valid point. The goal region has been "
                        "removed."
                    )
                    raise ValueError
                elif non_overlapping_region.geom_type == "Polygon":
                    goal_list.append(non_overlapping_region)
                    resized_goal = True
                elif non_overlapping_region.geom_type == "MultiPolygon":
                    for g_subregion in non_overlapping_region.geoms:
                        goal_list.append(g_subregion)
                    resized_goal = True
                else:
                    print(
                        f"{CmdColors.FAIL}[Env2D]{CmdColors.ENDC} the goal region "
                        "has an invalid geometry."
                    )
                    raise ValueError
                if resized_goal:
                    print(
                        f"{CmdColors.WARNING}[Env2D]{CmdColors.ENDC} the goal region "
                        f"defined by vertices {', '.join(map(str, goal_vert))} has a "
                        "non-empty intersection with the obstacle region. The goal "
                        "region has been resized."
                    )
            else:
                goal_list.append(g)

        # Merge all obstacle_region in a single MultiPolygon object and return it
        return goal_list, shapely.unary_union(goal_list)

    def generate_tether_configuration(self) -> LineString:
        """
        Checks that the tether configuration is valid and generates the LineString
        tether configuration object.

        Returns:
            LineString: LineString object representing the tether configuration

        Raises:
            ValueError: if the tether configuration intersects with an obstacle
            ValueError: if the tether configuration exceeds the max tether length
            ValueError: if the tether configuration does not start in the anchor point
            ValueError: if the tether configuration does not end in the robot position
        """
        # create LineString object from the input list of vertices
        init_point = self.tether_state[0]
        end_point = self.tether_state[-1]
        config = LineString(self.tether_state)

        # check for intersections with obstacles
        if self.obstacle_region.intersects(config):
            print(
                f"{CmdColors.FAIL}[Env2D]{CmdColors.ENDC} the initial tether "
                "configuration intersects with an obstacle."
            )
            raise ValueError

        # check that the tether does not exceed its maximum length
        if config.length > self.tether_length:
            print(
                f"{CmdColors.FAIL}[Env2D]{CmdColors.ENDC} the initial tether "
                "configuration exceeds the maximum length."
            )
            raise ValueError

        # check that the tether start point coincides with the anchor point
        if not (
            init_point[0] == self.anchor_point[0]
            and init_point[1] == self.anchor_point[1]
        ):
            print(
                f"{CmdColors.FAIL}[Env2D]{CmdColors.ENDC} the initial tether "
                "configuration does not start from the anchor point."
            )
            raise ValueError

        # check that the tether end point coincides with the robot initial position
        if not (
            end_point[0] == self.robot_initial_pos[0]
            and end_point[1] == self.robot_initial_pos[1]
        ):
            print(
                f"{CmdColors.FAIL}[Env2D]{CmdColors.ENDC} the initial tether "
                "configuration does not end at the robot initial position."
            )
            raise ValueError

        # Return the generated LineString
        return config

    def check_robot_initial_state(self) -> None:
        """
        Checks if the initial state of the robot is valid, i.e., if the robot does not
        overlap with obstacles (including its radius) and if the anchor point is valid.

        Returns:
            None

        Raises:
            ValueError: if the robot initial location is invalid
            ValueError: if the robot (including the radius) overlaps with an obstacle
            ValueError: if the anchor point is invalid
        """
        # check if the robot initial position is inside an obstacle
        if not self.is_valid_point(
            self.robot_initial_pos[0], self.robot_initial_pos[1]
        ):
            print(
                f"{CmdColors.FAIL}[Env2D]{CmdColors.ENDC} the initial robot location "
                "specified in the environment file is invalid."
            )
            raise ValueError

        # check if the robot radius intersects with an obstacle
        robot_point = Point(self.robot_initial_pos[0], self.robot_initial_pos[1])
        robot_occupation = robot_point.buffer(self.robot_radius)
        if robot_occupation.intersects(self.obstacle_region):
            print(
                f"{CmdColors.FAIL}[Env2D]{CmdColors.ENDC} the robot initial location "
                "has an occupation that intersects with an obstacle."
            )
            raise ValueError

        # check if the anchor point is inside an obstacle
        if not self.is_valid_point(self.anchor_point[0], self.anchor_point[1]):
            print(
                f"{CmdColors.FAIL}[Env2D]{CmdColors.ENDC} the anchor point specified "
                "in the environment file is invalid."
            )
            raise ValueError

    def is_valid_point(self, x: float, y: float) -> bool:
        """
        Check if a point is inside or on the boundary of the obstacle region.

        Args:
            x (float): x-coordinate of the point to check
            y (float): y-coordinate of the point to check

        Returns:
            bool: True if the point is outside of the obstacle region, False otherwise
        """
        # Perform check and return result
        return not (
            shapely.contains_xy(self.obstacle_region, x, y)
            or shapely.contains_xy(self.obstacle_region.boundary, x, y)
        )

    def is_valid_edge(self, point1: np.ndarray, point2: np.ndarray) -> bool:
        """
        Check if an edge intersects with the obstacle region.

        Args:
            point1 (np.ndarray): (2, ) ndarray with the coordinates of the first
            endpoint of the edge to check
            point2 (np.ndarray): (2, ) ndarray with the coordinates of the second
            endpoint of the edge to check

        Returns:
            bool: True if edge does not intersect with obstacle region, False otherwise
        """
        return not self.obstacle_region.intersects(LineString([point1, point2]))

    def is_goal_reached(self, point: np.ndarray) -> bool:
        """
        Check if a point is inside (in the `interior`) the goal_region region.

        Args:
            point (np.ndarray): (2, ) ndarray with the coordinates of the point to check

        Returns:
            bool: True if the point is in the goal_region region, False otherwise
        """
        return shapely.contains_xy(self.goal_region, point[0], point[1])

    def sample_free_space(self) -> np.ndarray:
        """
        Returns a point sampled from the environment free configuration space.

        Returns:
            np.ndarray: (2, ) ndarray with the coordinates of the sampled point

        Raises:
            StopIteration: if  maximum number of  sampling attempts is reached
        """
        # Function settings
        max_attempts = 20

        # Sample points until a valid one is found or max attempts is reached
        for _ in range(0, max_attempts):
            x = np.random.uniform(0, self.size[0])
            y = np.random.uniform(0, self.size[1])
            if self.is_valid_point(x, y):
                return np.array([x, y])

        # If no valid point has been found raise an error
        print(
            f"{CmdColors.FAIL}[Env2D]{CmdColors.ENDC} failed to sample point (max "
            "number of iterations reached)."
        )
        raise StopIteration
