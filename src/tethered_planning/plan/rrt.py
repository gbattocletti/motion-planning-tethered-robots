from __future__ import annotations

import random
import warnings
from typing import TYPE_CHECKING

import numpy as np

from ..utils.colors import CmdColors

if TYPE_CHECKING:
    from ..env.env_2d import Env2D
    from ..utils.settings import Settings


class RRT:
    """
    RRT class.
    """

    def __init__(self, env: Env2D, settings: Settings) -> None:
        """
        Initialize the RRT planner.

        Args:
            env (Env2D): The environment in which the planner operates.
            settings (Settings): The settings for the planner.

        Returns:
            None
        """
        self.env: Env2D = env
        self.settings: Settings = settings

        # Unpack relevant settings
        self.max_edge_length: float = self.settings.planner.max_edge_length
        self.goal_reached_termination_condition: bool = (
            self.settings.planner.goal_reached_termination_condition
        )
        self.max_nodes_termination_condition: bool = (
            self.settings.planner.max_nodes_n_termination_condition
        )
        self.max_nodes: int = self.settings.planner.max_nodes_n
        self.hard_nodes_limit = 100_000
        if self.max_nodes_termination_condition:
            if self.max_nodes > self.hard_nodes_limit:
                self.max_nodes = self.hard_nodes_limit
                warnings.warn(
                    f"{CmdColors.WARNING}[RRT]{CmdColors.ENDC} the RRT termination "
                    "numberof nodes is larger than the hard-coded max number of nodes "
                    f"({self.max_nodes} > {self.hard_nodes_limit}). See rrt.py to "
                    "change this value.",
                    Warning,
                )
            self.hard_nodes_limit = self.max_nodes  # update max number of nodes in RRT

        # Initialize data structures
        self.n_nodes: int = 0  # counter of nodes in the graph (excluding the root node)
        self.nodes: np.ndarray = np.empty((self.hard_nodes_limit + 1, 2), dtype=float)
        self.edges: np.ndarray = np.empty((self.hard_nodes_limit + 1, 2), dtype=int)
        self.parent: np.ndarray = np.empty((self.hard_nodes_limit + 1,), dtype=int)

        # Set initial condition (root node)
        self.nodes[self.n_nodes] = self.env.robot_initial_pos
        self.parent[self.n_nodes] = -1  # negative index indicates that node is root

    def sample(self) -> np.ndarray:
        """
        Sample a point in the free configuration space.

        Returns:
            np.ndarray: The sampled point.
        """
        # If goal biasing is used, sample from centroids of goal region polygons
        if (
            self.settings.planner.goal_bias
            and np.random.random() < self.settings.planner.goal_bias_rate
        ):
            # Get centroid of shapely goal polygon
            centroid = random.sample(self.env.goal_polygons, 1)[0].centroid
            point = np.array([centroid.x, centroid.y])

        # Sample random point from the free configuration space
        else:
            point = self.env.sample_free_space()

        # Return the sampled point
        return point

    def closest_neighbor(self, point: np.ndarray) -> int:
        """
        Finds the node in the RRT graph that is closest to a point.

        Args:
            point (np.ndarray): The point to find the closest neighbor of

        Returns:
            int: The index of the node in the RRT graph that is closest to point
        """
        # Compute the distances between the point and the existing nodes
        dist_vec = np.linalg.norm(self.nodes[: self.n_nodes + 1] - point, axis=1)

        # Return the index of the closest point
        return np.argmin(dist_vec)

    def linear_steer(self, point: np.ndarray, neighbor: np.ndarray) -> np.ndarray:
        """
        Steer the point towards the neighbor to respect the maximum edge length.

        Args:
            point (np.ndarray): the point to steer
            neighbor (np.ndarray): the neighboring point to steer towards

        Returns:
            np.ndarray: The steered point.
        """
        # Compute the distance between the point and the neighbor
        dist = np.linalg.norm(point - neighbor)

        # Determine if steering is needed
        if dist > self.max_edge_length:
            steered_point = (
                neighbor
                + (point - neighbor)
                / np.linalg.norm(point - neighbor)
                * self.max_edge_length
            )
        else:
            steered_point = point

        # Return the steered point
        return steered_point

    def new_node(self) -> np.ndarray | None:
        """
        Generates a new node and connects it to the RRT tree.

        Returns:
            np.ndarray | None: new node location if a node was added, None otherwise
        """
        # Sample new point in the free configuration space
        point: np.ndarray = self.sample()

        # Steer the node to ensure that the new edge is not longer than max_edge_length
        closest_point_idx: int = self.closest_neighbor(point)
        closest_point: np.ndarray = self.nodes[closest_point_idx]
        steered_point: np.ndarray = self.linear_steer(point, closest_point)

        # Add the node to the tree
        if self.env.is_valid_edge(steered_point, closest_point):
            self.n_nodes += 1
            self.nodes[self.n_nodes] = steered_point
            self.edges[self.n_nodes] = [self.n_nodes, closest_point_idx]
            self.parent[self.n_nodes] = closest_point_idx
            return steered_point
        else:
            return None

    def plan(self) -> list[dict]:
        """
        Run the RRT algorithm to incrementally build a RRT tree in the environment free
        configuration space, to find an obstacle-free path to the goal region.

        Returns:
            list[dict]: a list of dictionary objects. If self.settings.anim.animate is
            True, the list will contain the state of the RRT graph at each step.
            Otherwise, it will contain only the last dictionary.
        """
        # Initialize data structures
        graph_list: list[dict] = []

        # Execute the RRT main routine
        for i in range(self.hard_nodes_limit):
            new_node = self.new_node()
            if self.settings.anim.animate:
                graph_list.append(
                    {
                        "n_nodes": self.n_nodes,
                        "nodes": self.nodes.copy(),
                        "edges": self.edges.copy(),
                        "parent": self.parent.copy(),
                    }
                )
            if self.goal_reached_termination_condition and new_node is not None:
                x = new_node[0]
                y = new_node[1]
                if self.env.is_goal_reached(x, y):
                    print(
                        f"{CmdColors.OKBLUE}[RRT]{CmdColors.ENDC} the goal has been "
                        "reached by RRT."
                    )
                    break
            if self.max_nodes_termination_condition and i >= self.max_nodes:
                print(
                    f"{CmdColors.OKBLUE}[RRT]{CmdColors.ENDC} the maximum number of "
                    "nodes has been reached by RRT. The planning will be terminated."
                )
                break

        # Return output data
        if not graph_list:
            graph_list.append(
                {
                    "n_nodes": self.n_nodes,
                    "nodes": self.nodes.copy(),
                    "edges": self.edges.copy(),
                    "parent": self.parent.copy(),
                }
            )
        return graph_list
