"""
Old implementations of RRT and RRT*.

The classes in this file are temporarily maintained for testing purposes.
"""

from __future__ import annotations

import random
import warnings
from typing import TYPE_CHECKING

import networkx as nx
import numpy as np

from ..utils.colors import CmdColors

if TYPE_CHECKING:
    from ..env.env_2d import Env2D
    from ..utils.settings import Settings


class RRT:
    """
    Tuple-based implementation of RRT.
    """

    def __init__(self, env: Env2D, settings: Settings):
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
        self.graph: nx.Graph = nx.Graph()  # create empty networkx graph
        self.graph.add_node(0, pos=self.env.robot_initial_pos)  # add root node

        # unpack relevant settings
        self.n_nodes: int = 0  # counter of nodes in the graph (excluding the root node)
        self.max_edge_length: float = self.settings.planner.max_edge_length
        self.goal_reached_termination_condition: bool = (
            self.settings.planner.goal_reached_termination_condition
        )
        self.max_nodes_n_termination_condition: bool = (
            self.settings.planner.max_nodes_n_termination_condition
        )
        self.max_nodes_n: int = self.settings.planner.max_nodes_n
        self.hard_nodes_limit = 10000
        if (
            self.max_nodes_n_termination_condition
            and self.max_nodes_n > self.hard_nodes_limit
        ):
            warnings.warn(
                f"{CmdColors.WARNING}[RRT]{CmdColors.ENDC} the RRT termination number "
                "of nodes is larger than the hard-coded max number of nodes "
                f"({self.max_nodes_n} > {self.hard_nodes_limit}). See rrt.py to "
                "change this value.",
                Warning,
            )

    def new_node(self) -> tuple | None:
        """
        Generates a new node and connects it to the RRT tree.

        Returns:
            tuple | None: new node location if a node was added, None otherwise
        """
        # Sample new point in the free configuration space
        point: tuple = self.sample()

        # Steer the node to ensure that the new edge is not longer than max_edge_length
        closest_point_idx: int = self.closest_neighbor(point)
        closest_point: tuple = self.graph.nodes[closest_point_idx]["pos"]
        steered_point: tuple = self.linear_steer(point, closest_point)
        if self.env.is_valid_edge(steered_point, closest_point):
            self.n_nodes += 1
            self.graph.add_node(self.n_nodes, pos=steered_point)
            self.graph.add_edge(self.n_nodes, closest_point_idx)
            return steered_point
        else:
            return None

    def sample(self) -> tuple:
        """
        Sample a point in the free configuration space.

        Returns:
            tuple: The sampled point.
        """
        point: tuple
        if (
            self.settings.planner.goal_bias
            and np.random.random() < self.settings.planner.goal_bias_rate
        ):
            # Use centroid coordinates from randomly selected shapely polygon object
            centroid = random.sample(self.env.goal_polygons, 1)[0].centroid
            point = (centroid.x, centroid.y)
        else:
            # Sample random point from the free configuration space
            point = self.env.sample_free_space()
        return point

    def linear_steer(self, point: tuple, neighbor: tuple) -> tuple:
        """
        Steer the point towards the neighbor to respect the maximum edge length.

        Args:
            point (tuple): the point to steer
            neighbor (tuple): the neighboring point to steer towards

        Returns:
            tuple: The steered point.
        """

        point = np.array(point)
        neighbor = np.array(neighbor)
        # TODO: fix data types so that this type transformation is not necessary anymore

        dist = np.linalg.norm(point - neighbor)
        if dist > self.max_edge_length:
            steered_point = (
                neighbor
                + (point - neighbor)
                / np.linalg.norm(point - neighbor)
                * self.max_edge_length
            )
        else:
            steered_point = point
        return tuple(steered_point)

    def closest_neighbor(self, point: tuple) -> int:
        """
        Finds the node in the RRT graph that is closest to a point.

        Args:
            point (tuple): The point to find the closest neighbor of

        Returns:
            int: The index of the node in the RRT graph that is closest to point
        """
        pos_mat = np.array(list(nx.get_node_attributes(self.graph, "pos").values()))
        dist_vec = np.linalg.norm(pos_mat - point, axis=1)
        closest_index = np.argmin(dist_vec)
        return closest_index

    def plan(self) -> None | list[nx.Graph]:
        """
        Run the RRT algorithm to incrementally build a RRT tree in the environment free
        configuration space, to find an obstacle-free path to the goal region.

        Returns:
            list[nx.Graph]: a list of Graph objects if self.settings.anim.animate is
            True, None otherwise.
        """
        # Initialize data structures (only used if animate = True)
        animation_steps: list[nx.Graph] = []

        # Execute the RRT main routine
        for i in range(self.hard_nodes_limit):
            new_node = self.new_node()
            if self.settings.anim.animate:
                animation_steps.append(self.graph.copy())
            if self.goal_reached_termination_condition and new_node is not None:
                x = new_node[0]
                y = new_node[1]
                if self.env.is_goal_reached(x, y):
                    print(
                        f"{CmdColors.OKBLUE}[RRT]{CmdColors.ENDC} the goal has been "
                        "reached by RRT."
                    )
                    break
            if self.max_nodes_n_termination_condition and i >= self.max_nodes_n:
                print(
                    f"{CmdColors.OKBLUE}[RRT]{CmdColors.ENDC} the maximum number of "
                    "nodes has been reached by RRT. The planning will be terminated."
                )
                break

        # Return output data
        if self.settings.anim.animate:
            return animation_steps
        else:
            return None
