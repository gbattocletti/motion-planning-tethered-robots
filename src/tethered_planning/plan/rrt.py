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
        self.graph: nx.Graph = nx.Graph()  # create empty networkx graph
        self.graph.add_node(0, pos=self.env.robot_initial_pos)  # add root node

        # Unpack relevant settings
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
        closest_point: np.ndarray = self.graph.nodes[closest_point_idx]["pos"]
        steered_point: np.ndarray = self.linear_steer(point, closest_point)
        if self.env.is_valid_edge(steered_point, closest_point):
            self.n_nodes += 1
            self.graph.add_node(self.n_nodes, pos=steered_point)
            self.graph.add_edge(self.n_nodes, closest_point_idx)
            return steered_point
        else:
            return None

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
            centroid = random.sample(self.env.goal_polygons, 1)[0].centroid
            point = np.array([centroid.x, centroid.y])
            # TODO: check type of centroid, maybe this is not needed

        # Sample random point from the free configuration space
        else:
            point = self.env.sample_free_space()

        # Return the sampled point
        return point

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

    def closest_neighbor(self, point: np.ndarray) -> int:
        """
        Finds the node in the RRT graph that is closest to a point.

        Args:
            point (np.ndarray): The point to find the closest neighbor of

        Returns:
            int: The index of the node in the RRT graph that is closest to point
        """
        # Get the matrix of the positions of all the points in the graph
        # TODO: can this operation be made more efficient?
        pos_mat = np.array(list(nx.get_node_attributes(self.graph, "pos").values()))

        # Compute the distances from the point
        dist_vec = np.linalg.norm(pos_mat - point, axis=1)

        # Return the index of the closest point
        return np.argmin(dist_vec)

    def plan(self) -> list[nx.Graph]:
        """
        Run the RRT algorithm to incrementally build a RRT tree in the environment free
        configuration space, to find an obstacle-free path to the goal region.

        Returns:
            list[nx.Graph]: a list of Graph objects. If self.settings.anim.animate is
            True, the list will contain the state of the RRT graph at each step.
            Otherwise, it will contain only the last Graph object.
        """
        # Initialize data structures
        graph_list: list[nx.Graph] = []

        # Execute the RRT main routine
        for i in range(self.hard_nodes_limit):
            new_node = self.new_node()
            if self.settings.anim.animate:
                graph_list.append(self.graph.copy())
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
        if not graph_list:
            graph_list.append(self.graph.copy())
        return graph_list
