from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from typing_extensions import override

from ..plan.rrt import RRT
from ..utils.colors import CmdColors

if TYPE_CHECKING:
    from ..env.env_2d import Env2D
    from ..utils.settings import Settings


class RRTStar(RRT):
    """
    RRT* class.
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

        super().__init__(env, settings)

        # Create attributes for RRT_star parameters
        self.eta = self.settings.planner.max_edge_length  # eta radius for RRT_star

        # Initialize additional data structures
        self.cost: np.ndarray = np.empty((self.hard_nodes_limit + 1,), dtype=float)
        self.children: list[list[int]] = [[] for _ in range(self.hard_nodes_limit + 1)]

        # Set initial conditions
        self.cost[self.n_nodes] = 0.0
        self.children[self.n_nodes] = []

    def radius_neighbors(self, point: np.ndarray, radius: float) -> list[int]:
        """
        Finds the list of nodes in the RRT_star graph whose distance from a point is
        less than radius.

        Args:
            point (np.ndarray): The point to find the neighbors of
            radius (float): Radius around point in which to look for neighbors

        Returns:
            list[int]: The list of indexes of nodes in the RRT_star graph whose distance
            from 'point' is less than 'radius'.
        """
        # Compute distance from all nodes in the graph
        dist_vec = np.linalg.norm(self.nodes[: self.n_nodes + 1] - point, axis=1)

        # Find all nodes within the radius
        dist_less_than_radius = dist_vec <= radius
        idx_list = np.where(dist_less_than_radius)[0].tolist()  # CHECKME
        # idx_list = [i for i, x in enumerate(dist_less_than_radius) if x]  # TODO del
        return idx_list

    def update_cost(self, idx: int, delta_cost: float) -> None:
        """
        Recursively updates the cost of a subtree of the RRT* tree

        Args:
            idx (int): index of node from which the cascade cost update must be started
            delta_cost (float): difference in cost by which the node and its children
            are updated

        Returns:
            None
        """
        # Update the cost of the node
        self.cost[idx] -= delta_cost

        # Update the cost of all the children nodes
        for child_idx in self.children[idx]:
            self.update_cost(child_idx, delta_cost)

    @override
    def new_node(self) -> np.ndarray | None:
        """
        Generates a new node and connects it to the RRT_star tree.

        Returns:
            steered_point (np.ndarray | None): the position of the new node if a node
            was added, None otherwise
        """

        # Sample new point in the free configuration space
        point: np.ndarray = self.sample()

        # Steer the node to ensure that the new edge is not longer than max_edge_len
        closest_point_idx: int = self.closest_neighbor(point)
        closest_point: np.ndarray = self.nodes[closest_point_idx]
        steered_point: np.ndarray = self.linear_steer(point, closest_point)

        # TODO: add a True/False return to linear_steer that is True only when the node
        # was steered. If the node was steered, in fact, there will be no rewiring and
        # there is also no need to check the neighborhood -- this happens only if the
        # initial node has not been steered!).

        # Check cost of nodes in the neighborhood and select the one with lowest cost
        # NOTE: eta value is slightly inflated to avoid numerical precision issues
        neighbors_idx_list: list[int] = self.radius_neighbors(
            steered_point, 1.01 * self.eta
        )
        cost_list: np.ndarray[float] = self.cost[neighbors_idx_list] + np.linalg.norm(
            steered_point - self.nodes[neighbors_idx_list]
        )

        # Retireve the index of the lowest cost node and use it to find parent point
        parent_point_idx: int = neighbors_idx_list[np.argmin(cost_list)]
        parent_point: np.ndarray = self.nodes[parent_point_idx]

        # Add the node to the tree
        if self.env.is_valid_edge(steered_point, parent_point):
            self.n_nodes += 1
            new_node_idx = self.n_nodes  # only for conceptual clarity
            self.nodes[new_node_idx] = steered_point
            self.edges[new_node_idx] = [new_node_idx, parent_point_idx]
            self.parent[new_node_idx] = parent_point_idx
            self.children[parent_point_idx].append(new_node_idx)
            self.cost[new_node_idx] = np.min(cost_list)

            # Rewire
            for idx in neighbors_idx_list:
                new_cost = self.cost[new_node_idx] + np.linalg.norm(
                    self.nodes[new_node_idx] - self.nodes[idx]
                )
                old_cost = self.cost[idx]
                if new_cost < old_cost:
                    # Update the tree structure
                    parent_idx = self.parent[idx]
                    self.children[parent_idx].remove(idx)
                    self.children[new_node_idx].append(idx)
                    self.parent[idx] = new_node_idx

                    # Update edge
                    edge_idx = np.where(np.all(self.edges == [idx, parent_idx], axis=1))
                    self.edges[edge_idx] = [idx, new_node_idx]

                    # Update the cost of the node and of its children
                    delta_cost = old_cost - new_cost
                    self.update_cost(idx, delta_cost)

            # Return the newly added point
            return steered_point
        else:
            # If no new node was added, return None
            return None

    @override
    def plan(self) -> list[dict]:
        """
        Run the RRT* algorithm to incrementally build a RRT* tree in the environment
        free configuration space, to find an obstacle-free path to the goal region.

        Returns:
            list[dict]: a list of dictionary objects. If self.settings.anim.animate is
            True, the list will contain the state of the RRT* graph at each step.
            Otherwise, it will contain only the last dictionary.
        """
        # Initialize data structures
        graph_list: list[dict] = []

        # Execute the RRT* main routine
        for i in range(self.hard_nodes_limit):
            new_node = self.new_node()
            if self.settings.anim.animate:
                graph_list.append(
                    {
                        "n_nodes": self.n_nodes,
                        "nodes": self.nodes.copy(),
                        "edges": self.edges.copy(),
                        "parent": self.parent.copy(),
                        "cost": self.cost.copy(),
                    }
                )
            if self.goal_reached_termination_condition and new_node is not None:
                x = new_node[0]
                y = new_node[1]
                if self.env.is_goal_reached(x, y):
                    print(
                        f"{CmdColors.OKBLUE}[RRT_star]{CmdColors.ENDC} the goal has "
                        "been reached by RRT_star."
                    )
                    break
            if self.max_nodes_termination_condition and i >= self.max_nodes:
                print(
                    f"{CmdColors.OKBLUE}[RRT_star]{CmdColors.ENDC} the maximum number "
                    "of nodes has been reached by RRT_star. The planning will be "
                    "terminated."
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
                    "cost": self.cost.copy(),
                }
            )
        return graph_list
