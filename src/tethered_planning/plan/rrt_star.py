from __future__ import annotations

from typing import TYPE_CHECKING

import networkx as nx
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

        # Initialize graph with also the cost information stored in each node
        self.graph: nx.Graph = nx.Graph()
        self.graph.add_node(
            0, pos=self.env.robot_initial_pos, cost=0, parent=0, children=[]
        )

    @override
    def new_node(self) -> np.ndarray | None:
        """
        Generates a new node and connects it to the RRT_star tree.

        Returns:
            steered_point (np.ndarray | None): the position of the new node if a node
            was added, None otherwise
        """

        # TODO: restructure function to increase efficiency and clarity + verify that
        # the implementation is bug-free

        # Sample new point in the free configuration space
        point: np.ndarray = self.sample()

        # Steer the node to ensure that the new edge is not longer than max_edge_len
        closest_point_idx: int = self.closest_neighbor(point)
        closest_point: np.ndarray = self.graph.nodes[closest_point_idx]["pos"]
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
        cost_list: list[float] = []
        for neighbor_idx in neighbors_idx_list:
            cost = self.graph.nodes[neighbor_idx]["cost"] + np.linalg.norm(
                steered_point - np.array(self.graph.nodes[neighbor_idx]["pos"])
            )
            cost_list.append(cost)

        # cost_list: list[float] = [self.graph.nodes[idx]['cost']
        #                           + np.linalg.norm(steered_point_np
        #                           - np.array(self.graph.nodes[idx]['pos']))
        #                           for idx in neighbors_idx_list]

        # Retireve the index of the lowest cost node and use it to select the target
        # node to connect
        target_point_idx: int = neighbors_idx_list[np.argmin(cost_list)]
        target_point: np.ndarray = self.graph.nodes[target_point_idx]["pos"]

        # Connect new node
        if self.env.is_valid_edge(steered_point, target_point):
            self.n_nodes += 1
            new_node_idx = self.n_nodes
            new_node_cost: float = np.min(cost_list)
            self.graph.add_node(
                new_node_idx,
                pos=steered_point,
                cost=new_node_cost,
                parent=target_point_idx,
                children=[],
            )
            # TODO: by using directed edges it should be possible to remove necessity
            # of storing parent and children
            self.graph.add_edge(self.n_nodes, target_point_idx)
            self.graph.nodes[target_point_idx]["children"].append(new_node_idx)

            # Rewire
            for idx in neighbors_idx_list:
                new_cost = new_node_cost + np.linalg.norm(
                    steered_point - np.array(self.graph.nodes[idx]["pos"])
                )
                old_cost = self.graph.nodes[idx]["cost"]
                if new_cost < old_cost:
                    # Update the tree structure
                    parent_idx = self.graph.nodes[idx]["parent"]
                    self.graph.nodes[parent_idx]["children"].remove(idx)
                    self.graph.nodes[new_node_idx]["children"].append(idx)
                    self.graph.nodes[idx]["parent"] = new_node_idx
                    self.graph.remove_edge(idx, parent_idx)
                    self.graph.add_edge(new_node_idx, idx)

                    # Update the cost of the node and of its children
                    delta_cost = old_cost - new_cost
                    self.update_cost(idx, delta_cost)

            # Return the newly added point
            return steered_point
        else:
            # If no new node was added, return None
            return None

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
        pos_mat = np.array(list(nx.get_node_attributes(self.graph, "pos").values()))
        dist_vec = np.linalg.norm(pos_mat - point, axis=1)

        # Find all nodes within the radius
        dist_less_than_radius = dist_vec <= radius

        # FIXME: check if the following line is correct
        idx_list = np.where(dist_less_than_radius)[0].tolist()
        # idx_list = [i for i, x in enumerate(dist_less_than_radius) if x]
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
        self.graph.nodes[idx]["cost"] -= delta_cost

        # Update the cost of all the children nodes
        for child_idx in self.graph.nodes[idx]["children"]:
            self.update_cost(child_idx, delta_cost)

    @override
    def plan(self) -> list[nx.Graph]:
        """
        Run the RRT* algorithm to incrementally build a RRT* tree in the environment
        free configuration space, to find an obstacle-free path to the goal region.

        Returns:
            list[nx.Graph]: a list of Graph objects. If self.settings.anim.animate is
            True, the list will contain the state of the RRT* graph at each step.
            Otherwise, it will contain only the last Graph object.
        """
        # Initialize data structures
        graph_list: list[nx.Graph] = []

        # Execute the RRT* main routine
        for i in range(self.hard_nodes_limit):
            new_node = self.new_node()
            if self.settings.anim.animate:
                graph_list.append(self.graph.copy())
            if self.goal_reached_termination_condition and new_node is not None:
                x = new_node[0]
                y = new_node[1]
                if self.env.is_goal_reached(x, y):
                    print(
                        f"{CmdColors.OKBLUE}[RRT_star]{CmdColors.ENDC} the goal has "
                        "been reached by RRT_star."
                    )
                    break
            if self.max_nodes_n_termination_condition and i >= self.max_nodes_n:
                print(
                    f"{CmdColors.OKBLUE}[RRT_star]{CmdColors.ENDC} the maximum number "
                    "of nodes has been reached by RRT_star. The planning will be "
                    "terminated."
                )
                break

        # Return output data
        if not graph_list:
            graph_list.append(self.graph.copy())
        return graph_list
