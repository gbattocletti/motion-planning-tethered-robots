"""
Graph search algorithms for path planning.
"""

import numpy as np


def a_star_search(
    nodes: np.ndarray | list[int, list[int]],
    edges: list[list[int]],
    start_idx: int,
    goal_idx: int,
    nodes_2d: np.ndarray | None = None,
    **kwargs,
) -> list[int]:
    """
    Perform A* search on the graph. Also suitable for Dijkstra's algorithm (UCS) if
    use_heuristic is set to False.

    Args:
        nodes (np.ndarray | list[int, list[int]]): List of graph nodes. If np.ndarray,
            should be of shape (N, d) where N is the number of nodes and d is the
            dimension. In case of homotopy-augmented graphs, nodes can be represented
            as a list of [node_index, homotopy_signature], where node_index points to
            an element in nodes_2d. The homotopy signature is a list of signed integers.
        edges (list[list[int]]): Adjacency list representing graph edges.
        start_idx (int | list[int, list[int]]): Index of the start node (from nodes).
        goal_idx (int | list[int, list[int]]): Index of the goal node (from nodes).
            In case of homotopy-augmented graphs, the kwarg ignore_goal_homotopy can
            be set to True to make so that goal_idx is an index of nodes_2d instead.
            This results in all nodes in nodes that project to nodes_2d[goal_idx] being
            considered goal nodes.
        nodes_2d (np.ndarray | None, optional): 2D coordinates of the nodes. Required
            for homotopy-augmented graphs. Defaults to None.

    Kwargs:
        use_heuristic (bool): Whether to use a heuristic for A* search. Default is True.
            If false, the algorithm is equivalent to Dijkstra's algorithm, and is
            sometimes referred to as Uniform Cost Search (UCS).
        ignore_goal_homotopy (bool): If True, and if nodes is a homotopy-augmented
            graph, goal_idx points to nodes_2d instead of nodes. This means that all
            nodes in nodes that project to nodes_2d[goal_idx] are considered goal nodes.
            Default is False.
    Returns:
        list[int]: Optimal path as a list of indexes referring to the input vertices.

    Raises:
        ValueError: If nodes is a list but vertices_2d is not provided.
        ValueError: If nodes array is empty.
        ValueError: If no path is found from start to goal.
        TypeError: If input arguments are not of the expected type.
    """

    # TODO: add option to select goal among multiple goal nodes (list of indexes). This
    # should be managed properly both in case ignore_goal_homotopy is True or False.
    # Moreover, the use of an heuristic should be supported in both cases. A good
    # starting point for this could be to compute the cost-to-goal from each node to
    # each goal node, and take the minimum. This ensures the heuristic is admissible.

    # parse args
    n_nodes: int
    h_augmented: bool = False
    if isinstance(nodes, list):
        if not isinstance(nodes[0][0], int) or not isinstance(nodes[0][1], list):
            raise TypeError(
                "nodes does not match the expected type for homotopy-augmented graphs"
            )
        if nodes_2d is None:
            raise ValueError(
                "vertices_2d must be provided for homotopy-augmented graphs"
            )
        if not isinstance(nodes_2d, np.ndarray):
            raise TypeError("vertices_2d must be a numpy array")
        h_augmented = True
        n_nodes = len(nodes)
    elif isinstance(nodes, np.ndarray):
        n_nodes = nodes.shape[0]
    else:
        raise TypeError("nodes must be either a list or a numpy array")
    if n_nodes == 0:
        raise ValueError("nodes array is empty")

    # parse kwargs
    use_heuristic: bool = kwargs.get("use_heuristic", False)
    ignore_goal_homotopy: bool = kwargs.get("ignore_goal_homotopy", False)

    # validate start and goal indexes

    if not 0 <= start_idx < n_nodes:
        raise ValueError("start_idx is out of bounds for nodes")

    # NOTE: goal_idx_list contains all indexes in nodes that are considered goal nodes.
    # They refer to indexes in nodes, not nodes_2d.
    goal_idx_list: list[int]
    if h_augmented and ignore_goal_homotopy:
        if not 0 <= goal_idx < nodes_2d.shape[0]:
            raise ValueError(
                "goal_idx is out of bounds for vertices_2d in homotopy-augmented graph"
            )
        if use_heuristic is True:
            # TODO: add support for heuristic with multiple goal nodes. Simplest
            # solution it to compute the cost-to-goal from each node to each goal node,
            # and take the minimum.
            raise NotImplementedError(
                "Heuristic not implemented for homotopy-augmented graphs with "
                "ignore_goal_homotopy=True."
            )
        goal_idx_list = [idx for idx, node in enumerate(nodes) if node[0] == goal_idx]
    else:
        if not 0 <= goal_idx < n_nodes:
            raise ValueError("goal_idx is out of bounds for nodes")
        goal_idx_list = [goal_idx]

    # initialize algorithm data structures
    open_set: set[int] = set()  # nodes to be evaluated
    open_set.add(start_idx)
    closed_set: set[int] = set()  # nodes already evaluated
    parent: list[int | None] = [None] * n_nodes  # to reconstruct path (None for root)

    # cost functions
    g_score: np.ndarray = np.full(n_nodes, np.inf)  # cost from start to node
    g_score[start_idx] = 0.0
    h_score: np.ndarray = np.zeros(n_nodes)  # heuristic cost from node to goal
    if use_heuristic:
        if h_augmented:
            goal_point: np.ndarray = nodes_2d[nodes[goal_idx][0]]
            indexes = [n[0] for n in nodes]  # extract indexes of 2d nodes
            h_score = np.linalg.norm(nodes_2d[indexes] - goal_point, axis=1)
        else:
            goal_point: np.ndarray = nodes[goal_idx]
            h_score = np.linalg.norm(nodes - goal_point, axis=1)
    else:
        pass  # h_score is constant zero for all nodes

    # main loop
    while open_set:

        # select best node in open set
        # NOTE: current_idx point to an index in nodes (not nodes_2d)
        current_idx: int = min(open_set, key=lambda n: g_score[n] + h_score[n])
        open_set.remove(current_idx)  # move current node from open to closed set
        closed_set.add(current_idx)

        # termination condition
        if current_idx in goal_idx_list:
            path = []
            idx = current_idx
            while idx is not None:
                path.append(idx)
                idx = parent[idx]
            return path[::-1]  # reverse path

        # evaluate neighbors
        neighbors: list[int] = [
            idx2 if idx1 == current_idx else idx1
            for idx1, idx2 in edges
            if current_idx in (idx1, idx2)
        ]
        for neighbor_idx in neighbors:

            # check if neighbor has already been evaluated
            if neighbor_idx in closed_set:
                continue

            # add neighbor to open set if not already present
            if neighbor_idx not in open_set:
                open_set.add(neighbor_idx)

            # update g_score and parent if a path through current node is better
            edge_cost: float  # cost from current node to neighbor
            if h_augmented:
                edge_cost = np.linalg.norm(
                    nodes_2d[nodes[current_idx][0]] - nodes_2d[nodes[neighbor_idx][0]]
                )
            else:
                edge_cost = np.linalg.norm(nodes[current_idx] - nodes[neighbor_idx])
            tentative_g_score: float = g_score[current_idx] + edge_cost
            if tentative_g_score < g_score[neighbor_idx]:
                g_score[neighbor_idx] = tentative_g_score
                parent[neighbor_idx] = current_idx

    # if this point is reached, no path was found
    raise ValueError("No path found from start to goal")
