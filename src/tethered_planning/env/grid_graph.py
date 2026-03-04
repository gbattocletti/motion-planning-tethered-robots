from __future__ import annotations

from collections.abc import Callable

import numpy as np

from tethered_planning.env.env_2d import Env2D
from tethered_planning.utils import curves, entanglement
from tethered_planning.utils.colors import CmdColors


class GridGraph:
    """
    Class to build a length-constrained homotopy-augmented grid graph on a 2D
    environment.
    """

    def __init__(self, env: Env2D) -> None:
        """
        Initialize the GridGraph class.

        Args:
            env (Env2D): The 2D environment to be triangulated.

        Returns:
            None

        Raises:
            TypeError: if data types are incorrect
        """
        # Env and parameters
        self.env: Env2D = env
        self.anchor_point: np.ndarray  # anchor point
        if env.anchor_point is not None:
            if not isinstance(env.anchor_point, np.ndarray):
                raise TypeError("Anchor point must be a numpy array.")
            self.anchor_point = env.anchor_point
        else:
            print(
                f"{CmdColors.WARNING}[GridGraph]{CmdColors.WARNING} Undefined anchor "
                "point in GridGraph."
            )

        # Max dist between anchor point and vertices (termination criterion)
        self.max_dist: float = 10.0

        # Max number of nodes (safety termination condition)
        self.n_max: int = 10_000

        # Grid resolution in x and y directions
        self.res_x: float = 1.0
        self.res_y: float = 1.0

        # Grid graph
        # NOTE: edges are stored as tuples (i, j) where i and j are the indices of
        # the vertices in self.vertices. The indexes are sorted such that i < j.
        self.points_x: np.ndarray  # x coordinates of grid points
        self.points_y: np.ndarray  # y coordinates of grid points
        self.vertices: np.ndarray  # (x, y) coordinates of vertices
        self.edges: list[tuple[int, int]]  # [[v_idx_1, v_idx_2], ...]
        self.init_grid()  # initialize grid (repeated if res is changed)

        # Homotopy-augmented grid graph
        self.root_idx: int  # index of anchor point in self.vertices
        self.vertices_lift: list[tuple[int, list[int]]]  # lifted vertices [idx, sign]
        self.edges_lift: list[list[int]]  # [[v1_lifted_idx, v2_lifted_idx], ...]

        # Entanglement verification
        # True if vertex is entanglement-admissible, False otherwise. Indexes match
        # those of vertices_lift.
        self.entanglement_function: Callable | None = None
        self.entanglement_vertices_lift: list[bool]

        # Debug info
        self.INFO: bool = False
        self.DEBUG: bool = False

    def set_max_dist(self, max_dist: float) -> None:
        """
        Setter for max distance between anchor point and vertices.

        Args:
            max_dist (float): maximum distance between anchor point and vertices
        """
        self.max_dist = max_dist

    def set_grid_resolution(self, res_x: float, res_y: float) -> None:
        """
        Setter for grid resolution.

        Args:
            res_x (float): grid resolution in x direction
            res_y (float): grid resolution in y direction

        Raises:
            TypeError: if res_x or res_y are not int or float values
            ValueError: if res_x or res_y are not positive values
        """
        if not isinstance(res_x, (int, float)) or not isinstance(res_y, (int, float)):
            raise TypeError("Grid resolutions must be numeric values.")
        if res_x <= 0 or res_y <= 0:
            raise ValueError("Grid resolutions must be positive values.")
        self.res_x = res_x
        self.res_y = res_y
        self.init_grid()

    def set_entanglement_definition(self, entanglement_definition: str) -> None:
        """
        Setter for entanglement definition.

        Args:
            entanglement_definition (str): the entanglement definition to use for
                checking the entanglement of the vertices in the homotopy-augmented
                graph. Must be one of "convex_hull", "linear_homotopy", or
                "local_visibility_homotopy".
        Raises:
            ValueError: if the input entanglement definition is not recognized.
        """
        if not isinstance(entanglement_definition, str):
            raise TypeError("entanglement_definition must be a string.")
        if entanglement_definition == "convex_hull":
            self.entanglement_function = entanglement.convex_hull
        elif entanglement_definition == "linear_homotopy":
            self.entanglement_function = entanglement.linear_homotopy
        elif entanglement_definition == "local_visibility_homotopy":
            self.entanglement_function = entanglement.local_visibility_homotopy
        else:
            raise ValueError(
                f"Unknown entanglement definition: {entanglement_definition}"
            )

    def init_grid(self) -> None:
        """
        Initialize the 2D grid points.

        Returns:
            None
        """
        # Initialize lattice of points
        # NOTE: The modulo term is added to ensure that the anchor point is a grid point
        self.points_x = (
            np.arange(0, self.env.size[0] + self.res_x, self.res_x)
            + self.anchor_point[0] % self.res_x
        )
        self.points_y = (
            np.arange(0, self.env.size[1] + self.res_y, self.res_y)
            + self.anchor_point[1] % self.res_y
        )
        self.vertices = np.meshgrid(self.points_x, self.points_y)
        self.vertices = np.vstack(
            (self.vertices[0].ravel(), self.vertices[1].ravel())
        ).T

        # Remove points inside obstacles
        invalid_idx = [
            not self.env.is_valid_point(*p, invalid_boundary=False)
            for p in self.vertices
        ]
        self.vertices = np.delete(self.vertices, invalid_idx, axis=0)

        # Find anchor point index
        dists = np.linalg.norm(self.vertices - self.anchor_point, axis=1)
        self.root_idx = np.argmin(dists)

        # Iterate over points to find edges
        # NOTE: edges are undirected, so (i,j) is the same as (j,i). For convenience,
        # they are stored as (min(i,j), max(i,j)).
        self.edges = []

        # NOTE: during the edge creation, only neighbors in the +x and +y directions
        # are checked. This is more efficient than checking all 4 neighbors, and
        # checking -x and -y is redundant since the edge expansion starts from [0, 0]
        # toward [max_x, max_y], and the edges are undirected.
        neighbors_to_check = [np.array([0, self.res_y]), np.array([self.res_x, 0])]

        for i, v1 in enumerate(self.vertices):
            for neighbor_offset in neighbors_to_check:
                v2 = v1 + neighbor_offset
                j = np.where((np.isclose(self.vertices, v2)).all(axis=1))[0]  # find idx
                if j.size == 0:
                    continue  # vertex does not exist (invalid point)
                elif j.size == 1:
                    i, j = sorted((i, int(j[0])))  # sort edge indices (smaller first)
                    if (i, j) not in self.edges:
                        self.edges.append((i, j))
                    else:
                        print("Duplicate edges exist!?")
                else:
                    raise ValueError("Duplicate vertex detected")

    def build_homotopy_augmented_graph(
        self,
        allow_boundary_overlap: bool = True,
        check_entanglement: bool = False,
    ) -> None:
        """
        Build the length-constrained homotopy-augmented grid graph.

        Args:
            allow_boundary_overlap (bool, optional): if True, allows points of the graph
                to lie on the boundary of obstacles. If False, points on the boundary
                are considered invalid. (Default: True)
            check_entanglement (bool, optional): whether to check the entanglement of
                each triangle before adding it to the lifted triangulation. Default is
                False.

        Returns:
            None
        """
        # Select entanglement function based on definition
        if check_entanglement and self.entanglement_function is None:
            raise ValueError(
                "Error: entanglement function not set. Please set the entanglement "
                "definition using set_entanglement_definition() method before building "
                "the homotopy-augmented graph with entanglement checking enabled."
            )

        # Initialize lifted graph
        self.vertices_lift = []
        self.edges_lift = []
        self.entanglement_vertices_lift = []

        # Initialize counter
        n: int = 0  # current number of nodes

        # Initialize queues
        open_queue: list[tuple] = []  # list of points to visit
        closed_queue: list[tuple] = []  # list of points already visited

        # Initial condition
        open_queue.append((self.root_idx, [], [], 0.0))

        # Specify neighbors (4-connectivity)
        # NOTE: due to the use of 4-connectivity, the length of the offsets can be
        # computed using a 1-norm. In case of 8-connectivity, 2-norm should be used.
        offsets = [
            np.array([0, self.res_y]),
            np.array([self.res_x, 0]),
            np.array([0, -self.res_y]),
            np.array([-self.res_x, 0]),
        ]

        # Temporary variables for main loop
        idx: int  # index of vertex in self.vertices
        sign: list[int]  # homotopy signature
        parent_vec: list[int]  # history vec of parent idx starting from anchor node
        a_len: float  # approximate length from anchor point to (x,y)

        # Main loop
        # NOTE: ideally the loop should terminate when open_queue is empty, meaning that
        # all the nodes within the max_dist have been explored. However, for safety an
        # upper limit on the number of nodes is set to avoid infinite loops. This value
        # should be sufficiently high to avoid premature termination.
        while len(open_queue) > 0 and n < self.n_max:

            idx, sign, parent_vec, a_len = open_queue.pop(0)
            idx = int(idx)

            # Check if vertex has already been visited
            if (idx, sign) in closed_queue:
                continue

            # Check if vertex is within max distance from anchor point
            # NOTE: the length from the anchor point to each vertex is approximated as
            # the manhattan distance (i.e., L1 norm) between the two points. When this
            # distance exceeds the max distance, a more correct (and more expensive)
            # check is performed by shortening the sequence of edges from the anchor
            # point to the vertex and checking the actual length of the resulting curve.
            # The sequence of edges is maintained in the parent_vec variable. Since
            # multiple paths can lead to the same vertex, the path with the shortest
            # manhattan distance is the one stored in memory.
            if a_len > self.max_dist:
                curve = np.array(self.vertices[parent_vec + [idx]])
                curve = curves.shorten_curve(curve, self.env)
                curve_len = curves.measure_length(curve)
                if curve_len > self.max_dist:
                    closed_queue.append((idx, sign))
                    continue
                a_len = curve_len

            if check_entanglement:
                curve = np.array(self.vertices[parent_vec + [idx]])
                curve = curves.shorten_curve(curve, self.env)
                if not self.entanglement_function(curve, self.env):
                    closed_queue.append((idx, sign))
                    self.entanglement_vertices_lift.append(
                        False
                    )  # entanglement-inadmissible
                    continue
                self.entanglement_vertices_lift.append(True)  # entanglement-admissible

            # Add lifted vertex
            self.vertices_lift.append((idx, sign))  # add new lifted vertex
            closed_queue.append((idx, sign))  # mark vertex as visited

            # Add neighbors to open queue
            for offset in offsets:
                neighbor: np.ndarray = self.vertices[idx] + offset
                neighbor_idx: int = np.where(
                    (np.isclose(self.vertices, neighbor)).all(axis=1)
                )
                if neighbor_idx[0].size == 0:
                    continue  # neighbor is not in self.vertices (invalid point)
                elif neighbor_idx[0].size == 1:
                    neighbor_idx = int(neighbor_idx[0][0])
                    edge: np.ndarray = np.array(
                        [self.vertices[idx], self.vertices[neighbor_idx]]
                    )
                    if not self.env.is_valid_edge(
                        *edge, allow_boundary_overlap=allow_boundary_overlap
                    ):
                        continue  # skip invalid edges (obstacle collision)
                    neighbor_sign = sign + curves.compute_signature(edge, self.env)
                    neighbor_sign = curves.simplify_signature(neighbor_sign)

                    # Check if neighbor has already been visited. If not, add it to the
                    # open queue. If yes, add an edge connecting to it.
                    if (neighbor_idx, neighbor_sign) in closed_queue:
                        try:
                            neighbor_lifted_idx = self.vertices_lift.index(
                                (neighbor_idx, neighbor_sign)
                            )  # Find the index of the neighbor in lifted vertices list
                            i, j = sorted((n, neighbor_lifted_idx))  # sort edge indices
                            self.edges_lift.append((i, j))
                        except ValueError:
                            # CHECKME: temporary solution
                            if self.DEBUG:
                                print(
                                    f"{CmdColors.WARNING}[GridGraph]{CmdColors.ENDC} "
                                    "Warning: could not find lifted neighbor vertex "
                                    f"({neighbor_idx}, {neighbor_sign}) in vertices "
                                    "list when adding edge."
                                )
                            pass
                    else:
                        if (n, neighbor_idx, neighbor_sign) not in open_queue:
                            open_queue.append(
                                (
                                    neighbor_idx,
                                    neighbor_sign,
                                    parent_vec + [idx],
                                    a_len + np.linalg.norm(offset, ord=1),
                                )
                            )
                else:
                    raise ValueError(f"Duplicate vertex detected for point {neighbor}")

            # Increment counter
            n += 1

        if self.INFO or self.DEBUG:
            if n >= self.n_max:
                print(
                    f"{CmdColors.WARNING}[GridGraph]{CmdColors.ENDC} Warning: "
                    "maximum number of nodes reached before exploring all reachable "
                    "vertices within max_dist. Consider increasing n_max or reducing "
                    "max_dist."
                )
            print(
                f"{CmdColors.OKBLUE}[GridGraph]{CmdColors.ENDC} Homotopy-augmented "
                f"graph built with {len(self.vertices_lift)} vertices and "
                f"{len(self.edges_lift)} edges."
            )
