from __future__ import annotations

import numpy as np
import shapely
from shapely import LineString, Point, Polygon

from tethered_planning.env.env_2d import Env2D
from tethered_planning.utils import curves
from tethered_planning.utils.colors import CmdColors


class Triangulation:
    """
    Class to build a length-constrained simplicial complex on a 2D environment.
    """

    def __init__(self, env: Env2D) -> None:
        """
        Initialize the Triangulation class.

        Args:
            env (Env2D): The 2D environment to be triangulated.

        Returns:
            None
        """
        # Env and parameters
        self.env: Env2D = env
        self.anchor_point: np.ndarray  # anchor point
        if env.anchor_point is not None:
            self.anchor_point = env.anchor_point
        else:
            print(
                f"{CmdColors.WARNING}[Triang]{CmdColors.WARNING} Undefined anchor "
                "point in Triangulation."
            )

        # Triangulation
        self.triangulated: bool = False  # flag to indicate if triangulation is done
        self.triang: shapely.MultiPolygon  # triangulation
        self.triang_tree: shapely.STRtree  # tree from triangulation (fast operations)
        self.root_idx: int  # index of the root triangle in the triangulation

        # Triangulation components
        self.vertices: np.ndarray  # vertices of the triangulation
        self.edges: np.ndarray  # edges of the triangulation
        self.triangles: np.ndarray  # faces of the triangulation
        self.vertices_dual: np.ndarray  # vertices of the dual graph (voronoi centroids)
        self.edges_dual: np.ndarray  # edges of the dual graph (edges between centroids)

        # Counters
        self.n_triangs: int  # number of triangles (same as dual graph vertices)
        self.n_vertices: int  # number of vertices in triangulation
        self.n_edges: int
        self.n_edges_dual: int

        # Simplicial complex
        # NOTE: the triangles are expressed as a tuple (idx, signature) where the index
        # corresponds to a triangle (or dual graph node), and signature is a list[int]
        # NOTE: the edges of the simplicial complex are pairs of integers pointing to
        # the lifted triangles in self.triangles_lift
        self.triangles_lift: list[tuple[int, list[int]]] = []
        self.edges_lift: list[list[int]] = []

        # Termination criteria for lifting (with large default values)
        self.max_lifted_triangles: int = 500  # max number of triangles to expand
        self.max_dist: float = 1e3  # Maximum distance between anchor point and vertices

        # Debugging
        self.DEBUG: bool = False

    def set_max_dist(self, max_dist: float) -> None:
        """
        Setter for max distance

        Args:
            max_dist (float): maximum distance between anchor point and vertices
        """
        if not isinstance(max_dist, (int, float)):
            raise TypeError("max_dist must be a number.")
        if max_dist <= 0:
            raise ValueError("max_dist must be a positive number.")
        self.max_dist = max_dist

    def set_max_triangles(self, max_triangles: int) -> None:
        """
        Setter for max number of lifted triangles

        Args:
            max_triangles (int): maximum number of lifted triangles
        """
        if not isinstance(max_triangles, int):
            raise TypeError("max_triangles must be an integer.")
        if max_triangles <= 0:
            raise ValueError("max_triangles must be a positive integer.")
        self.max_lifted_triangles = max_triangles

    def triangulate(self) -> None:
        """
        Perform triangulation on the 2D environment.
        """

        # Perform the constrained Delaunay triangulation using the shapely method
        self.triang = shapely.constrained_delaunay_triangles(self.env.free_workspace)
        self.triang = shapely.MultiPolygon(self.triang)  # change data type
        self.triang_tree = shapely.STRtree(
            self.triang.geoms
        )  # for fast geometry lookup
        self.n_triangs = len(self.triang.geoms)  # number of triangles in triangulation

        # find index root triangle (where anchor point lies)
        self.root_idx = int(
            self.triang_tree.query(
                Point(self.env.anchor_point),
                predicate="intersects",
            )[0]
        )

        # Fill triangulation components
        #     - vertices is a list of [x, y] coordinates
        #     - edges is a list of [idx_1, idx_2] pairs with idx_1 < idx_2, where idx_i
        #       is the index of a vertex in self.vertices
        #     - vertices_dual is a list of [x, y] coordinates of the centroids of the
        #       triangles
        #     - edges_dual is a list of [idx_1, idx_2] pairs with idx_1 < idx_2, where
        #       idx_i is the index of a vertex in self.vertices_dual
        #     - triangles are defined as a [idx_1, idx_2, idx_3] list, where the indexes
        #       are sorted in increasing order and refer to elements of self.vertices
        #     - for simplicity, the triangles indexes are the same as the
        #       Shapely-generated Delaunay triangulation. More importantly, the nodes
        #       of the dual graph (i.e., the triangles centroids) are indexed the same
        #       way, which means the centroids and triangles share the same indices.
        self.vertices = np.unique(
            np.array(
                [
                    coord
                    for triangle in self.triang.geoms
                    for coord in triangle.exterior.coords[:-1]
                ]
            ),
            axis=0,
        )  # [x, y]
        self.edges = np.array([], dtype=int).reshape(0, 2)  # [v1, v2]
        self.vertices_dual = np.zeros([self.n_triangs, 2], dtype=float)  # [x, y]
        self.edges_dual = np.array([], dtype=int).reshape(0, 2)  # [v1, v2]
        self.triangles = np.zeros([self.n_triangs, 3], dtype=int)  # [v1, v2, v3]
        for idx, triang in enumerate(self.triang.geoms):

            # Dual vertices
            self.vertices_dual[idx] = triang.centroid.coords[0]

            # Triangles vertices indexes
            indexes: np.ndarray = np.array(
                [
                    int(self.find_vertex_idx(triang.boundary.coords[0])),
                    int(self.find_vertex_idx(triang.boundary.coords[1])),
                    int(self.find_vertex_idx(triang.boundary.coords[2])),
                ]
            )
            self.triangles[idx] = np.sort(indexes)

            # Fill edges and dual edges
            edge: LineString
            for edge in self.list_triangle_edges(triang):

                # Find indexes of endpoints of edge
                idx1 = int(self.find_vertex_idx(edge.coords[0]))
                idx2 = int(self.find_vertex_idx(edge.coords[1]))
                idx1, idx2 = (idx1, idx2) if idx1 <= idx2 else (idx2, idx1)  # sort

                # Append edge to list
                if not (self.edges == [idx1, idx2]).all(axis=1).any():
                    self.edges = np.append(self.edges, [[idx1, idx2]], axis=0)

                # Find index of the triangles sharing the edge
                indexes: list[int] = list(
                    self.triang_tree.query(edge, predicate="covered_by")
                )
                idx1 = int(idx)  # current triangle index
                if len(indexes) == 2:
                    indexes.remove(idx1)
                    idx2 = int(indexes[0])  # neighboring triangle index
                    idx1, idx2 = (idx1, idx2) if idx1 <= idx2 else (idx2, idx1)  # sort
                else:
                    continue  # boundary edge

                # Add edge to list
                if not (self.edges_dual == [idx1, idx2]).all(axis=1).any():
                    self.edges_dual = np.append(self.edges_dual, [[idx1, idx2]], axis=0)

        # Update counters
        self.n_vertices = self.vertices.shape[0]
        self.n_edges = self.edges.shape[0]
        self.n_edges_dual = self.edges_dual.shape[0]

        # Set triangulated flag
        self.triangulated = True

    def find_vertex_idx(self, p: np.ndarray) -> int | None:
        """
        Find the index of a vertex in the self.vertices array (primary vertices).

        Args:
            p (np.ndarray): The point to find.

        Returns:
            int | None: The index of the vertex if found, None otherwise.
        """
        idx = np.where(np.all(self.vertices == p, axis=1))[0]
        return idx[0] if idx.size > 0 else None

    def list_triangle_edges(self, triangle: Polygon | int) -> list[LineString | int]:
        """
        Returns the edges of a triangle.

        Args:
            triangle (int | Polygon): The triangle to get the edges from. If the type
            is Polygon, the edges will be extracted from the Polygon object. If the type
            is int, the edges will be extracted from the self.faces ndarray.

        Returns:
            list[LineString | int]: The edges of the triangle. If the input type is
            Polygon, the output will be a list of shapely LineString objects. If the
            input is an int, the output will be a list of integers referring to the
            vertex indices in the self.vertices np.ndarray.
        """
        if isinstance(triangle, Polygon):
            return list(
                map(
                    LineString,
                    zip(triangle.exterior.coords[:-1], triangle.exterior.coords[1:]),
                )
            )
        elif isinstance(triangle, int):
            return []  # TODO: implement this
        else:
            return []

    def get_neighbors(self, idx: int) -> list[int]:
        """
        Returns the neighboring triangles of a given triangle. The search is performed
        through the dual graph since the triangle indices and dual nodes indices
        coincide.

        Args:
            idx (int): The index of the triangle (or dual node) to find the neighbors of

        Returns:
            list[int]: A list of indices of the neighboring triangles (or nodes of the
                dual graph connected by a dual edge).
        """
        neighbors = []
        for edge in self.edges_dual:
            if idx in edge:
                neighbors.append(int(edge[0]) if edge[1] == idx else int(edge[1]))
        return neighbors

    def lift_triangulation(self) -> None:
        """
        Turn the triangulated environment into a length-constrained simplicial complex,
        and lift it to obtain a manifold corresponding to a subset of the universal
        cover of the environment.

        Returns:
            None
        """
        # Check if env was triangulated (if not, execute triangulation)
        if not self.triangulated:
            self.triangulate()

        # Initialize queues
        open_queue: list[int] = []  # list of triangles to lift
        closed_queue: list[int] = []  # list of triangles already visited

        # Initialize counter (termination condition)
        n: int = 0

        # Initial conditions
        # TODO: the distance between the anchor and the vertices should technically be
        # checked before adding it, but for the time being we assume that they are valid
        # as otherwise the triangle could not be added at all
        open_queue.append((self.root_idx, [], 0))

        # Initialize temporary variables
        idx: int  # index of the current triangle (dual node)
        sign: list[int]  # signature of path from anchor to dual node along dual graph
        parent_idx: int  # index of the parent triangle

        # Main lifting loop
        while open_queue and n < self.max_lifted_triangles:

            # Pop the next triangle to lift and unpack it
            idx, sign, parent_idx = open_queue.pop(0)

            # Check if the triangle is valid
            # TODO: this section can be updated by only checking the vertices that are
            # not shared with the parent triangle, as those have already been checked
            # TODO: check distance of edges from anchor point

            # Add element to graph; append it to closed queue; increase counter
            n += 1  # Increase counter
            self.triangles_lift.append((idx, sign))
            i, j = sorted((n - 1, parent_idx))  # sort edge indices (smaller first)
            self.edges_lift.append((i, j))
            closed_queue.append((idx, sign))

            # Add new triangles to the open queue
            # NOTE: index of current triangle is added to keep track of the parent
            for neighbor_idx in self.get_neighbors(idx):
                edge = self.vertices_dual[[idx, neighbor_idx], :]
                neighbor_sign = sign + curves.compute_signature(edge, self.env)
                neighbor_sign = curves.simplify_signature(neighbor_sign)
                if (neighbor_idx, neighbor_sign) not in closed_queue:
                    open_queue.append((neighbor_idx, neighbor_sign, n - 1))

        # Print debug info
        if self.DEBUG:
            if n >= self.max_lifted_triangles:
                print(
                    f"{CmdColors.WARNING}[Triang]{CmdColors.WARNING} Reached max "
                    f"number of triangles ({self.max_lifted_triangles}) during lifting."
                )
            elif not open_queue:
                print(
                    f"{CmdColors.INFO}[Triang]{CmdColors.INFO} No more triangles in "
                    "the open queue."
                )

    def check_length(self, p: np.ndarray) -> bool:
        """
        Check if the length of the geodesic between one point and the anchor point over
        the simplicial complex is within the length constraint.

        Args:
            p: the point to check

        Returns:
            bool: True if the length is less than self.max_dist, False otherwise.
        """
        # See https://www.uni-trier.de/fileadmin/fb4/prof/INF/DEA/Seminar0708/Hershberger-Snoeyink3.pdf  # pylint: disable=line-too-long
        # TODO: complete implementation
