from __future__ import annotations

import numpy as np
import shapely
from shapely import LineString, Point, Polygon

from ..utils import curves
from ..utils.colors import CmdColors
from .env_2d import Env2D


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

        # Max dist between anchor point and vertices (termination criterion for lifting)
        self.max_dist: float  # Maximum distance between anchor point and vertices

        # Triangulation
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

        # Debugging
        self.DEBUG: bool = False

    def set_max_dist(self, max_dist: float) -> None:
        """
        Setter for max distance

        Args:
            max_dist (float): maximum distance between anchor point and vertices
        """
        self.max_dist = max_dist

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
        self.root_idx = self.triang_tree.query(
            Point(self.env.anchor_point),
            predicate="intersects",
        )[
            0
        ]  # find index root triangle (where anchor point lies)

        # Fill triangulation components
        # NOTE: the triangles indexes (and thus the dual graph vertices indexes) are the
        #       same as the Shapely-generated Delaunay triangulation
        # NOTE: the edges are defined as [idx_1, idx_2] where idx_1 < idx_2
        # NOTE: the triangles are defined as [idx_1, idx_2, idx_3] in CW order starting
        #       from the lowest index

        # Fill primary vertices
        self.vertices = np.unique(
            np.array(
                [
                    coord
                    for triangle in self.triang.geoms
                    for coord in triangle.exterior.coords[:-1]
                ]
            ),
            axis=0,
        )

        # Fill dual vertices
        self.vertices_dual = np.array(
            [triangle.centroid.coords[0] for triangle in self.triang.geoms]
        )

        # Fill edges, edges_dual, and triangles lists
        self.edges = np.array([]).reshape(0, 3)  # [v1, v2, len]
        self.edges_dual = np.array([]).reshape(0, 3)  # [v1, v2, len]
        self.triangles = np.array([]).reshape(0, 3)  # [v1, v2, v3]
        for idx, triang in enumerate(self.triang.geoms):

            # Fill triangles list
            # Find triangles vertices indexes
            indexes = np.array(
                [
                    int(self.find_vertex_idx(triang.boundary.coords[0])),
                    int(self.find_vertex_idx(triang.boundary.coords[1])),
                    int(self.find_vertex_idx(triang.boundary.coords[2])),
                ]
            )

            # Add triangle to the list
            self.triangles = np.append(self.triangles, [np.sort(indexes)], axis=0)

            # Fill edges and dual edges
            for edge in self.list_triangle_edges(triang):

                # Find indexes of endpoints of edge
                idx1 = int(self.find_vertex_idx(edge.coords[0]))
                idx2 = int(self.find_vertex_idx(edge.coords[1]))
                idx1, idx2 = (idx1, idx2) if idx1 <= idx2 else (idx2, idx1)  # sort

                # Find edge length
                dist = np.linalg.norm(self.vertices[idx1] - self.vertices[idx2])

                # Append edge to list
                self.edges = np.append(self.edges, [[idx1, idx2, dist]], axis=0)

                # Find index of dual edge endpoints (triang and triangle across edge)
                indexes = list(self.triang_tree.query(edge, predicate="covered_by"))
                idx1 = int(idx)  # current triangle index
                if len(indexes) == 2:
                    indexes.remove(idx1)
                    idx2 = int(indexes[0])  # neighboring triangle index
                    idx1, idx2 = (idx1, idx2) if idx1 <= idx2 else (idx2, idx1)  # sort
                else:
                    continue

                # Find edge length
                dist = np.linalg.norm(
                    self.vertices_dual[idx1] - self.vertices_dual[idx2]
                )

                # Add edge to list
                self.edges_dual = np.append(
                    self.edges_dual, [[idx1, idx2, dist]], axis=0
                )

        # Remove duplicates from arrays (arrays are already sorted along axis 1)
        self.edges = np.unique(self.edges, axis=0)
        self.edges_dual = np.unique(self.edges_dual, axis=0)
        self.triangles = np.unique(self.triangles, axis=0)  # should not need changes

        # Update counters
        self.n_triangs = len(self.triang.geoms)
        self.n_vertices = self.vertices.shape[0]
        self.n_edges = self.edges.shape[0]
        self.n_edges_dual = self.edges_dual.shape[0]

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
        # Initialize queues
        open_queue: list[int] = []  # list of triangles to lift
        closed_queue: list[int] = []  # list of triangles already visited

        # Initialize counter (termination condition)
        n_max: int = 20  # max number of triangles
        n: int = 0

        # Initial conditions
        # TODO: the distance between the anchor and the vertices should technically be
        # checked before adding it, but for the time being we assume that they are valid
        # as otherwise the triangle could not be added at all
        open_queue.append((self.root_idx, [], -1))

        # Initialize temporary variables
        idx: int  # index of the current triangle (dual node)
        sign: list[int]  # signature of path from anchor to dual node along dual graph
        parent_idx: int  # index of the parent triangle

        # Main lifting loop
        while open_queue and n < n_max:

            # Pop the next triangle to lift and unpack it
            idx, sign, parent_idx = open_queue.pop(0)

            # Check if the triangle is valid
            # TODO: this section can be updated by only checking the vertices that are
            # not shared with the parent triangle, as those have already been checked
            # TODO: check distance of edges from anchor point

            # Add element to graph; append it to closed queue; increase counter
            n += 1  # Increase counter
            self.triangles_lift.append((idx, sign))
            i, j = sorted((n, parent_idx))  # sort edge indices (smaller first)
            self.edges_lift.append((i, j))
            closed_queue.append((idx, sign))

            # Add new triangles to the open queue
            # NOTE: index of current triangle is added to keep track of the parent
            for neighbor_idx in self.get_neighbors(idx):
                edge = self.vertices_dual[[idx, neighbor_idx], :]
                neighbor_sign = sign + curves.compute_signature(edge, self.env)
                neighbor_sign = curves.simplify_signature(neighbor_sign)
                if (neighbor_idx, neighbor_sign) not in closed_queue:
                    open_queue.append((neighbor_idx, neighbor_sign, n))

        # Print debug info
        if self.DEBUG:
            if n >= n_max:
                print(
                    f"{CmdColors.WARNING}[Triang]{CmdColors.WARNING} Reached max "
                    f"number of triangles ({n_max}) during lifting."
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
