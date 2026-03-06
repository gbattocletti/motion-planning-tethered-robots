from __future__ import annotations

import pickle
from collections.abc import Callable
from datetime import datetime

import numpy as np
import shapely
from shapely import LineString, Point, Polygon

from tethered_planning.env.env_2d import Env2D
from tethered_planning.plan import graph_search
from tethered_planning.utils import curves, entanglement
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
        #     - vertices is a list of [x, y] coordinates
        #     - edges is a list of [idx_1, idx_2] pairs with idx_1 < idx_2, where idx_i
        #       is the index of a vertex in self.vertices
        #     - vertices_dual is a list of [x, y] coordinates of the centroids of the
        #       triangles
        #     - edges_dual is a list of [idx_1, idx_2] pairs with idx_1 < idx_2, where
        #       idx_i is the index of a vertex in self.vertices_dual
        #     - triangles are defined as a [idx_1, idx_2, idx_3] list, where the indexes
        #       are sorted in increasing order and refer to elements of self.vertices
        # NOTE: for simplicity, the triangles indexes are the same as the
        # Shapely-generated Delaunay triangulation. More importantly, the nodes of the
        # dual graph (i.e., the triangles centroids) are indexed the same way, which
        # means the centroids and triangles share the same indices.
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
        #     - vertices_lift: lifted primal graph vertices (vertices of triangles along
        #       with their signature)
        #     - edges_lift: lifted primal graph edges represented as [int, int] tuples
        #       of indexes pointing to the elements of vertices_lift
        #     - vertices_dual_lift: a tuple (idx, signature) representing a triangle in
        #       the simplicial complex, where the index corresponds to a triangle in the
        #       base triangulation (identified as a dual graph node, i.e., the centroid
        #       of such triangle), and signature is a list[int] identifying the homotopy
        #       class of the path to the triangle;
        #     - triangles_lift: a list [idx1, idx2, idx3] where the indexes point to the
        #       lifted vertices (self.vertices_lift).
        #     - edges_dual_lift: a list [idx1, idx2] where the indexes point to the
        #       lifted vertices (self.vertices_dual_lift). A tuple indicates adjacency
        #       between the two triangles in the simplicial complex, i.e.,
        #       dges_dual_lift correspond to the edges of the dual lifted graph;
        #     - parent_dual_lift: a dictionary mapping each lifted dual vertex index
        #       to the index of its parent lifted dual vertex (which is unique). This
        #       structure can be useful to reconstruct paths in the simplicial complex.
        self.vertices_lift: list[tuple[int, list[int]]]
        self.edges_lift: list[list[int, int]]
        self.triangles_lift: list[list[int, int, int]]
        self.vertices_dual_lift: list[tuple[int, list[int]]]
        self.edges_dual_lift: list[tuple[int, int]]
        self.parent_dual_lift: dict[int, int]

        # Entanglement state of simplices
        # Lists of boolean with the same length as the corresponding lifted simplicial
        # component, where True indicates that an entanglement-free path to the
        # simplex exists, and False indicates that it does not.
        self.entanglement_function: Callable | None = None
        self.entanglement_vertices_lift: list[bool]
        self.entanglement_triangles_lift: list[bool]
        self.entanglement_vertices_dual_lift: list[bool]

        # List for extra simplices (conservativeness reduction)
        # Each simplex is saved as a tuple containing
        #   - the list of vertices
        #   - the index of the parent triangle in the lifted triangulation
        self.extra_simplices: list[tuple[list[np.ndarray], int]]

        # Termination criteria for lifting (large default values)
        self.max_lifted_triangles: int = 1000  # max number of triangles to expand
        self.max_dist: float = 10.0  # max distance between anchor point and vertices

        # Debug info
        self.INFO: bool = False
        self.DEBUG: bool = False

    def save(self, filename: str = None) -> str:
        """
        Save the triangulation to a file.

        Args:
            filename (str, optional): The name of the file to save the triangulation to.
                If None, a default name is generated based on the current date and time.
                Default is None.

        Returns:
            str: The name of the file where the triangulation was saved.
        """
        # Build filename
        if filename is None:
            now = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
            filename = f"triangulation-{now}.pkl"
        else:
            if not filename.endswith(".pkl"):
                filename += ".pkl"

        # Write triangulation to file
        with open(filename, "wb") as f:
            pickle.dump(self, f)

        # Return filename
        return filename

    def load(self, filename: str) -> None:
        """
        Load a triangulation from a file.

        Args:
            filename (str): The name of the file to load the triangulation from.

        Returns:
            None
        """
        with open(filename, "rb") as f:
            loaded_triangulation = pickle.load(f)

        # Update current triangulation with loaded triangulation
        self.__dict__.update(loaded_triangulation.__dict__)

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

    def set_entanglement_definition(self, entanglement_definition: str) -> None:
        """
        Setter for entanglement definition

        Args:
            entanglement_definition (str): the entanglement definition to use for
                checking the entanglement state of the simplices.

        Raises:
            ValueError: if the input entanglement definition is not recognized.
        """
        if not isinstance(entanglement_definition, str):
            raise TypeError("ent_def must be a string.")
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

    def lift_triangulation(self, **kwargs) -> None:
        """
        Turn the triangulated environment into a length-constrained simplicial complex,
        and lift it to obtain a manifold corresponding to a subset of the universal
        cover of the environment.

        Kwargs:
            check_distance (bool, optional): whether to check the distance between the
                anchor point and the vertices of each triangle before adding it to the
                lifted triangulation. Default is True.
            search_algorithm (str, optional): The search algorithm to use for finding
                the candidate path. The argument is passed directly to the method
                Triangulation.geodesic_distance.
            check_entanglement (bool, optional): whether to check the entanglement of
                each triangle before adding it to the lifted triangulation. Default is
                False.
            reduce_conservativeness (bool, optional): whether to reduce the
                conservativeness of the simplicial complex by adding extra triangles at
                the boundaries of the simplicial complex. Default is False. If True,
                adds the additional triangles. The extra triangles are evaluated with
                respect to the followig criteria:
                -   no extra triangles if check_distance is False and check_entanglement
                    is False;
                -   extra triangles must be length-admissible if check_distance is True
                    and check_entanglement is False;
                -   extra triangles must be both length-admissible and
                    entanglement-admissible if check_distance is True and
                    check_entanglement is True .

        Returns:
            None
        """
        # Parse kwargs
        check_distance: bool = True  # default value
        search_algorithm: str = "parent"  # default value
        check_entanglement: bool = False  # default value
        reduce_conservativeness: bool = False  # default value
        for key, value in kwargs.items():
            if key == "check_distance":
                if not isinstance(value, bool):
                    raise TypeError("check_distance must be a boolean.")
                check_distance = value
            elif key == "search_algorithm":
                # NOTE: the availability of the specified method is specified in
                # geodesic_distance.
                if not isinstance(value, str):
                    raise TypeError("search_algorithm must be a string.")
                search_algorithm = value
            elif key == "check_entanglement":
                if not isinstance(value, bool):
                    raise TypeError("check_entanglement must be a boolean.")
                check_entanglement = value
            elif key == "reduce_conservativeness":
                if not isinstance(value, bool):
                    raise TypeError("reduce_conservativeness must be a boolean.")
                if (
                    reduce_conservativeness is True
                    and check_distance is False
                    and check_entanglement is False
                ):
                    print(
                        f"{CmdColors.WARNING}[Triang]{CmdColors.WARNING} "
                        "reduce_conservativeness is set to True, but check_distance "
                        "and check_entanglement are both False. This setting will be "
                        "ignored."
                    )
                    reduce_conservativeness = False
                else:
                    reduce_conservativeness = value
            else:
                raise KeyError(f"Unknown keyword argument: {key}")

        if check_entanglement and self.entanglement_function is None:
            raise ValueError(
                "Error: entanglement function not set. Please set the entanglement "
                "definition using set_entanglement_definition() method before building "
                "the homotopy-augmented graph with entanglement checking enabled."
            )

        # Check if env was triangulated (if not, execute triangulation)
        if not self.triangulated:
            self.triangulate()

        # Initialize lifted triangulation
        self.vertices_lift = []  # lifted primal graph vertices
        self.edges_lift = []  # lifted primal graph edges
        self.triangles_lift = []  # lifted triangles
        self.vertices_dual_lift = []  # lifted dual graph vertices
        self.edges_dual_lift = []  # lifted dual graph edges
        self.parent_dual_lift = {}  # parent mapping

        # Initialize entanglement state of simplices
        self.entanglement_vertices_lift = []
        self.entanglement_triangles_lift = []
        self.entanglement_vertices_dual_lift = []

        # Initialize list of extra simplices (conservativeness reduction)
        self.extra_simplices = []

        # Initialize queues
        open_queue: list[int] = []  # list of triangles to lift
        closed_queue: list[int] = []  # list of triangles already visited

        # Initialize counter (termination condition)
        n: int = 0

        # Initial conditions
        # Check that all the vertices of the root triangle are within max_dist. This
        # test should never fail as otherwise no triangle can be added in the lift.
        # However, we keep it for safety. Entanglement is assumed to be satisfied for
        # the root triangle if the length check is passed.
        for i in self.triangles[self.root_idx]:
            v = self.vertices[i, :]
            dist = np.linalg.norm(self.env.anchor_point - v)
            if dist > self.max_dist:
                raise ValueError(
                    "The root triangle is too small and its vertices cannot be reached "
                    "from the anchor point. Consider increasing max_dist."
                )
            else:
                s = curves.compute_signature(
                    np.array([self.env.anchor_point, v]), self.env, simplify=False
                )
                self.vertices_lift.append((int(i), s))
        self.edges_lift.append([0, 1])  # manually append first three edges
        self.edges_lift.append([1, 2])
        self.edges_lift.append([0, 2])
        self.triangles_lift.append([0, 1, 2])
        self.parent_dual_lift[0] = -1  # root triangle has no parent

        # Mark first simplices as entanglement admissible if check_entanglement is True.
        if check_entanglement is True:
            for i in range(3):
                self.entanglement_vertices_lift.append(True)  # for 3 vertices
            self.entanglement_vertices_dual_lift.append(True)
            self.entanglement_triangles_lift.append(True)

        # If the test is passed, add root triangle to open queue
        # Elements in the open queue have the following elements:
        # - idx: index of the triangle in the base triangulation
        # - sign: signature of the path from anchor to triangle along dual graph
        # - parent_idx: index of the parent triangle in the lifted triangulation
        # - d: approximate distance from anchor point to triangle centroid. The
        #      approximation is always an upper bound on the actual distance, and can be
        #      used to skip expensive geodesic distance computations when the triangle
        #      is close enough to the anchor point.
        open_queue.append((self.root_idx, [], -1, 0))

        # Initialize temporary variables
        idx: int  # index of the current triangle (dual node)
        sign: list[int]  # signature of path from anchor to dual node along dual graph
        parent_idx: int  # index of the parent triangle
        d_approx: float  # approximate distance from anchor to triangle centroid

        # Initialize counter for new lifted vertex. Start from 2 as three vertices are
        # already added from the root triangle (last occupied index is 2)
        vert_lift_n: int = 2

        # Main lifting loop
        while open_queue and n < self.max_lifted_triangles:

            # Pop the next triangle to lift and unpack it
            # NOTE: idx is the index of a triangle in base triangulation (not lifted)
            # NOTE: sign is the signature of the centroid identified by idx
            # NOTE: parent_idx is the index of a triangle in the lifted triangulation
            idx, sign, parent_idx, d_approx = open_queue.pop(0)
            n += 1  # Increase counter

            # Add new dual vertex and dual edge to lifted graph
            # NOTE: these two elements are required for geodesic_distance to be computed
            # in the 'if' check below. The other lifted graph components are added later
            self.vertices_dual_lift.append((idx, sign))
            i, j = sorted((n - 1, parent_idx))  # sort edge indices (smaller first)
            if not (i == -1 or j == -1):  # no edge when adding root triangle
                self.edges_dual_lift.append((i, j))
            self.parent_dual_lift[n - 1] = parent_idx  # add parent mapping to dict

            # Check if the triangle is valid
            # Check that the geodesic connecting each vertex to the anchor point is
            # less than self.max_dist, and/or if it is not entangled. In practice, only
            # one vertex is checked for each new triangle, since the other two are
            # shared with the parent triangle, which has already been checked in a
            # previous iteration (when the parent was added).
            if (
                check_distance is True or check_entanglement is True
            ) and parent_idx != -1:  # skip for root triangle

                # Reset useful variables
                length_admissible: bool = False
                entanglement_admissible: bool = False
                geodesic: None | np.ndarray = None

                # Find the new vertex added with respect to the parent triangle (in the
                # base triangulation)
                new_vertex_idx = np.setdiff1d(
                    self.triangles[idx],
                    self.triangles[self.vertices_dual_lift[parent_idx][0]],
                )
                new_vertex = self.vertices[new_vertex_idx, :][0]

                # Compute distance between anchor point and new vertex. If the approx
                # distance is small enough, skip expensive exact geodesic computation
                d_approx_new_vertex = d_approx + np.linalg.norm(
                    new_vertex
                    - self.vertices_dual[self.vertices_dual_lift[parent_idx][0]]
                )
                if d_approx_new_vertex > self.max_dist or check_entanglement is True:
                    # If entanglement is checked, then the geodesic is needed for the
                    # entanglement check. Otherwise, an approximate distance is
                    # sufficient to check the length constraint.
                    dist, geodesic = self.geodesic_distance(
                        new_vertex,
                        sign,
                        self.env.anchor_point,
                        [],
                        t1=idx,  # specify t2 since new_vertex always lies on boundary
                        t2=self.root_idx,  # specify t1 in case anchor lies on boundary
                        search_algorithm=search_algorithm,
                    )
                else:
                    dist = d_approx_new_vertex  # skip computation when close enough

                # Determine if triangle is valid
                if dist > self.max_dist:
                    length_admissible = False
                    entanglement_admissible = False
                elif check_entanglement is True:
                    length_admissible = True
                    entanglement_admissible = self.entanglement_function(
                        geodesic, self.env
                    )
                else:
                    length_admissible = True
                    entanglement_admissible = False  # entanglement ignored

            else:
                # Distance not checked (both check_distance and check_entanglement are
                # false), all triangles considered admissible. In this case,
                # entanglement is ignored.
                length_admissible = True
                entanglement_admissible = False

            # Resolve addition of simplices depending on result of length check
            # and entanglement check. Three scenarios are possible:
            # - not length admissible: simplices not added, continue to next iter
            # - length admissible but not entanglement admissible: simplices added,
            #   but marked as not entanglement admissible
            # - length admissible and entanglement admissible: simplices added and
            #   marked as entanglement admissible
            if length_admissible is False and parent_idx != -1:
                # Vertex too far: the triangle currently being checked is not valid
                # and must not be added to the lifted triangulation. This means it also
                # cannot be entanglement-admissible.

                # First, we perform the conservativeness correction operation (optional)
                if reduce_conservativeness is True:

                    # 1. Main triangle
                    # Find vertices of the triangle
                    old_nodes_idx = np.intersect1d(
                        self.triangles[idx],
                        self.triangles[self.vertices_dual_lift[parent_idx][0]],
                    )
                    v1 = self.vertices[old_nodes_idx[0]]
                    v2 = self.vertices[old_nodes_idx[1]]
                    p1 = new_vertex  # new vertex that cannot be added
                    p2 = (v1 + v2) / 2  # midpoint of the two shared (old) vertices
                    p = self.binary_search(p1, p2, sign, idx, search_algorithm)

                    # Add new simplex to list of extra simplices. Note that the new
                    # simplex is not directly connected to the simplicial complex, as
                    # some additional changes are required to do so.
                    # NOTE: parent_idxx refers to the lifted triangulation.
                    if p is not None:
                        self.extra_simplices.append(([v1, v2, p], parent_idx))

                    # TODO: compute also other 2 triangles
                    # 2. Side triangles (either 1 or 2)

                # Second we remove the last added info (which was added to enable the
                # computation of geodesic_distance) from the lifted triangulation.
                self.vertices_dual_lift.pop()
                self.edges_dual_lift.pop()
                self.parent_dual_lift.pop(n - 1)
                n -= 1  # Bring counter back (triangle was removed)

                # Third, we add the triangle to the closed queue to avoid checking
                # it again in the future.
                closed_queue.append((idx, sign))

                # Finally, we stop this iteration and move to the next triangle in
                # the open queue. This also skips the addition of the neighboring
                # triangles to the open queue, since they would be unreachable too.
                continue

            elif length_admissible is True and parent_idx != -1:
                # Length admissibility is ok. Add vertex and edges to lifted primal
                # graph, store vertices indexes in lifted triangles

                # Find signature of the new vertex
                sign_new_vertex = curves.simplify_signature(
                    sign
                    + curves.compute_signature(
                        np.array([self.vertices_dual[idx], new_vertex]),
                        self.env,
                        simplify=False,
                    )
                )

                # Add new vertex to lifted primal graph
                vert_lift_n += 1
                new_vertex_lift = (int(new_vertex_idx[0]), sign_new_vertex)
                if new_vertex_lift not in self.vertices_lift:
                    self.vertices_lift.append(new_vertex_lift)
                else:
                    raise ValueError(
                        f"Vertex ({new_vertex_idx}, {sign_new_vertex} already "
                        "present in lifted primal graph"
                    )  # sanity check (should never happen)

                # Add new edges to primal graph
                indexes: list = [vert_lift_n]
                shared_vertices_idx = np.intersect1d(
                    self.triangles[idx],
                    self.triangles[self.vertices_dual_lift[parent_idx][0]],
                )
                parent_lifted_vertices_idx = self.triangles_lift[parent_idx]
                for p_idx in parent_lifted_vertices_idx:
                    if self.vertices_lift[p_idx][0] in shared_vertices_idx:
                        self.edges_lift.append(sorted((p_idx, vert_lift_n)))
                        indexes.append(p_idx)

                # Add vertices to lifted triangles
                self.triangles_lift.append(sorted(indexes))

                # Mark simplices as entanglement admissible or not
                if entanglement_admissible is True:
                    self.entanglement_vertices_lift.append(True)
                    self.entanglement_vertices_dual_lift.append(True)
                    self.entanglement_triangles_lift.append(True)
                else:
                    self.entanglement_vertices_lift.append(False)
                    self.entanglement_vertices_dual_lift.append(False)
                    self.entanglement_triangles_lift.append(False)

            # Add current triangle to closed queue to avoid checking it again
            closed_queue.append((idx, sign))

            # Add adjacent triangles to the open queue
            # NOTE: index of current triangle is added to keep track of the parent
            for neighbor_idx in self.get_neighbors(idx):
                edge = self.vertices_dual[[idx, neighbor_idx], :]
                neighbor_sign = curves.simplify_signature(
                    sign + curves.compute_signature(edge, self.env)
                )
                if (neighbor_idx, neighbor_sign) not in closed_queue:
                    d_approx_neighbor = d_approx + np.linalg.norm(
                        self.vertices_dual[neighbor_idx] - self.vertices_dual[idx]
                    )  # distance between current centroid and neighbor centroid
                    open_queue.append(
                        (neighbor_idx, neighbor_sign, n - 1, d_approx_neighbor)
                    )  # append tuple with all info about neighbor

        # Termination conditions
        if self.INFO or self.DEBUG:
            # Sanity check on lifted simplicial complex dimensions
            if not (
                len(self.vertices_lift)
                == len(self.vertices_dual_lift) * 3 - len(self.edges_dual_lift) * 2
            ):
                # NOTE: this check holds only because all the vertices of the triangles
                # coincide with obstacle vertices.
                raise ValueError(
                    f"{CmdColors.FAIL}[Triang]{CmdColors.ENDC} The dimension of the "
                    "lifted primal graph vertices does not match the number of lifted "
                    "triangles."
                )
            if not len(self.vertices_dual_lift) == len(self.triangles_lift):
                raise ValueError(
                    f"{CmdColors.FAIL}[Triang]{CmdColors.ENDC} The dimension of the "
                    "lifted dual graph vertices does not match the number of lifted "
                    "triangles."
                )

            # Print output status
            if n >= self.max_lifted_triangles:
                print(
                    f"{CmdColors.WARNING}[Triang]{CmdColors.ENDC} Warning: maximum "
                    f"number of triangles ({self.max_lifted_triangles}) reached "
                    "during lifting. Consider increasing n_max or reducing max_dist."
                )
            elif not open_queue:
                print(
                    f"{CmdColors.OKBLUE}[Triang]{CmdColors.ENDC} Simplicial complex "
                    f"built with {len(self.vertices_dual_lift)} triangles."
                )

    def geodesic_distance(
        self,
        p1: np.ndarray,
        s1: list[int],
        p2: np.ndarray,
        s2: list[int],
        **kwargs,
    ) -> tuple[float, np.ndarray]:
        """
        Measure the length of the shortest path (geodesic) between two points in the
        lifted triangulation.

        Args:
            p1 (np.ndarray): the first point to check
            s1 (list[int]): the signature of the first point
            p2 (np.ndarray): the second point to check
            s2 (list[int]): the signature of the second point

        Kwargs:
            t1 (int, optional): index of the triangle containing p1 (in the base space).
                If None, the triangle will be searched for. This option is useful when
                dealing with points on the boundary of a triangle, where it can be
                important to distinguish between different triangles. If t1 is None,
                the selection of the triangle in case of multiple intersections is
                arbitrary. Default is None.
            t2 (int, optional): index of the triangle containing p2 (in the base space).
            search_algorithm (str, optional): The search algorithm to use for finding
                the candidate path. Options are: {'astar', 'dijkstra', 'bfs', 'dfs',
                'parent'}. The method 'parent' can only be used to find paths to the
                triangle where the anchor point lies. Default is 'dfs'.

        Returns:
            length (float): length of the shortest path (geodesic) between two points.
            geodesic (np.ndarray): The geodesic path as a list of [x, y] coordinates.

        Raises:
            KeyError: If an unknown keyword argument is provided.
        """
        # Parse kwargs
        search_algorithm: str = "dfs"  # default value
        t1: int | None = None
        t2: int | None = None
        for key, value in kwargs.items():
            if key == "search_algorithm":
                if not isinstance(value, str):
                    raise TypeError("search_algorithm must be a string.")
                if search_algorithm not in [
                    "astar",
                    "dijkstra",
                    "bfs",
                    "dfs",
                    "parent",
                ]:
                    raise ValueError(
                        "search_algorithm must be one of the following: "
                        "{'astar', 'dijkstra', 'bfs', 'dfs', 'parent'}."
                    )
                if search_algorithm == "parent" and self.DEBUG:
                    print(
                        f"{CmdColors.WARNING}[Triang]{CmdColors.ENDC} Warning: the "
                        "'parent' search algorithm can only be used to find paths to "
                        "the triangle where the anchor point lies. The point (p2, s2) "
                        "will be ignored."
                    )
                search_algorithm = value
            elif key == "t1":
                if not isinstance(value, int):
                    raise TypeError("t1 must be an integer or None.")
                t1 = value
            elif key == "t2":
                if not isinstance(value, int):
                    raise TypeError("t2 must be an integer or None.")
                t2 = value
            else:
                raise KeyError(f"Unknown keyword argument: {key}")

        # Parse inputs
        if isinstance(p1, list):
            p1 = np.array(p1)
        if isinstance(p2, list):
            p2 = np.array(p2)

        # Initialize variables for triangle search
        intersections: list[int]
        tri_idx_1: int
        tri_idx_2: int

        # Find triangle containing (p1, s1)
        intersections = self.triang_tree.query(Point(p1), predicate="intersects")
        if t1 is None:
            tri_idx_1 = intersections[0]
        else:
            if not t1 in intersections:
                raise ValueError(
                    f"The provided triangle index t1={t1} does not contain point {p1}."
                )
            tri_idx_1 = t1
        lift_idx_1 = [
            i
            for i, (t_idx, t_sign) in enumerate(self.vertices_dual_lift)
            if t_idx == tri_idx_1 and t_sign == s1
        ]
        if not lift_idx_1:
            raise ValueError(
                f"The signature was not found in the lifted tree for point {p1}."
            )
        lift_idx_1 = lift_idx_1[0]

        # Find triangle containing (p2, s2)
        # NOTE: when using 'parent' this step is skipped as p2 is automatically assigned
        # to the anchor point.
        if search_algorithm != "parent":
            intersections = self.triang_tree.query(Point(p2), predicate="intersects")
            if t2 is None:
                tri_idx_2 = intersections[0]
            else:
                if not t2 in intersections:
                    raise ValueError(
                        f"The provided triangle index t2={t2} does not contain "
                        f"point {p2}."
                    )
                tri_idx_2 = t2
            lift_idx_2 = [
                i
                for i, (t_idx, t_sign) in enumerate(self.vertices_dual_lift)
                if t_idx == tri_idx_2 and t_sign == s2
            ]
            if not lift_idx_2:
                raise ValueError(
                    f"The signature was not found in the lifted tree for point {p2}."
                )
            lift_idx_2 = lift_idx_2[0]

        # Compute path between (p1, s1) and (p2, s2)
        alpha_lift: list[int] = []
        match search_algorithm:
            case "parent":
                alpha_lift = []
                current_idx = lift_idx_1
                while current_idx >= 0:  # root is -1
                    alpha_lift.append(current_idx)
                    current_idx = self.parent_dual_lift[current_idx]  # move to parent
                p2 = self.anchor_point  # set p2 to anchor point when using this method
            case "dfs":
                alpha_lift = graph_search.dfs(
                    self.vertices_dual_lift,
                    self.edges_dual_lift,
                    lift_idx_1,
                    lift_idx_2,
                )
            case "bfs":
                alpha_lift = graph_search.bfs(
                    self.vertices_dual_lift,
                    self.edges_dual_lift,
                    lift_idx_1,
                    lift_idx_2,
                )
            case ["astar", "dijkstra"]:
                alpha_lift = graph_search.a_star_search(
                    self.vertices_dual_lift,
                    self.edges_dual_lift,
                    lift_idx_1,
                    lift_idx_2,
                    h_augmented=True,
                    nodes_2d=self.vertices_dual,
                    use_heuristic=False,
                )

        # Project the representative path onto the 2D triangulation
        alpha: list[int] = [self.vertices_dual_lift[idx][0] for idx in alpha_lift]

        # Call shortest homotopic path (geodesic)
        geodesic = self.homotopic_shortest_path(
            alpha=alpha,
            p_init=p1,
            p_end=p2,
        )

        # Compute length of shortest path
        diffs: np.ndarray = np.diff(geodesic, axis=0)  # consecutive segment
        seg_lengths: np.ndarray = np.linalg.norm(diffs, axis=1)  # segment lengths
        length: float = np.sum(seg_lengths)

        # Return distance
        return length, geodesic

    def homotopic_shortest_path(
        self,
        alpha: list[int],
        p_init: np.ndarray | None = None,
        p_end: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Compute the shortest path between two points that is homotopic to a given path.

        The algorithm is based on:
            - J. Hershberger, J. Snoeyink, "Computing minimum length paths of a given
              homotopy class" (1994).
            - D. Lee, F. Preparata, "Euclidean Shortest Paths in the Presence of
              Rectilinear Barriers" (1984).
            - Reza Teshnizi, "The Funnel Algorithm Explained Visually" (2018)
              https://medium.com/@reza.teshnizi/the-funnel-algorithm-explained-visually-41e374172d2d

        Args:
            alpha (list[int]): Representative path to shorten. It uniquely identifies
                the homotopy class in which to find the shortest path. It is represented
                as a list of indices of the dual lifted graph nodes (triangles in the
                primal lifted graph).
            p_init (np.ndarray, optional): The starting point of the path. It must lie
                in the first triangle of alpha. Default is None, in which case the
                centroid of the first triangle is used.
            p_end (np.ndarray, optional): The ending point of the path. It must lie in
                the last triangle of alpha. Default is None, in which case the centroid
                of the last triangle is used.

        Returns:
            np.ndarray: The shortest path homotopic to alpha, represented as a list of
                [x, y] coordinates. The coordinates correspond to obstacle vertices,
                except for the first and last points, which coincide with p_init and
                p_end respectively (or with the centroids of the first and last
                triangles in alpha).
        """
        # Initialize data structures
        tail: np.ndarray = np.array([])  # shortest path (sequence of points)
        left: np.ndarray = np.array([])  # left side of the funnel (sequence of points)
        right: np.ndarray = np.array([])  # right side of the funnel (same as above)
        tri_idx: int  # current triangle index
        tri_idx_prev: int  # previous triangle index
        v_left: np.ndarray  # left vertex of the edge being crossed (between triangles)
        v_right: np.ndarray  # right vertex of the edge being crossed

        # Add initial point to the funnel
        # NOTE: for convenience, the apex of the funnel is always part of both sides of
        # the funnel and not part of the tail. At the end of the algorithm, the last
        # apex (current apex point after tracing alpha) is added to the tail.
        if p_init is not None:
            tri_idx_prev = self.triang_tree.query(
                Point(p_init),
                predicate="intersects",
            )
            if tri_idx_prev.size != 1:
                # p_init does not lie in a unique triangle. We assumeit lies in the
                # first triangle of alpha.
                tri_idx_prev = alpha[0]
                alpha = alpha[1:]
            else:
                tri_idx_prev = int(tri_idx_prev)  # convert to int
        else:
            p_init = self.vertices_dual[alpha[0]]
            tri_idx_prev = alpha[0]
            alpha = alpha[1:]
        left = np.array(p_init).reshape(1, 2)  # initialize left side of funnel
        right = left.copy()  # initialize right side of funnel

        # Trace alpha through the triangulation
        for tri_idx in alpha:

            # Skip if the same triangle (can happen if alpha is not simplified)
            if tri_idx == tri_idx_prev:
                continue

            # Find vertices of the edge being crossed
            shared_vertices = np.intersect1d(
                self.triangles[tri_idx_prev],
                self.triangles[tri_idx],
            )
            if len(shared_vertices) != 2:
                raise ValueError("Triangles do not share an edge")
            v_left, v_right = self.vertices[shared_vertices]  # NOTE: not sorted yet

            # Determine which vertex is left and which is right
            # NOTE: to do so, the angle between the previous triangle centroid and the
            # edge endpoints is computed. The angles to the endpoints are then compared.
            # After the sorting, v1 is the left vertex and v2 the right vertex. This
            # method is enabled by the fact that the difference between the two angles
            # is no greater than pi in absolute value.
            alpha_1 = np.atan2(
                v_left[1] - self.vertices_dual[tri_idx_prev, 1],
                v_left[0] - self.vertices_dual[tri_idx_prev, 0],
            )
            alpha_2 = np.atan2(
                v_right[1] - self.vertices_dual[tri_idx_prev, 1],
                v_right[0] - self.vertices_dual[tri_idx_prev, 0],
            )
            delta = (alpha_2 - alpha_1 + np.pi) % (2 * np.pi) - np.pi
            if delta > 0:
                v_left, v_right = (
                    v_right,
                    v_left,
                )  # swap to ensure v1 is left and v2 is right
            else:
                pass  # order is correct

            # Update funnel sides
            # NOTE: for each edge crossed except the first one, only one endpoint of the
            # edge change with respect to the previous time step. The if statement is
            # used to check this and avoid unnecessary computations.

            # Left side
            if not np.array_equal(v_left, left[-1]):

                # Check left side
                # During this check, the angle to the new point is compared to the
                # angle of the existing side of the funnel to determine if the new
                # point forms a supporting edge, i.e., if it makes the funnel narrower.
                for i in range(len(left) - 1, 0, -1):
                    angle_new = np.arctan2(
                        v_left[1] - left[i, 1],
                        v_left[0] - left[i, 0],
                    )
                    angle_old = np.arctan2(
                        left[i, 1] - left[i - 1, 1],
                        left[i, 0] - left[i - 1, 0],
                    )
                    delta = (angle_new - angle_old + np.pi) % (2 * np.pi) - np.pi
                    if delta > 0:
                        break  # angle formed by new point is larger than previous
                    elif delta <= 0:
                        left = left[:-1]  # remove last point from left

                else:
                    # This block is executed only if the for loop is not broken, i.e.,
                    # if the left side of the funnel has been completely emptied. In
                    # this case, points on the right side of the funnel must be checked
                    # as well to determine if the funnel (or part of it) has closed.
                    # NOTE: this block also contains the tail update.
                    new_idx: int = 0
                    if len(right) >= 2:
                        for i in range(len(right) - 1):
                            angle_new = np.arctan2(
                                v_left[1] - right[i, 1],
                                v_left[0] - right[i, 0],
                            )
                            angle_old = np.arctan2(
                                right[i + 1, 1] - right[i, 1],
                                right[i + 1, 0] - right[i, 0],
                            )
                            delta = (angle_new - angle_old + np.pi) % (
                                2 * np.pi
                            ) - np.pi
                            if delta <= 0:
                                # The funnel has closed up to this point. Therefore,
                                # this point from the right side becomes the new apex of
                                # the funnel. Therefore, it is replaced to the current
                                # only remaining point in the left side, which in turn
                                # is added to the tail
                                tail = (
                                    np.vstack((tail, right[i]))
                                    if tail.size > 0
                                    else right[i]
                                )  # move old apex to tail
                                left = right[i + 1].reshape(1, 2)  # new apex of funnel
                                new_idx = i + 1

                            elif delta > 0:
                                break

                    right = right[new_idx:].copy()  # keep right side from new_idx

                left = np.vstack((left, v_left))  # add new left point (always)

            # Right side
            if not np.array_equal(v_right, right[-1]):

                # Check right side
                # During this check, the angle to the new point is compared to the
                # angle of the existing side of the funnel to determine if the new
                # point forms a supporting edge, i.e., if it makes the funnel narrower.
                for i in range(len(right) - 1, 0, -1):
                    angle_new = np.arctan2(
                        v_right[1] - right[i, 1],
                        v_right[0] - right[i, 0],
                    )
                    angle_old = np.arctan2(
                        right[i, 1] - right[i - 1, 1],
                        right[i, 0] - right[i - 1, 0],
                    )
                    delta = (angle_new - angle_old + np.pi) % (2 * np.pi) - np.pi
                    if delta < 0:  # NOTE: comparison is inverted w.r.t. left side
                        break  # angle formed by new point is larger than previous
                    elif delta >= 0:
                        right = right[:-1]  # remove last point from right

                else:
                    # This block is executed only if the for loop is not broken, i.e.,
                    # if the right side of the funnel has been emptied (except for the
                    # apex). In this case, points on the left side of the funnel must be
                    # checked as well to determine if the funnel (or part of it) has
                    # closed.
                    # NOTE: this block also contains the tail update.
                    new_idx: int = 0
                    if len(left) >= 2:
                        for i in range(len(left) - 1):
                            angle_new = np.arctan2(
                                v_right[1] - left[i, 1],
                                v_right[0] - left[i, 0],
                            )
                            angle_old = np.arctan2(
                                left[i + 1, 1] - left[i, 1],
                                left[i + 1, 0] - left[i, 0],
                            )
                            delta = (angle_new - angle_old + np.pi) % (
                                2 * np.pi
                            ) - np.pi
                            if delta >= 0:
                                # The funnel has closed up to this point. Therefore,
                                # this point from the left side becomes the new apex of
                                # the funnel. Therefore, it is replaced to the current
                                # only remaining point in the right side, which in turn
                                # is added to the tail
                                tail = (
                                    np.vstack((tail, left[i]))
                                    if tail.size > 0
                                    else left[i]
                                )  # move old apex to tail
                                right = left[i + 1].reshape(1, 2)  # new apex of funnel
                                new_idx = i + 1

                            elif delta < 0:
                                break

                    left = left[new_idx:].copy()  # keep left side from new_idx

                right = np.vstack((right, v_right))  # add new right point (always)

            # Update previous triangle index
            tri_idx_prev = tri_idx

        # Final step
        if p_end is None:
            p_end = self.vertices_dual[tri_idx]

        # Check left side and add points to tail
        added_left: bool = False
        if left.size > 0:
            for i in range(len(left) - 1, 0, -1):
                angle_new = np.arctan2(
                    p_end[1] - left[i, 1],  # delta y
                    p_end[0] - left[i, 0],  # delta x
                )
                angle_old = np.arctan2(
                    left[i, 1] - left[i - 1, 1],
                    left[i, 0] - left[i - 1, 0],
                )
                delta = (angle_new - angle_old + np.pi) % (2 * np.pi) - np.pi
                if delta >= 0:
                    # Add all points from left side up to i (including i) to the tail
                    tail = (
                        np.vstack((tail, left[: i + 1]))
                        if tail.size > 0
                        else left[: i + 1]
                    )
                    added_left = True
                    break
                else:
                    continue

        # Check right side and add points to tail
        added_right: bool = False
        if right.size > 0:
            for i in range(len(right) - 1, 0, -1):
                angle_new = np.arctan2(
                    p_end[1] - right[i, 1],  # delta y
                    p_end[0] - right[i, 0],  # delta x
                )
                angle_old = np.arctan2(
                    right[i, 1] - right[i - 1, 1],
                    right[i, 0] - right[i - 1, 0],
                )
                delta = (angle_new - angle_old + np.pi) % (2 * np.pi) - np.pi
                if delta <= 0:
                    # Add all points from right side up to i (including i) to the tail
                    tail = (
                        np.vstack((tail, right[: i + 1]))
                        if tail.size > 0
                        else right[: i + 1]
                    )
                    added_right = True
                    break
                else:
                    continue

        # If no points were added left and right, add the first point of left (which
        # coincides with the first point of right), since, being the same, these two
        # points represents the last point of the tail. This step ensures that the whole
        # tail has been considered in the solution.
        if not added_left and not added_right:
            if np.array_equal(left[0], right[0]):
                tail = (
                    np.vstack((tail, left[0]))
                    if tail.size > 0
                    else left[0].reshape(1, 2)
                )

        # If tail is still empty, it means that the funnel never narrowed, so we
        # manually add the initial point to the tail
        if tail.size == 0:
            tail = p_init.reshape(1, 2)

        # Add final point to tail (always)
        tail = np.vstack((tail, p_end))

        # Return the shortest path
        return tail

    def binary_search(
        self,
        p1: np.ndarray,
        p2: np.ndarray,
        sign: list[int],
        idx: int,
        search_algorithm: str,
    ) -> np.ndarray | None:
        """
        Perform binary search along a line segment between two points p1 and
        p2 to find the point farthest from the anchor point in a given
        homotopy class while respecting the length constraint.

        Args:
            p1 (np.ndarray): the first endpoint of the line segment
            p2 (np.ndarray): the second endpoint of the line segment. Assumed to lie in 
                the same triangle as p1
            sign (list[int]): the signature of the triangle where the points lie
            idx (int): the index of the triangle where the points lie (expressed as a
                triangle index referring to the base triangulation)
            search_algorithm (str): graph search algorithm to use in the geodesic 
                distance computation.

        Returns
            np.ndarray | None: None indicates that no admissible point was found, 
                otherwise the point farthest from the anchor point is returned as a
                np.ndarray with the coordinates of the point.
        """
        # Function settings
        n_iter: int = 5

        # Initialize variables
        admissible_found: bool = False

        # Run binary search
        p = (p1 + p2) / 2  # initial point to start the binary search from
        for _ in range(n_iter):

            # Check if p is admissible
            dist, _ = self.geodesic_distance(
                p,
                sign,  # signature of triangle where p lies
                self.env.anchor_point,
                [],
                t1=idx,
                t2=self.root_idx,
                search_algorithm=search_algorithm,
            )

            # If p is admissible move closer to p1, otherwise closer to p2
            if dist <= self.max_dist:
                p2 = p
                admissible_found = True
            else:
                p1 = p

            # Update p as midpoint between p1 and p2
            p = (p1 + p2) / 2

        if admissible_found:
            return p
        else:
            return None
