from __future__ import annotations

import numpy as np

from ..env.env_2d import Env2D


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
        self.anchor_point: np.ndarray
        self.max_dist: float  # Maximum distance between anchor point and vertices

        # Triangulation
        # TODO: check shapely methods

    def triangulate(self, method: str) -> None:
        """
        Perform triangulation on the 2D environment.

        Args:
            method (str): The method to use for triangulation.

        Returns:
            None
        """
        # TODO: complete implementation
        # https://shapely.readthedocs.io/en/2.0.6/reference/shapely.delaunay_triangles.html
        # https://shapely.readthedocs.io/en/stable/reference/shapely.constrained_delaunay_triangles.html
        # https://web.stanford.edu/~cm5/tm.pdf
        # https://mathoverflow.net/questions/264330/not-all-manifolds-can-be-triangulated-in-which-dimensions

        # TODO: useful links about sampling from triangulations/simplicial complexes
        # https://codereview.stackexchange.com/questions/69833/generate-sample-coordinates-inside-a-polygon#comment127620_69839 # pylint: disable=line-too-long
        # https://cs.stackexchange.com/questions/14007/random-sampling-in-a-polygon

    def build_simplicial_complex(self) -> None:
        """
        Turn the triangulated environment into a length-constrained simplicial complex,
        and lift it to obtain a manifold corresponding to a subset of the universal
        cover of the environment.

        Returns:
            None
        """
        # TODO: complete implementation

    def check_length(self, p: np.ndarray) -> bool:
        """
        Check if the length of the geodesic between one point and the anchor point over
        the simplicial complex is within the length constraint.

        Args:
            p: the point to check

        Returns:
            bool: True if the length is less than self.max_dist, False otherwise.
        """
        # TODO: complete implementation
