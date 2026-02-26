"""
Functions to evaluate the entanglement state of curves. The module contains methods to
evaluate the entanglement state of tether configurations under different entanglement
definitons selected from the paper "Entanglement definitions for tethered robots:
Exploration and analysis", IEEE Access (2024).
"""

import numpy as np
import shapely
from shapely.geometry import LineString, MultiPoint, Polygon

from tethered_planning.env.env_2d import Env2D
from tethered_planning.utils import curves


def null_homotopy(curve: np.ndarray | LineString, env: Env2D) -> bool:
    """
    Compute the entanglement stte of the input curve by checking if the curve is null
    homotopic to the anchor point of the environment. The curve is assumed to have start
    and end at the anchor point.

    Args:
    curve (np.ndarray | LineString): piecewise affine curve representing the tether
        configuration to check the entanglement state of.
    env (Env2D): 2D environment, which include the obstacles used in the evaluation
        of the entanglement definition

    Returns:
        bool: 0 if the curve is entangled, 1 if it is not.

    Raises:
        TypeError: if the input curve is not a numpy array or a LineString.
        ValueError: if the input curve does not start and end at the anchor point.
    """
    # Preprocess points to obtain a np.ndarray
    if isinstance(curve, LineString):
        points = np.array(curve.coords)
    elif isinstance(curve, np.ndarray):
        points = curve
    else:
        raise TypeError("curve must be a numpy array or a LineString.")

    # Check if initial and final points match
    if not np.allclose(points[0], env.anchor_point) or not np.allclose(
        points[-1], env.anchor_point
    ):
        raise ValueError("curve must start and end at the anchor point.")

    # Check if any of the obstacles is fully encircled by the curve
    null_homotopy_polygon = Polygon(points)
    for obs in env.obstacle_polygons:
        if shapely.within(obs, null_homotopy_polygon):
            return False
    return True


def convex_hull(curve: np.ndarray | LineString, env: Env2D) -> bool:
    """
    Compute the entanglement stte of the input curve by checking if the convex hull of
    the curve intersects with the obstacle region.

    Args:
        curve (np.ndarray | LineString): piecewise affine curve representing the tether
            configuration to check the entanglement state of.
        env (Env2D): 2D environment, which include the obstacles used in the evaluation
            of the entanglement definition

    Returns:
        bool: 0 if the curve is entangled, 1 if it is not.

    Raises:
        TypeError: if the input curve is not a numpy array or a LineString.
    """
    # Preprocess points to obtain a MultiPoints object
    if isinstance(curve, LineString):
        points = MultiPoint(np.array(curve.coords))
    elif isinstance(curve, np.ndarray):
        points = MultiPoint(curve)
    else:
        raise TypeError("curve must be a numpy array or a LineString.")

    # Compute the convex hull
    hull = Polygon(points.convex_hull)

    # Check entanglement condition
    if shapely.intersection(hull, env.obstacle_region).area > 0:
        # Verify if intersection is only at the boundary, where it is acceptable
        if shapely.intersection(hull, env.obstacle_region) != shapely.intersection(
            hull, env.obstacle_region.boundary
        ):
            return False
    return True


def linear_homotopy(curve: np.ndarray | LineString, env: Env2D) -> bool:
    """
    Compute the entanglement stte of the input curve by checking if the linear homotopy
    of the curve to the anchor point of intersects with the obstacle region (i.e., if
    the curve is null homotopic along a linear homotopy).

    Args:
        curve (np.ndarray | LineString): piecewise affine curve representing the tether
            configuration to check the entanglement state of.
        env (Env2D): 2D environment, which include the obstacles used in the evaluation
            of the entanglement definition

    Returns:
        bool: 0 if the curve is entangled, 1 if it is not.

    Raises:
        TypeError: if the input curve is not a numpy array or a LineString.
        ValueError: if the input curve does not start or end at the anchor point.
    """
    # Preprocess curve to ensure correct data type
    if isinstance(curve, LineString):
        points = np.array(curve.coords)
    elif isinstance(curve, np.ndarray):
        points = curve
    else:
        raise TypeError("curve must be a numpy array or a LineString.")
    n = len(points)

    # anchor_idx = 0
    # start_idx = 1
    # end_idx = n - 1
    # Check if anchor point is first or last
    if np.allclose(points[0], env.anchor_point):
        anchor_idx = 0
        start_idx = 1
        end_idx = n - 1
    elif np.allclose(points[-1], env.anchor_point):
        anchor_idx = n - 1
        start_idx = n - 2
        end_idx = 0
    else:
        raise ValueError("curve must start or end at the anchor point.")

    # Iterate over the points to verify if any of the regions spanned by the curve
    # intersect with the obstacle region
    for idx in range(start_idx, end_idx, 1 if start_idx < end_idx else -1):
        triangle = Polygon([points[idx], points[idx + 1], points[anchor_idx]])
        if shapely.intersection(triangle, env.obstacle_region).area > 0:
            # Check if intersection is only at the boundary, where it is acceptable
            if shapely.intersection(
                triangle, env.obstacle_region
            ) != shapely.intersection(triangle, env.obstacle_region.boundary):
                return False
    return True


def local_visibility_homotopy(curve: np.ndarray | LineString, env: Env2D) -> bool:
    """
    Compute the entanglement stte of the input curve by checking if the local visibility
    homotopy condition holds for all the subsections of the curve.

    Args:
        curve (np.ndarray | LineString): piecewise affine curve representing the tether
            configuration to check the entanglement state of.
        env (Env2D): 2D environment, which include the obstacles used in the evaluation
            of the entanglement definition

    Returns:
        bool: 0 if the curve is entangled, 1 if it is not.

    Raises:
        TypeError: if the input curve is not a numpy array or a LineString.
    """
    # Preprocess curve to ensure correct data type
    if isinstance(curve, LineString):
        points = np.array(curve.coords)
    elif isinstance(curve, np.ndarray):
        points = curve
    else:
        raise TypeError("curve must be a numpy array or a LineString.")
    n = len(points)

    # Iterate over points to determine if there is any subsection of the curve that
    # violates the entanglement definition. To do so, check if the signature of
    # subsections of the curve match that of the underlying straight-line segment.
    for idx_1 in range(n - 2):
        for idx_2 in range(n - 1, idx_1 + 1, -1):
            curve_1 = np.array(points[idx_1 : idx_2 + 1])
            sign_1 = curves.compute_signature(curve_1, env, simplify=True)
            curve_2 = np.array([points[idx_1], points[idx_2]])
            if env.obstacle_region.intersects(LineString(curve_2)):
                continue  # if straight line is not feasible, skip check
            sign_2 = curves.compute_signature(curve_2, env, simplify=True)

            # Check entanglement state
            if sign_1 != sign_2:
                return False
    return True
