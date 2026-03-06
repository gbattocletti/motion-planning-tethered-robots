"""
Functions to work with curves. The module contains methods to manipulate curves,
including methods for resampling, interpolation, computation of signature, shortening,
and visual generation.
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backend_bases import KeyEvent, MouseEvent
from shapely.geometry import LineString, MultiLineString, Point

from ..env.env_2d import Env2D
from .colors import CmdColors, PlotColors
from .plot import plot_env


def unpack_curve(curve: LineString) -> np.ndarray:
    """
    Unpacks a LineString object into a numpy array.

    Args:
        curve (LineString): input curve

    Returns:
        np.ndarray: numpy array of curve points

    Raises:
        ValueError: if the input type is not LineString
    """
    if not isinstance(curve, LineString):
        raise ValueError("Input curve must be a LineString")
    points = np.array(curve.coords)
    return points


def measure_length(curve: LineString | np.ndarray) -> float:
    """
    Measures the length of a curve.
    Args:
        curve (LineString | np.ndarray): input curve

    Returns:
        float: length of the curve

    Raises:
        TypeError: if the input type is not LineString or np.ndarray
    """
    if isinstance(curve, LineString):
        return curve.length
    elif isinstance(curve, np.ndarray):
        return np.sum(np.linalg.norm(np.diff(curve, axis=0), axis=1))
    else:
        raise TypeError("Input curve must be a LineString or a numpy array")


def resample_curve(
    curve: LineString | np.ndarray, n: int, resampling_type: str
) -> LineString | np.ndarray:
    """
    Resamples a curve by adding a number of intermediate points between each pair of
    adjacent points in the original curve.

    Args:
        curve (LineString | np.ndarray): input curve
        n (int): number of intermediate points to add between each pair of adjacent
            points in the curve
        resampling_type (str): type of resolution enhancement. Available options:
            - "global": total number of points in the curve is set to `n`. The points
            are distributed uniformly along the curve. Note that the total number of
            points in the resampled curve may not be exactly `n` due to rounding errors.
            - "segment": `n` points are added between each pair of adjacent points in
            the curve, i.e., given p initial segments the final point count is `n*p`.

    Returns:
        (LineString | np.ndarray): resampled curve. Output type matches the input one.

    Raises:
        TypeError: if n is not an int
        ValueError: if resampling_type is not one of the implemented types
    """

    # parse inputs
    if not isinstance(n, int):
        raise TypeError(f"Invalid type for n. Expected int, got {type(n)}.")
    if not resampling_type in ["global", "segment"]:
        raise ValueError("Invalid resampling_type. Expected 'global' or 'segment'.")

    # unpack curve
    points: np.ndarray
    if isinstance(curve, LineString):
        points = unpack_curve(curve)
    elif isinstance(curve, np.ndarray):
        points = curve
    else:
        raise TypeError(
            f"Invalid type for curve. "
            f"Expected LineString or np.ndarray, got {type(curve)}."
        )

    # resample curve
    if resampling_type == "global":

        points_stack = []  # initialize empty stack of new points
        len_tot = measure_length(curve)  # total length of the curve

        for i in range(0, len(points) - 1):

            # compute number of points to add between each pair of adjacent points.
            # Round up to avoid missing points.
            len_seg = measure_length(LineString([points[i], points[i + 1]]))
            res_seg = int(np.ceil(n * len_seg / len_tot))

            # interpolation coefficients for current segment
            a = np.linspace(0, 1, res_seg, endpoint=False)
            a = a[:, np.newaxis]

            # add points between the current pair of adjacent points
            points_seg = points[i] + a * (points[i + 1] - points[i])
            points_stack.append(points_seg)

        # merge all arrays
        points_stack.append(points[-1])  # add last point
        new_points = np.vstack(points_stack)

    elif resampling_type == "segment":

        # interpolation coefficients
        a = np.linspace(0, 1, n, endpoint=False)
        a = a[:, np.newaxis, np.newaxis]

        # compute new points
        start_points = points[:-1]
        end_points = points[1:]
        differences = end_points - start_points
        new_points = start_points + a * differences
        new_points = new_points.reshape((-1, 2), order="F")
        np.append(new_points, points[-1])  # add last point

    # create new curve
    if isinstance(curve, LineString):
        return LineString(new_points)
    else:
        return new_points


def straight_segment(
    p1: tuple[float] | list[float] | np.ndarray | Point,
    p2: tuple[float] | list[float] | np.ndarray | Point,
    n: int = 10,
    output_type: str = "LineString",
) -> LineString:
    """
    Computes a straight line segment between two points.

    Args:
        p1 (tuple[float] | list[float] | np.ndarray | Point): First point.
        p2 (tuple[float] | list[float] | np.ndarray | Point): Second point.
        n (int, optional): Number of points in the segment (default 10).
        output_type (str, optional): Output type (default "LineString").

    Returns:
        (LineString | np.ndarray) object representing the straight segment.

    Raises:
        ValueError: If output_type is not one of the expected values.
    """
    # extract coordinates from Points (depending on input types)
    if isinstance(p1, Point):
        p1 = p1.coords[0]
    if isinstance(p2, Point):
        p2 = p2.coords[0]

    # compute straight line segment
    if output_type in ["LineString", "linestring"]:
        return LineString(
            np.column_stack(
                [np.linspace(p1[0], p2[0], n), np.linspace(p1[1], p2[1], n)]
            )
        )
    elif output_type in ["array", "numpy", "ndarray"]:
        return np.column_stack(
            [np.linspace(p1[0], p2[0], n), np.linspace(p1[1], p2[1], n)]
        )
    else:
        raise ValueError(f"Invalid output_type: {output_type}")


def shorten_curve(
    curve: LineString | np.ndarray,
    env: Env2D,
    iterations: int = 1,
) -> LineString | np.ndarray:
    """
    Approximate shortening of a piecewise linear curve.

    Args:
        curve (LineString | np.ndarray): object representing the curve.
        env (Env2D): Env object containing the obstacles.
        iterations (int, optional): number of times the shortening algorithm must be run
            in sequence (default: 1).

    Returns:
        shortened_curve (LineString | np.ndarray): object representing the shortened
            curve. Output type matches the input one.

    Raises:
        TypeError: If the input types are not as expected.
    """
    # check input types
    if not isinstance(iterations, int):
        raise TypeError(f"Expected int for iterations, got {type(iterations)} instead.")

    # get points of the curve
    if isinstance(curve, LineString):
        points = np.array(curve.coords)
    else:
        points = np.array(curve)

    # run the shortening algorithm for the specified number of times
    for _ in range(iterations):

        # initialize algorithm
        n = points.shape[0] - 1  # number of points in the curve (starting from 0)
        shortened_curve = np.array([points[0]])

        # run shortening algorithm
        i = 0
        j = 0
        while j <= n:
            # TODO: it would be nice to distinguish the case of one endpoint lying on
            # the boundary of an obstacle vs the whole edge being on the boundary of an
            # obstacle. The former case would improve the quality of the solution, even
            # if not by much.
            if j < n and env.is_valid_edge(
                points[i], points[j + 1], allow_boundary_overlap=False
            ):
                j += 1
            elif i != j:
                shortened_curve = np.append(
                    shortened_curve,
                    np.array([points[j]]),
                    axis=0,
                )
                i = j

                # termination condition
                if j == n:
                    break

            else:
                # This case prevents the shortening algorithm to get stuck in an
                # infinite loop in case part of the curve lies on the boundary of an
                # obstacle when allow_boundary_overlap is set to False. In this case,
                # the point is added and the shortening process moves to the next one.
                shortened_curve = np.append(
                    shortened_curve,
                    np.array([points[j]]),
                    axis=0,
                )
                i += 1  # force advance of both points
                j += 1

        # update points with the shortened curve before the next iteration
        points = shortened_curve

    # return the shortened curve as a LineString object
    if isinstance(curve, LineString):
        return LineString(shortened_curve)
    else:
        return shortened_curve


def compute_signature(
    curve: LineString | np.ndarray,
    env: Env2D,
    simplify: bool = True,
    perturbation: float = 1e-3,
) -> list[int]:
    """
    Compute the signature of a curve given a set of generators in the environment.

    Args:
        curve (LineString | np.ndarray): object representing the curve.
        env (Env2D): Env object containing the generators.
        simplify (bool, optional): Simplify the signature (default True).
        perturbation (float, optional): Perturbation in x for curve points lying on
            generators (default 1e-3).

    Returns:
        signature (list[int]): Curve signature represented as a list of signed integers.

    Raises:
        TypeError: If the input types are not as expected.
    """
    # check input types
    if not isinstance(simplify, bool):
        raise TypeError(f"Expected bool for simplify, got {type(simplify)} instead.")
    if not isinstance(perturbation, (int, float)):
        raise TypeError(
            f"Expected int or float for perturbation, got {type(perturbation)} "
            "instead."
        )

    # get points from input
    if isinstance(curve, LineString):
        points = list(curve.coords)
    elif isinstance(curve, np.ndarray):
        points = list(curve)
    else:
        raise TypeError(
            "Expected LineString or np.ndarray for curve, "
            f"got {type(curve)} instead."
        )

    # perturb points lying on generators
    for point_idx, point in enumerate(points):
        point = Point(point)
        if point.intersects(env.generators):
            points[point_idx] = (point.x - perturbation, point.y)

    # compute signature
    signature = []  # initialize empty signature
    for point1, point2 in zip(points[:-1], points[1:]):

        # compute segment and direction
        segment = LineString([point1, point2])
        if point1[0] <= point2[0]:
            direction = 1  # left to right
        else:
            direction = -1  # right to left

        # find crossings of segment with generators
        crossings = []
        if isinstance(env.generators, LineString):
            if segment.intersects(env.generators):
                crossings.append(
                    [1, list(segment.intersection(env.generators).coords[0])]
                )
        elif isinstance(env.generators, MultiLineString):
            for gen_idx, gen in enumerate(env.generators.geoms):
                if segment.intersects(gen):
                    crossings.append(
                        [gen_idx + 1, list(segment.intersection(gen).coords[0])]
                    )
        else:
            raise TypeError(
                "Expected LineString or MultiLineString for env.generators, "
                f"got {type(env.generators)} instead."
            )

        # sort crossings by x coordinate
        if direction == 1:
            crossings.sort(key=lambda x: x[1][0])
        else:
            crossings.sort(key=lambda x: x[1][0], reverse=True)

        # add crossings to signature
        for crossing in crossings:
            signature.append(crossing[0] * direction)

    # simplify the signature
    if simplify:
        signature = simplify_signature(signature)

    return signature


def simplify_signature(sig: list[int]) -> list[int]:
    """
    Simplify the signature by removing pairs of adjacent oppostite elements from the
    list.

    Args:
        sig (list[int]): Signature as list of signed integers.

    Returns:
        list[int]: Simplified signature.
    """
    simp_sig = []  # initialize empty stack
    for char in sig:
        if simp_sig and simp_sig[-1] == -char:
            simp_sig.pop()  # remove last element of the stack
        else:
            simp_sig.append(char)  # add the element if no match
    return simp_sig


def compare_signatures(sig1: list[int], sig2: list[int]) -> bool:
    """
    Compare two signatures for equality.

    Args:
        sig1 (list[int]): First signature.
        sig2 (list[int]): Second signature.

    Return:
        bool: True if the signatures are equal, False otherwise.
    """
    return sig1 == sig2


def generate_curve(
    env: Env2D,
    **kwargs,
) -> LineString | np.ndarray:
    """
    Manually generate a curve in a given environment. The curve points are interactively
    selected by the user. The curve is represented as a LineString object.

    Relevant references:
    - https://matplotlib.org/3.1.1/api/_as_gen/matplotlib.pyplot.ginput.html
    - https://matplotlib.org/stable/users/explain/figure/event_handling.html
    - https://matplotlib.org/stable/api/backend_bases_api.html

    Args:
        env (Env2D): Env object containing the environment
        settings (Settings): Settings object containing the plot settings
        output_type (str, **kwargs): Type of output to generate.
        init_point (**kwargs): Initial point for the curve generation (default None).
        init_curve (**kwargs): Initial curve to which the new curve is appended
            (default None). If set to different from None, init_point is set to be the
            terminal point of init_curve.
        check_obs (**kwargs): Check collisions of the generated curve with the
            obstacles present in the environment (default True).
        check_self_intersection (**kwargs): Check self-intersections of the curve,
            preventing the creation of loops (default False).
        other_curves (**kwargs): List of other curves to display (default None).
        check_other_curves (**kwargs): Check intersections of the curve with the
            other curves, preventing intersections (default False).
        label_other_curves (**kwargs): Label the other curves in the plot (default
            True).
        max_points (**kwargs): Maximum number of points in the curve (default 20).
        title (**kwargs): Title of the plot (default "Interactive curve generation").
        **kwargs (**kwargs): The function accepts all kwargs of the plot_env function.
            These kwargs are passed directly to the plot_env function.

    Returns:
        LineString | np.ndarray: LineString object representing the curve.

    Raises:
        TypeError: If the input types are not as expected.
    """

    ## KWARGS ##
    # kwargs default values
    output_type: str = "linestring"  # type of output to generate
    init_point: Point | np.ndarray[float] | list[float] = None  # initial point
    init_curve: np.ndarray | LineString = None  # curve to start generation from
    check_obs: bool = True  # prevent collisions of the curve with the obstacles
    check_self_intersection: bool = False  # prevent self-intersections of the curve
    other_curves: list[np.ndarray | LineString] = []  # list of other curves to display
    check_other_curves: bool = False  # prevent intersections with other curves
    label_other_curves: bool = True  # label the other curves in the plot
    max_points: int = 20  # maximum number of points in the curve

    # Parse kwargs
    # Iteration is done over list(keys) to allow deletion of keys during iteration
    for key in list(kwargs.keys()):
        if key == "output_type":
            if not isinstance(kwargs[key], str):
                raise TypeError(
                    f"Expected str for output_type, got {type(kwargs[key])} instead."
                )
            if kwargs[key] in ["LineString", "linestring"]:
                output_type = "linestring"
            elif kwargs[key] in ["array", "numpy", "ndarray"]:
                output_type = "array"
            else:
                raise ValueError(f"Invalid value for output_type: {kwargs[key]}")
            del kwargs["output_type"]
        if key == "init_point":
            if not isinstance(kwargs[key], (Point, np.ndarray, list)):
                raise TypeError(
                    "Expected Point, np.ndarray, or list for init_point, got "
                    f"{type(kwargs[key])} instead."
                )
            init_point = kwargs[key]
            del kwargs["init_point"]
        elif key == "init_curve":
            if not isinstance(kwargs[key], LineString):
                raise TypeError(
                    "Expected LineString for init_curve, "
                    f"got {type(kwargs[key])} instead."
                )
            init_curve = kwargs[key]
            del kwargs["init_curve"]
        elif key == "check_obs":
            if not isinstance(kwargs[key], bool):
                raise TypeError(
                    f"Expected bool for check_obs, got {type(kwargs[key])} instead."
                )
            check_obs = kwargs[key]
            del kwargs["check_obs"]
        elif key == "check_self_intersection":
            if not isinstance(kwargs[key], bool):
                raise TypeError(
                    "Expected bool for check_self_intersection, "
                    f"got {type(kwargs[key])} instead."
                )
            check_self_intersection = kwargs[key]
            del kwargs["check_self_intersection"]
        elif key == "other_curves":
            if not isinstance(kwargs[key], list):
                raise TypeError(
                    f"Expected list for other_curves, got {type(kwargs[key])} instead."
                )
            elif not all(isinstance(curve, LineString) for curve in kwargs[key]):
                # note: skipped in case of empty list
                raise TypeError("Expected list of LineString objects for other_curves.")
            other_curves = kwargs[key]
            del kwargs["other_curves"]
        elif key == "check_other_curves":
            if not isinstance(kwargs[key], bool):
                raise TypeError(
                    f"Expected bool for check_other_curves, got {type(kwargs[key])} "
                    "instead."
                )
            check_other_curves = kwargs[key]
            del kwargs["check_other_curves"]
        elif key == "label_other_curves":
            if not isinstance(kwargs[key], bool):
                raise TypeError(
                    f"Expected bool for label_other_curves, "
                    f"got {type(kwargs[key])} instead."
                )
            label_other_curves = kwargs[key]
            del kwargs["label_other_curves"]
        elif key == "max_points":
            if not isinstance(kwargs[key], int):
                raise TypeError(
                    f"Expected int for max_points, got {type(kwargs[key])} instead."
                )
            max_points = kwargs[key]
            del kwargs["max_points"]
        else:
            pass  # leave other kwargs for plot_env (ValueError will be raised there)

    # Set title and legend visibility (default values)
    if "title" not in kwargs:
        kwargs["title"] = "Interactive curve generation (`esc' to terminate)."
    if "show_legend" not in kwargs:
        kwargs["show_legend"] = True

    ## EVENT MANAGEMENT ##
    # event management for keyboard input
    def on_press(event: KeyEvent):
        if event.key == "escape":
            plt.close()

    # event management for mouse click
    def on_click(event: MouseEvent):

        # invalid click (e.g., out of axis area)
        if not event.xdata or not event.ydata:
            return

        # middle click -> terminate curve generation (also achieved by closing the plot)
        if event.button == 2:
            plt.close()
            return

        # right click -> remove last point from the curve
        elif event.button == 3:

            # remove initial point
            if len(points) == 1:
                points.pop()
                polyline.set_data([], [])
                temp_line.set_data([], [])

            # remove generic point
            elif len(points) > 1:
                points.pop()
                last_point = points[-1]
                mouse_point = (event.xdata, event.ydata)
                polyline.set_data(*zip(*points))
                temp_line.set_data(
                    [last_point[0], mouse_point[0]], [last_point[1], mouse_point[1]]
                )

        # left click -> add point to the curve (init point variant)
        elif event.button == 1 and len(points) == 0:
            x, y = event.xdata, event.ydata
            if check_obs and not env.is_valid_point(x, y):
                print(
                    f"{CmdColors.WARNING}[curve_fcns]{CmdColors.ENDC}: the point "
                    f"({x:.2f}, {y:.2f}) is in collision with the obstacle region and "
                    "cannot be added to the curve."
                )
                return
            else:
                points.append([x, y])

        # left click -> add point to the curve (generic point variant)
        elif event.button == 1 and len(points) < max_points:
            x, y = event.xdata, event.ydata
            temp_points = points + [[x, y]]
            temp_curve = LineString(temp_points)

            # check if the point is in collision with the obstacles
            if check_obs:
                if not env.is_valid_edge(points[-1], [x, y]):
                    print(
                        f"{CmdColors.WARNING}[curve_fcns]{CmdColors.ENDC}: the point "
                        f"({x:.2f}, {y:.2f}) results in a collision with the obstacle "
                        "region and will not be added to the curve."
                    )
                    return

            # check if the new point results in a curve with self-intersections
            if check_self_intersection:
                if len(temp_points) > 1 and not temp_curve.is_simple:
                    print(
                        f"{CmdColors.WARNING}[curve_fcns]{CmdColors.ENDC}: the point "
                        f"({x:.2f}, {y:.2f}) results in a self-intersection of the new "
                        "segment with the rest of the curve, and will not be added."
                    )
                    return

            # check if the new point results in intersections with other curves (if any)
            if check_other_curves:
                for idx, curve in enumerate(other_curves):
                    if temp_curve.intersects(curve):
                        print(
                            f"{CmdColors.WARNING}[curve_fcns]{CmdColors.ENDC}: the  "
                            f"point ({x:.2f}, {y:.2f}) results in an intersection "
                            f"with other curve {idx} and will not be added."
                        )
                        return

            # add point to the curve
            points.append([x, y])
            polyline.set_data(*zip(*points))

            # check if the curve has reached the maximum number of points
            if len(points) == max_points:
                print(
                    f"{CmdColors.WARNING}[curve_fcns]{CmdColors.ENDC}: the maximum "
                    f"number of points in the curve ({max_points}) has been reached. "
                    "No more points can be added."
                )
                # fig.canvas.mpl_disconnect(click_event_id)  # disconnect event
                return

        # update plot
        fig.canvas.draw_idle()

    # event management for mouse movement
    def on_motion(event: MouseEvent):
        if not points or not event.xdata or not event.ydata:
            return
        last_point = points[-1]
        mouse_point = (event.xdata, event.ydata)
        temp_line.set_data(
            [last_point[0], mouse_point[0]], [last_point[1], mouse_point[1]]
        )
        fig.canvas.draw_idle()

    ## CURVE GENERATION ##
    # initialize curve
    if init_curve is not None:
        points = [list(point) for point in init_curve.coords]
        init_point = Point(points[0])
    elif init_point is not None:
        points = [init_point]
    else:
        points = []
    kwargs["show_anchor"] = init_point is not None  # show anchor if init_point is set

    # manage initial curve
    if init_curve:
        kwargs["tether"] = init_curve
        kwargs["show_tether"] = True
        kwargs["show_anchor"] = True
        kwargs["show_robot"] = True

    # manage other curves
    if other_curves:
        kwargs["curves"] = other_curves
        kwargs["show_curves_labels"] = label_other_curves
    else:
        kwargs["curves"] = []
        kwargs["show_curves_labels"] = False

    # plot environment
    fig, ax = plot_env(
        env,
        **kwargs,
    )

    # Initialize interactive curve
    (polyline,) = ax.plot(
        [],
        [],
        color=PlotColors.tether_color,
        linewidth=1.5,
        zorder=7,
    )
    (temp_line,) = ax.plot(
        [],
        [],
        color=PlotColors.tether_color,
        linewidth=1.5,
        zorder=7,
    )

    # enable interactive mode and generate curve
    fig.canvas.mpl_connect("motion_notify_event", on_motion)
    fig.canvas.mpl_connect("button_press_event", on_click)
    fig.canvas.mpl_connect("key_press_event", on_press)
    plt.show()

    # return the curve as a LineString object
    if len(points) == 0:
        print(
            f"{CmdColors.WARNING}[curve_fcns]{CmdColors.ENDC}: the curve has no points "
            "and cannot be generated."
        )
        return None
    elif len(points) == 1:
        print(
            f"{CmdColors.WARNING}[curve_fcns]{CmdColors.ENDC}: the curve has only one "
            "point. A point will be returned instead of a curve."
        )
        if output_type == "linestring":
            return Point(points[0])
        else:
            return np.array(points)
    else:
        if output_type == "linestring":
            return LineString(points)
        else:
            return np.array(points)


def find_intersection(
    p1: np.ndarray, p2: np.ndarray, p3: np.ndarray, p4: np.ndarray
) -> np.ndarray | None:
    """
    Find the intersection point of two line segments that are guaranteed to intersect in
    one and only one point.
    Segment 1: p1 -> p2
    Segment 2: p3 -> p4

    Args:
        p1 (np.ndarray): Starting point of segment 1.
        p2 (np.ndarray): Ending point of segment 1.
        p3 (np.ndarray): Starting point of segment 2.
        p4 (np.ndarray): Ending point of segment 2.

    Returns:
        np.ndarray | None: Intersection point if it exists, None otherwise.
    """
    # Unpack points
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    x4, y4 = p4

    # Compute denominator
    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)

    # Parallel or collinear
    if denom == 0:
        return None

    t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
    u = ((x1 - x3) * (y1 - y2) - (y1 - y3) * (x1 - x2)) / denom

    # Intersection is within both segments only if t and u are both in [0, 1]
    if 0 <= t <= 1 and 0 <= u <= 1:
        return np.array([x1 + t * (x2 - x1), y1 + t * (y2 - y1)])

    return None
