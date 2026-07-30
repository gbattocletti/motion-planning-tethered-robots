"""
Homotopy-class-constrained trajectory generation.

Takes a path in the lifted (h-augmented) dual graph, builds the corridor from the
triangles it visits, and solves for a dynamically feasible robot trajectory
confined to that corridor. The homotopy class is enforced purely by the corridor:
since the triangles triangulate free space, obstacle avoidance comes for free.

Control modes (TrajParams.control_mode):
    'force'     input = force [N]; double integrator, exact ZOH:
                    p+ = p + dt v + dt^2/(2 m) f,   v+ = v + dt/m f
                bounds: ||v|| <= max_speed, |f| <= m max_acceleration,
                        |f+ - f| <= m max_force_slew dt
    'position'  input = displacement [m]; p+ = p + d
                bounds: ||d|| <= max_speed dt, |d+ - d| <= max_acceleration dt^2,
                with virtual rest displacements before the first and after the
                last step, so the motion starts and ends at rest.

Both modes return absolute positions; feed those (interpolated to the FEM step)
to TetherFEM2D with input_mode='position', or the forces to input_mode='force'.
"""

from __future__ import annotations

from dataclasses import dataclass

import casadi as ca
import numpy as np


@dataclass
class TrajParams:
    n_steps: int = 100  # number of control intervals
    dt: float = 0.25  # duration of one control interval [s]
    control_mode: str = "position"  # {'position', 'force'}
    robot_mass: float = 1.0  # [kg], force mode only
    max_speed: float = 1.2  # [m/s]
    max_acceleration: float = 1.5  # [m/s^2]
    max_force_slew: float = 8.0  # [N/s], force mode only
    obstacle_clearance: float = 0.05  # corridor shrink at obstacle walls [m]
    weight_tracking: float = 1.0  # stay close to the geodesic reference
    weight_input: float = 0.2  # input effort
    weight_smoothness: float = 1.0  # penalize input changes


def find_corridor(
    triangulation,
    alpha_lift: list[int],
    start: np.ndarray,
    goal: np.ndarray,
    obstacle_clearance: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Corridor for one homotopy class, from an ordered path in the lifted dual graph.

    Args:
        triangulation: Triangulation with the lifted complex already built.
        alpha_lift: lifted dual vertex indices, ordered start -> goal (e.g. the
            output of graph_search.a_star_search / dfs / bfs).
        start, goal: endpoints, lying in the first and last triangle of alpha_lift.
        obstacle_clearance: inward shrink applied at obstacle walls [m].

    Returns:
        corridor_triangles (n_triangles, 3, 2): triangle vertices, in traversal
            order.
        geodesic (n_points, 2): shortest path of the class, used as the tracking
            reference.
        edge_normals (n_triangles, 3, 2) and edge_offsets (n_triangles, 3):
            outward half-spaces, so that a point lies in triangle t iff
                edge_normals[t] @ point <= edge_offsets[t].

    The clearance is applied only to boundary edges of the triangulation (those
    belonging to a single triangle, i.e. obstacle or workspace walls). Shrinking
    interior edges instead would carve a gap between consecutive corridor
    triangles and disconnect the corridor, which easily makes the problem
    infeasible around sharp turns.
    """
    base_triangles = [triangulation.vertices_dual_lift[node][0] for node in alpha_lift]
    corridor_vertex_ids = triangulation.triangles[base_triangles]  # (n_triangles, 3)
    corridor_triangles = triangulation.vertices[corridor_vertex_ids]
    geodesic = np.asarray(
        triangulation.homotopic_shortest_path(
            alpha=base_triangles, p_init=start, p_end=goal
        )
    )

    # One half-space per edge, oriented outward: the vertex opposite to an edge
    # must satisfy that edge's inequality.
    edge_start = corridor_triangles
    edge_end = np.roll(corridor_triangles, -1, axis=1)
    opposite_vertex = np.roll(corridor_triangles, -2, axis=1)
    edge_vector = edge_end - edge_start
    edge_normals = np.stack([edge_vector[..., 1], -edge_vector[..., 0]], axis=-1)
    normal_points_inward = (
        np.sum(edge_normals * (opposite_vertex - edge_start), axis=-1) > 0.0
    )
    edge_normals = np.where(
        normal_points_inward[..., None], -edge_normals, edge_normals
    )
    edge_offsets = np.sum(edge_normals * edge_start, axis=-1)

    # A boundary edge of the triangulated free space belongs to exactly one
    # triangle; every other edge is interior and must not be shrunk.
    all_edges = np.sort(
        np.concatenate(
            [
                triangulation.triangles[:, [0, 1]],
                triangulation.triangles[:, [1, 2]],
                triangulation.triangles[:, [2, 0]],
            ]
        ),
        axis=1,
    )
    unique_edges, times_used = np.unique(all_edges, axis=0, return_counts=True)
    boundary_edges = {tuple(edge) for edge in unique_edges[times_used == 1]}
    corridor_edges = np.sort(
        np.stack([corridor_vertex_ids, np.roll(corridor_vertex_ids, -1, axis=1)], -1),
        axis=-1,
    )
    edge_is_boundary = np.array(
        [
            [
                tuple(corridor_edges[triangle, edge]) in boundary_edges
                for edge in range(3)
            ]
            for triangle in range(len(corridor_vertex_ids))
        ]
    )
    edge_offsets -= (
        obstacle_clearance * np.linalg.norm(edge_normals, axis=-1) * edge_is_boundary
    )
    return corridor_triangles, geodesic, edge_normals, edge_offsets


def assign_triangles(
    edge_normals: np.ndarray, edge_offsets: np.ndarray, points: np.ndarray
) -> np.ndarray:
    """
    For every point, the index (within the corridor traversal order) of a triangle
    containing it, never moving backwards along the corridor: a class that winds
    around an obstacle visits the same base triangle more than once, so only the
    traversal order is meaningful. A point inside no triangle is given the
    least-violated triangle at or ahead of the current position.
    """
    triangle_of_point = np.zeros(len(points), dtype=int)
    current_triangle = 0
    for point_index, point in enumerate(points):
        violation = np.max(edge_normals @ point - edge_offsets, axis=1)
        violation = violation[current_triangle:]
        current_triangle += int(
            np.argmax(violation <= 1e-9)
            if violation.min() <= 1e-9
            else np.argmin(violation)
        )
        triangle_of_point[point_index] = current_triangle
    return triangle_of_point


def solve_nlp(
    edge_normals: np.ndarray,
    edge_offsets: np.ndarray,
    geodesic: np.ndarray,
    start: np.ndarray,
    goal: np.ndarray,
    params: TrajParams,
    max_outer_iterations: int = 10,
    verbose: bool = True,
) -> dict:
    """
    Path -> dynamically feasible trajectory inside the corridor (CasADi + IPOPT).

    The knot -> triangle assignment is held fixed within each solve, which keeps
    the problem a smooth NLP; the outer loop re-locates the solution in the
    corridor and re-solves until the assignment stops changing (typically one or
    two passes). The homotopy class already fixes the triangle sequence, so only
    the schedule is at stake here.

    Returns a dict with 'positions' (n_steps+1, 2), 'velocities' (n_steps+1, 2),
    'inputs' (n_steps, 2), 'triangle_of_knot' (n_steps+1,) and 'cost'.
    """
    n_steps, dt, mass = params.n_steps, params.dt, params.robot_mass

    # reference resampled uniformly in arc length, one point per knot
    segment_lengths = np.linalg.norm(np.diff(geodesic, axis=0), axis=1)
    arclength = np.concatenate([[0.0], np.cumsum(segment_lengths)])
    arclength_at_knots = np.linspace(0.0, arclength[-1], n_steps + 1)
    reference = np.stack(
        [
            np.interp(arclength_at_knots, arclength, geodesic[:, axis])
            for axis in range(2)
        ],
        axis=1,
    )

    triangle_of_knot = assign_triangles(edge_normals, edge_offsets, reference)
    initial_guess = reference
    solution: dict = {}

    for iteration in range(max_outer_iterations):
        opti = ca.Opti()
        positions = opti.variable(2, n_steps + 1)
        inputs = opti.variable(2, n_steps)

        if params.control_mode == "force":
            velocities = opti.variable(2, n_steps + 1)
            opti.subject_to(
                positions[:, 1:]
                == positions[:, :-1]
                + dt * velocities[:, :-1]
                + (0.5 * dt**2 / mass) * inputs
            )
            opti.subject_to(
                velocities[:, 1:] == velocities[:, :-1] + (dt / mass) * inputs
            )
            opti.subject_to(velocities[:, 0] == 0.0)
            opti.subject_to(velocities[:, n_steps] == 0.0)
            for knot in range(n_steps + 1):
                opti.subject_to(ca.sumsqr(velocities[:, knot]) <= params.max_speed**2)
            max_force = mass * params.max_acceleration
            opti.subject_to(opti.bounded(-max_force, ca.vec(inputs), max_force))
            max_force_step = mass * params.max_force_slew * dt
            opti.subject_to(
                opti.bounded(
                    -max_force_step,
                    ca.vec(inputs[:, 1:] - inputs[:, :-1]),
                    max_force_step,
                )
            )
        else:
            opti.subject_to(positions[:, 1:] == positions[:, :-1] + inputs)
            max_step = params.max_speed * dt
            for knot in range(n_steps):
                opti.subject_to(ca.sumsqr(inputs[:, knot]) <= max_step**2)
            # second differences, with a virtual step of zero before the first and
            # after the last one, so the trajectory starts and ends at rest
            step_changes = ca.horzcat(
                inputs[:, 0],
                inputs[:, 1:] - inputs[:, :-1],
                -inputs[:, n_steps - 1],
            )
            max_step_change = params.max_acceleration * dt**2
            opti.subject_to(
                opti.bounded(-max_step_change, ca.vec(step_changes), max_step_change)
            )

        opti.subject_to(positions[:, 0] == ca.DM(start))
        opti.subject_to(positions[:, n_steps] == ca.DM(goal))
        # endpoints are pinned and may sit exactly on the corridor boundary
        for knot in range(1, n_steps):
            triangle = int(triangle_of_knot[knot])
            opti.subject_to(
                ca.DM(edge_normals[triangle]) @ positions[:, knot]
                <= ca.DM(edge_offsets[triangle])
            )

        opti.minimize(
            params.weight_tracking * ca.sumsqr(positions - ca.DM(reference.T))
            + params.weight_input * ca.sumsqr(inputs)
            + params.weight_smoothness * ca.sumsqr(inputs[:, 1:] - inputs[:, :-1])
        )
        opti.set_initial(positions, initial_guess.T)
        opti.solver("ipopt", {"ipopt.print_level": 0, "print_time": 0})
        opti_solution = opti.solve()

        optimal_positions = np.array(opti_solution.value(positions)).T
        optimal_inputs = np.array(opti_solution.value(inputs)).T
        optimal_velocities = (
            np.array(opti_solution.value(velocities)).T
            if params.control_mode == "force"
            else np.vstack([np.zeros((1, 2)), optimal_inputs / dt])
        )
        solution = dict(
            positions=optimal_positions,
            velocities=optimal_velocities,
            inputs=optimal_inputs,
            triangle_of_knot=triangle_of_knot,
            cost=float(opti_solution.value(opti.f)),
        )

        updated_assignment = assign_triangles(
            edge_normals, edge_offsets, optimal_positions
        )
        updated_assignment[0] = triangle_of_knot[0]
        updated_assignment[-1] = triangle_of_knot[-1]
        n_reassigned = int(np.sum(updated_assignment != triangle_of_knot))
        if verbose:
            print(
                f"[solve_nlp:{params.control_mode}] outer {iteration}: "
                f"cost={solution['cost']:.4f}, reassigned {n_reassigned} knots"
            )
        if n_reassigned == 0:
            break
        triangle_of_knot = updated_assignment
        initial_guess = optimal_positions
    return solution


def solve_minlp(
    edge_normals: np.ndarray,
    edge_offsets: np.ndarray,
    geodesic: np.ndarray,
    start: np.ndarray,
    goal: np.ndarray,
    params: TrajParams,
    solver: str = "bonmin",
) -> dict:
    """
    Mixed-integer variant: the solver picks the knot -> triangle assignment.

        knot_in_triangle[t, k] = 1 iff knot k lies in triangle t
        edge_normals[t] p_k <= edge_offsets[t]
                               + big_m[t] (1 - knot_in_triangle[t, k])
        sum_t knot_in_triangle[t, k] = 1
        progress_k = sum_t t knot_in_triangle[t, k], with
            0 <= progress_{k+1} - progress_k <= 1,
            progress_0 = 0, progress_last = n_triangles - 1   (monotone traversal)

    Same dynamics, cost and geometry as solve_nlp(); an MIQP in both control
    modes. Needs a mixed-integer solver through CasADi (bonmin, or gurobi/cplex).
    Only worth it when the triangle schedule itself matters, e.g. very uneven
    triangle sizes that defeat the arc-length assignment used by solve_nlp().
    """
    n_steps, dt, mass = params.n_steps, params.dt, params.robot_mass
    n_triangles = len(edge_normals)

    segment_lengths = np.linalg.norm(np.diff(geodesic, axis=0), axis=1)
    arclength = np.concatenate([[0.0], np.cumsum(segment_lengths)])
    reference = np.stack(
        [
            np.interp(
                np.linspace(0.0, arclength[-1], n_steps + 1),
                arclength,
                geodesic[:, axis],
            )
            for axis in range(2)
        ],
        axis=1,
    )

    # big-M: the largest violation any half-space can take over a box enclosing
    # the corridor, so an unselected triangle imposes no constraint
    box_min = reference.min(axis=0) - 10.0
    box_max = reference.max(axis=0) + 10.0
    box_corners = np.array(
        [
            [box_min[0], box_min[1]],
            [box_min[0], box_max[1]],
            [box_max[0], box_min[1]],
            [box_max[0], box_max[1]],
        ]
    )
    big_m = np.max(edge_normals @ box_corners.T, axis=2) - edge_offsets + 1e-3

    opti = ca.Opti()
    positions = opti.variable(2, n_steps + 1)
    inputs = opti.variable(2, n_steps)
    knot_in_triangle = opti.variable(n_triangles, n_steps + 1)
    opti.subject_to(opti.bounded(0.0, ca.vec(knot_in_triangle), 1.0))

    if params.control_mode == "force":
        velocities = opti.variable(2, n_steps + 1)
        opti.subject_to(
            positions[:, 1:]
            == positions[:, :-1]
            + dt * velocities[:, :-1]
            + (0.5 * dt**2 / mass) * inputs
        )
        opti.subject_to(velocities[:, 1:] == velocities[:, :-1] + (dt / mass) * inputs)
        opti.subject_to(velocities[:, 0] == 0.0)
        opti.subject_to(velocities[:, n_steps] == 0.0)
        for knot in range(n_steps + 1):
            opti.subject_to(ca.sumsqr(velocities[:, knot]) <= params.max_speed**2)
        max_force = mass * params.max_acceleration
        opti.subject_to(opti.bounded(-max_force, ca.vec(inputs), max_force))
        max_force_step = mass * params.max_force_slew * dt
        opti.subject_to(
            opti.bounded(
                -max_force_step,
                ca.vec(inputs[:, 1:] - inputs[:, :-1]),
                max_force_step,
            )
        )
    else:
        opti.subject_to(positions[:, 1:] == positions[:, :-1] + inputs)
        max_step = params.max_speed * dt
        for knot in range(n_steps):
            opti.subject_to(ca.sumsqr(inputs[:, knot]) <= max_step**2)
        step_changes = ca.horzcat(
            inputs[:, 0], inputs[:, 1:] - inputs[:, :-1], -inputs[:, n_steps - 1]
        )
        max_step_change = params.max_acceleration * dt**2
        opti.subject_to(
            opti.bounded(-max_step_change, ca.vec(step_changes), max_step_change)
        )

    opti.subject_to(positions[:, 0] == ca.DM(start))
    opti.subject_to(positions[:, n_steps] == ca.DM(goal))
    for knot in range(1, n_steps):
        for triangle in range(n_triangles):
            opti.subject_to(
                ca.DM(edge_normals[triangle]) @ positions[:, knot]
                <= ca.DM(edge_offsets[triangle])
                + ca.DM(big_m[triangle]) * (1.0 - knot_in_triangle[triangle, knot])
            )
    for knot in range(n_steps + 1):
        opti.subject_to(ca.sum1(knot_in_triangle[:, knot]) == 1.0)

    progress = ca.DM(np.arange(n_triangles)).T @ knot_in_triangle
    opti.subject_to(opti.bounded(0.0, ca.vec(progress[:, 1:] - progress[:, :-1]), 1.0))
    opti.subject_to(progress[:, 0] == 0.0)
    opti.subject_to(progress[:, n_steps] == n_triangles - 1)

    opti.minimize(
        params.weight_tracking * ca.sumsqr(positions - ca.DM(reference.T))
        + params.weight_input * ca.sumsqr(inputs)
        + params.weight_smoothness * ca.sumsqr(inputs[:, 1:] - inputs[:, :-1])
    )
    opti.set_initial(positions, reference.T)
    # integrality mask follows declaration order: positions, inputs,
    # knot_in_triangle, then velocities in force mode
    is_discrete = (
        [False] * (2 * (2 * n_steps + 1))
        + [True] * (n_triangles * (n_steps + 1))
        + [False] * (2 * (n_steps + 1) if params.control_mode == "force" else 0)
    )
    opti.solver(solver, {"discrete": is_discrete})
    opti_solution = opti.solve()

    optimal_positions = np.array(opti_solution.value(positions)).T
    optimal_inputs = np.array(opti_solution.value(inputs)).T
    optimal_velocities = (
        np.array(opti_solution.value(velocities)).T
        if params.control_mode == "force"
        else np.vstack([np.zeros((1, 2)), optimal_inputs / dt])
    )
    return dict(
        positions=optimal_positions,
        velocities=optimal_velocities,
        inputs=optimal_inputs,
        triangle_of_knot=np.argmax(
            np.array(opti_solution.value(knot_in_triangle)), axis=0
        ),
        cost=float(opti_solution.value(opti.f)),
    )
