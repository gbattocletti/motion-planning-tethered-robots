"""
Homotopy-class-constrained trajectory generation.

Takes a path in the lifted (h-augmented) dual graph, builds the corridor from the
triangles it visits, and solves for a dynamically feasible robot trajectory
confined to that corridor. The homotopy class is enforced purely by the corridor:
since the triangles triangulate free space, obstacle avoidance comes for free.

Control modes (TrajParams.control_mode):
    'force'     input = force [N]; double integrator, exact ZOH:
                    p+ = p + dt v + dt^2/(2 m) f,   v+ = v + dt/m f
                bounds: |v| <= max_speed, |f| <= m max_acceleration,
                        |f+ - f| <= m max_force_slew dt
    'position'  input = displacement [m]; p+ = p + d
                bounds: |d| <= max_speed dt, |d+ - d| <= max_acceleration dt^2,
                with virtual rest displacements before the first and after the
                last step, so the motion starts and ends at rest.

All limits are per-axis (componentwise) box bounds, not Euclidean norms, which
keeps every constraint linear. A diagonal motion can therefore reach sqrt(2)
times max_speed and max_acceleration; scale them by 1/sqrt(2) if the limits are
meant to hold on the vector magnitude.

Both modes return absolute positions; feed those (interpolated to the FEM step)
to TetherFEM2D with input_mode='position', or the forces to input_mode='force'.
"""

from __future__ import annotations

from dataclasses import dataclass

import casadi as ca
import numpy as np

DIM = 2

# Gurobi, CPLEX and the QP solvers are CasADi *conic* plugins, not nlpsol ones,
# so they need an Opti stack created as Opti("conic"). IPOPT, bonmin and knitro
# go through nlpsol. Since every constraint here is linear and the cost is
# convex quadratic, both routes are valid: the problem is a QP (an MIQP once the
# binaries are added).
CONIC_PLUGINS = frozenset(
    {
        "gurobi",
        "cplex",
        "highs",
        "clarabel",
        "clp",
        "cbc",
        "osqp",
        "qpoases",
        "proxqp",
        "hpipm",
        "qrqp",
        "ipqp",
    }
)


@dataclass
class TrajParams:
    n_steps: int = 40  # number of control intervals
    dt: float = 0.25  # duration of one control interval [s]
    control_mode: str = "position"  # {'position', 'force'}
    robot_mass: float = 2.0  # [kg], force mode only
    max_speed: float = 1.2  # [m/s], per axis
    max_acceleration: float = 1.5  # [m/s^2], per axis
    max_force_slew: float = 8.0  # [N/s] per axis, force mode only
    obstacle_clearance: float = 0.05  # corridor shrink at obstacle walls [m]
    weight_tracking: float = 1.0  # stay close to the geodesic reference
    weight_input: float = 2.0  # input effort
    weight_smoothness: float = 1.0  # penalize input changes


def corridor(
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
            outward half-spaces with UNIT normals, so that a point lies in
            triangle t iff edge_normals[t] @ point <= edge_offsets[t], and
            edge_normals[t] @ point - edge_offsets[t] is the signed distance to
            each edge in metres.

    The clearance is applied only to boundary edges of the triangulation (those
    belonging to a single triangle, i.e. obstacle or workspace walls). Shrinking
    interior edges instead would carve a gap between consecutive corridor
    triangles and disconnect the corridor, which easily makes the problem
    infeasible around sharp turns.
    """
    start = np.asarray(start, dtype=float).reshape(DIM)
    goal = np.asarray(goal, dtype=float).reshape(DIM)
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
    # normalize: a rotated edge vector has the edge's length, so raw rows differ in
    # scale by the ratio of edge lengths in the triangulation (easily 100x around
    # slivers at obstacle corners), which is poor conditioning for the solver. With
    # unit normals every row is a signed distance in metres, so rows are
    # commensurate, the clearance below is exactly a distance, and the violations
    # compared in assign_triangles are comparable across triangles.
    edge_normals /= np.linalg.norm(edge_normals, axis=-1, keepdims=True)
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
    edge_offsets -= obstacle_clearance * edge_is_boundary
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
            np.argmax(violation <= 1e-6)
            if violation.min() <= 1e-6
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
    max_outer_iterations: int = 3,
    solver: str = "ipopt",
    solver_options: dict | None = None,
    verbose: bool = True,
) -> dict:
    """
    Path -> dynamically feasible trajectory inside the corridor (CasADi).

    The knot -> triangle assignment is held fixed within each solve, which keeps
    the problem a smooth NLP; the outer loop re-locates the solution in the
    corridor and re-solves until the assignment stops changing (typically one or
    two passes). The homotopy class already fixes the triangle sequence, so only
    the schedule is at stake here.

    With the assignment fixed every constraint is linear and the cost is convex
    quadratic, so each solve is a QP: a unique global optimum, no local minima.

    The corridor half-spaces enter as parameters, so the problem and its
    derivatives are built once and only re-valued between outer iterations.

    Args:
        solver: 'ipopt' (default) or any CasADi plugin. Because the problem is a
            QP, a dedicated QP solver in CONIC_PLUGINS ('gurobi', 'osqp',
            'qpoases', 'highs', ...) is usually much faster; those are dispatched
            through Opti("conic") automatically.
        solver_options: options for the chosen solver, merged over the defaults
            below. For IPOPT, to actually use the warm-started multipliers on the
            second and later outer iterations, pass
            {'warm_start_init_point': 'yes', 'warm_start_bound_push': 1e-8,
             'warm_start_mult_bound_push': 1e-8}.

    Returns a dict with 'positions' (n_steps+1, 2), 'velocities' (n_steps+1, 2),
    'inputs' (n_steps, 2), 'triangle_of_knot' (n_steps+1,) and 'cost'.
    """
    n_steps, dt, mass = params.n_steps, params.dt, params.robot_mass
    start = np.asarray(start, dtype=float).reshape(DIM)
    goal = np.asarray(goal, dtype=float).reshape(DIM)

    # reference resampled uniformly in arc length, one point per knot
    segment_lengths = np.linalg.norm(np.diff(geodesic, axis=0), axis=1)
    arclength = np.concatenate([[0.0], np.cumsum(segment_lengths)])
    arclength_at_knots = np.linspace(0.0, arclength[-1], n_steps + 1)
    reference = np.stack(
        [
            np.interp(arclength_at_knots, arclength, geodesic[:, axis])
            for axis in range(DIM)
        ],
        axis=1,
    )

    triangle_of_knot = assign_triangles(edge_normals, edge_offsets, reference)
    solution: dict = {}

    # Warm start: without this only the positions are initialized and IPOPT starts
    # the inputs (and velocities) at zero, which contradicts the position guess and
    # costs iterations. Differentiating the reference gives a guess consistent with
    # the dynamics.
    guess_positions = reference
    guess_velocities = np.vstack([np.zeros((1, DIM)), np.diff(reference, axis=0) / dt])
    guess_velocities[-1] = 0.0
    guess_inputs = (
        np.diff(guess_positions, axis=0)
        if params.control_mode == "position"
        else mass * np.diff(guess_velocities, axis=0) / dt
    )
    guess_duals = None
    use_conic = solver in CONIC_PLUGINS

    # ---- build the problem once; only the corridor values change later -------
    opti = ca.Opti("conic") if use_conic else ca.Opti()
    positions = opti.variable(DIM, n_steps + 1)
    inputs = opti.variable(DIM, n_steps)

    if params.control_mode == "force":
        velocities = opti.variable(DIM, n_steps + 1)
        opti.subject_to(
            positions[:, 1:]
            == positions[:, :-1]
            + dt * velocities[:, :-1]
            + (0.5 * dt**2 / mass) * inputs
        )
        opti.subject_to(velocities[:, 1:] == velocities[:, :-1] + (dt / mass) * inputs)
        opti.subject_to(velocities[:, 0] == 0.0)
        opti.subject_to(velocities[:, n_steps] == 0.0)
        opti.subject_to(
            opti.bounded(-params.max_speed, ca.vec(velocities), params.max_speed)
        )
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
        opti.subject_to(opti.bounded(-max_step, ca.vec(inputs), max_step))
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

    # the three half-spaces holding each knot are parameters, not constants: only
    # which triangle holds a knot changes between outer iterations, never the
    # structure, so nothing has to be rebuilt or re-differentiated
    corridor_normal_x = opti.parameter(3, n_steps + 1)
    corridor_normal_y = opti.parameter(3, n_steps + 1)
    corridor_offset = opti.parameter(3, n_steps + 1)
    opti.subject_to(
        ca.vec(
            corridor_normal_x * ca.repmat(positions[0, :], 3, 1)
            + corridor_normal_y * ca.repmat(positions[1, :], 3, 1)
            - corridor_offset
        )
        <= 0.0
    )

    opti.minimize(
        params.weight_tracking * ca.sumsqr(positions - ca.DM(reference.T))
        + params.weight_input * ca.sumsqr(inputs)
        + params.weight_smoothness * ca.sumsqr(inputs[:, 1:] - inputs[:, :-1])
    )
    plugin_options: dict = {}
    solver_defaults: dict = {}
    if solver == "ipopt":
        plugin_options["print_time"] = False
        solver_defaults = {"print_level": 0, "sb": "yes"}
    elif solver == "gurobi":
        solver_defaults = {"OutputFlag": int(verbose)}
    elif solver == "knitro":
        plugin_options["print_time"] = False
        solver_defaults = {"outlev": int(verbose)}
    opti.solver(solver, plugin_options, {**solver_defaults, **(solver_options or {})})

    for iteration in range(max_outer_iterations):
        knot_normals = edge_normals[triangle_of_knot]  # (n_steps+1, 3, 2), a copy
        knot_offsets = edge_offsets[triangle_of_knot]  # (n_steps+1, 3), a copy
        # the endpoints are pinned and may sit exactly on the corridor boundary,
        # so their constraints are made vacuous (0 <= 1) rather than dropped
        knot_normals[[0, -1]] = 0.0
        knot_offsets[[0, -1]] = 1.0
        opti.set_value(corridor_normal_x, knot_normals[:, :, 0].T)
        opti.set_value(corridor_normal_y, knot_normals[:, :, 1].T)
        opti.set_value(corridor_offset, knot_offsets.T)

        opti.set_initial(positions, guess_positions.T)
        opti.set_initial(inputs, guess_inputs.T)
        if params.control_mode == "force":
            opti.set_initial(velocities, guess_velocities.T)
        if guess_duals is not None and not use_conic:
            opti.set_initial(opti.lam_g, guess_duals)
        opti_solution = opti.solve()

        optimal_positions = np.array(opti_solution.value(positions)).T
        optimal_inputs = np.array(opti_solution.value(inputs)).T
        optimal_velocities = (
            np.array(opti_solution.value(velocities)).T
            if params.control_mode == "force"
            else np.vstack([np.zeros((1, DIM)), optimal_inputs / dt])
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
        # the next solve differs only in which triangle each knot is held in, so
        # the previous primal-dual point is an excellent starting point
        guess_positions = optimal_positions
        guess_velocities = optimal_velocities
        guess_inputs = optimal_inputs
        guess_duals = None if use_conic else opti_solution.value(opti.lam_g)
    return solution


def solve_minlp(
    edge_normals: np.ndarray,
    edge_offsets: np.ndarray,
    geodesic: np.ndarray,
    start: np.ndarray,
    goal: np.ndarray,
    params: TrajParams,
    solver: str = "gurobi",
    warm_start: dict | None = None,
    schedule_window: int | None = None,
    solver_options: dict | None = None,
    verbose: bool = True,
) -> dict:
    """
    Mixed-integer variant: the solver picks the knot -> triangle assignment.

        knot_in_triangle[t, k] = 1 iff knot k lies in triangle t
        edge_normals[t] p_k <= edge_offsets[t]
                               + big_m[t] (1 - knot_in_triangle[t, k])
        sum_t knot_in_triangle[t, k] = 1
        progress_k = sum_t t knot_in_triangle[t, k], nondecreasing in k, with
            progress_0 = 0, progress_last = n_triangles - 1   (monotone traversal)

    Same dynamics, cost and geometry as solve_nlp(); an MIQP in both control
    modes. Only worth it when the triangle schedule itself matters, e.g. very
    uneven triangle sizes that defeat the arc-length assignment used by
    solve_nlp().

    Args:
        warm_start: solution dict from solve_nlp() used to initialize every
            variable, including the binaries (its 'triangle_of_knot' is a
            complete, feasible schedule). If None, solve_nlp() is called here.
            Note that whether CasADi forwards this initial point to the solver
            as a genuine MIP start is plugin-dependent, so do not rely on it
            alone: schedule_window below fixes binaries through constraints and
            therefore always takes effect.
        schedule_window: if given, each knot may only be assigned a triangle
            within this many positions of the warm-started schedule; the rest of
            the binaries are fixed to zero. This is the reliable lever on solve
            time (a window of 1 or 2 removes most of the tree), at the price of
            optimality only over schedules near the warm start, not all of them.
            It also restores the tightness lost by allowing the schedule to
            advance freely, and anchors it to the winding of the geodesic; pick a
            window of at least the largest jump in the warm-started schedule,
            which is np.diff(warm_start['triangle_of_knot']).max().
        solver: 'gurobi' (default) is the right tool here, and is dispatched
            through Opti("conic") since it is a CasADi conic plugin, not an
            nlpsol one; 'cplex' likewise. 'bonmin' and 'knitro' go through
            nlpsol and also work, but are general MINLP codes solving what is
            really an MIQP.
        solver_options: options for the chosen solver, merged over the defaults.
            Accepting a small optimality gap is the cheapest speedup available:
            {'MIPGap': 0.01} for gurobi, {'mip_opt_gap_rel': 0.01} for knitro,
            {'allowable_fraction_gap': 0.01} for bonmin. Also useful for gurobi:
            'TimeLimit' (seconds, essential when sweeping many homotopy classes),
            'Threads', and 'MIPFocus' (1 finds feasible solutions sooner, 3
            closes the bound faster, which suits a warm-started run). For knitro:
            'numthreads' and 'mip_numthreads' (branch-and-bound threads; needs
            mip_method=1, which is set by default here), 'maxtime_real', and
            'mip_terminate' (1 stops at the first integer feasible point).

    Note that a big-M formulation has a weak relaxation, so branch-and-bound
    usually finds the optimal incumbent quickly and then spends most of its time
    proving optimality. If that proof is not what you need, cap it with a gap
    ('MIPGap' / 'mip_opt_gap_rel'), a time limit, or mip_terminate=1.

    With the binaries fixed this is a QP, so this is an MIQP: branch-and-bound
    gets valid bounds from its relaxations and every node solve is a QP.
    """
    n_steps, dt, mass = params.n_steps, params.dt, params.robot_mass
    n_triangles = len(edge_normals)
    start = np.asarray(start, dtype=float).reshape(DIM)
    goal = np.asarray(goal, dtype=float).reshape(DIM)

    if warm_start is None:
        warm_start = solve_nlp(
            edge_normals, edge_offsets, geodesic, start, goal, params, verbose=verbose
        )
    warm_schedule = np.asarray(warm_start["triangle_of_knot"])

    segment_lengths = np.linalg.norm(np.diff(geodesic, axis=0), axis=1)
    arclength = np.concatenate([[0.0], np.cumsum(segment_lengths)])
    reference = np.stack(
        [
            np.interp(
                np.linspace(0.0, arclength[-1], n_steps + 1),
                arclength,
                geodesic[:, axis],
            )
            for axis in range(DIM)
        ],
        axis=1,
    )

    # Big-M must be as tight as validity allows: a loose value weakens the
    # relaxation solved at every branch-and-bound node, and is a classic cause of
    # long solve times. Bound each knot by the corridor's own extent, intersected
    # with what is reachable from start and goal in the steps available. The
    # corridor vertices are the pairwise intersections of each triangle's three
    # edge lines (already including the clearance shrink).
    edge_pairs = [(0, 1), (1, 2), (2, 0)]
    corridor_vertices = np.concatenate(
        [
            np.linalg.solve(edge_normals[:, pair, :], edge_offsets[:, pair, None])[
                ..., 0
            ]
            for pair in edge_pairs
        ]
    )
    step_bound = params.max_speed * dt
    if params.control_mode == "force":
        step_bound += 0.5 * params.max_acceleration * dt**2
    knots = np.arange(n_steps + 1)[:, None]
    box_min = np.maximum(
        np.maximum(start - knots * step_bound, goal - (n_steps - knots) * step_bound),
        corridor_vertices.min(axis=0),
    )
    box_max = np.minimum(
        np.minimum(start + knots * step_bound, goal + (n_steps - knots) * step_bound),
        corridor_vertices.max(axis=0),
    )
    # largest value each half-space can take over that box, per knot
    big_m = (
        np.maximum(edge_normals, 0.0) @ box_max.T
        + np.minimum(edge_normals, 0.0) @ box_min.T
        - edge_offsets[:, :, None]
    )
    big_m = np.maximum(big_m, 0.0) + 1e-3  # (n_triangles, 3, n_steps + 1)

    use_conic = solver in CONIC_PLUGINS
    opti = ca.Opti("conic") if use_conic else ca.Opti()
    positions = opti.variable(DIM, n_steps + 1)
    inputs = opti.variable(DIM, n_steps)
    knot_in_triangle = opti.variable(n_triangles, n_steps + 1)
    opti.subject_to(opti.bounded(0.0, ca.vec(knot_in_triangle), 1.0))

    if params.control_mode == "force":
        velocities = opti.variable(DIM, n_steps + 1)
        opti.subject_to(
            positions[:, 1:]
            == positions[:, :-1]
            + dt * velocities[:, :-1]
            + (0.5 * dt**2 / mass) * inputs
        )
        opti.subject_to(velocities[:, 1:] == velocities[:, :-1] + (dt / mass) * inputs)
        opti.subject_to(velocities[:, 0] == 0.0)
        opti.subject_to(velocities[:, n_steps] == 0.0)
        opti.subject_to(
            opti.bounded(-params.max_speed, ca.vec(velocities), params.max_speed)
        )
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
        opti.subject_to(opti.bounded(-max_step, ca.vec(inputs), max_step))
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
                + ca.DM(big_m[triangle][:, knot])
                * (1.0 - knot_in_triangle[triangle, knot])
            )
    for knot in range(n_steps + 1):
        opti.subject_to(ca.sum1(knot_in_triangle[:, knot]) == 1.0)

    # The schedule may only advance, but it may advance by more than one triangle
    # per knot: a taut geodesic touches the triangles that fan around an obstacle
    # corner at a single point, so no knot ever lands inside them however fine the
    # discretization. Requiring one triangle per knot makes such corridors -- the
    # normal case -- infeasible.
    progress = ca.DM(np.arange(n_triangles)).T @ knot_in_triangle
    opti.subject_to(
        opti.bounded(
            0.0, ca.vec(progress[:, 1:] - progress[:, :-1]), float(n_triangles - 1)
        )
    )
    opti.subject_to(progress[:, 0] == 0.0)
    opti.subject_to(progress[:, n_steps] == n_triangles - 1)

    # keep the schedule within a window of the warm-started one, if asked
    if schedule_window is not None:
        for knot in range(n_steps + 1):
            for triangle in range(n_triangles):
                if abs(triangle - warm_schedule[knot]) > schedule_window:
                    opti.subject_to(knot_in_triangle[triangle, knot] == 0.0)

    opti.minimize(
        params.weight_tracking * ca.sumsqr(positions - ca.DM(reference.T))
        + params.weight_input * ca.sumsqr(inputs)
        + params.weight_smoothness * ca.sumsqr(inputs[:, 1:] - inputs[:, :-1])
    )
    # warm start: continuous variables from the NLP solution, binaries from its
    # schedule, so branch-and-bound starts from a feasible point instead of zero
    guess_binaries = np.zeros((n_triangles, n_steps + 1))
    guess_binaries[warm_schedule, np.arange(n_steps + 1)] = 1.0
    opti.set_initial(positions, warm_start["positions"].T)
    opti.set_initial(inputs, warm_start["inputs"].T)
    opti.set_initial(knot_in_triangle, guess_binaries)
    if params.control_mode == "force":
        opti.set_initial(velocities, warm_start["velocities"].T)
    # integrality mask follows declaration order: positions, inputs,
    # knot_in_triangle, then velocities in force mode
    is_discrete = (
        [False] * (DIM * (2 * n_steps + 1))
        + [True] * (n_triangles * (n_steps + 1))
        + [False] * (DIM * (n_steps + 1) if params.control_mode == "force" else 0)
    )
    plugin_options: dict = {"discrete": is_discrete, "record_time": True}
    solver_defaults: dict = {}
    if solver == "gurobi":
        solver_defaults = {
            "OutputFlag": 1,
            "LogToConsole": 1,
        }
    elif solver == "bonmin":
        plugin_options["print_time"] = False
        solver_defaults = {
            "print_level": 0,
        }
    elif solver == "knitro":
        # mip_numthreads only takes effect with the branch-and-bound method, so
        # mip_method is pinned to it rather than left to Knitro's automatic choice
        plugin_options["print_time"] = False
        solver_defaults = {
            "outlev": 1,
            "mip_method": 1,
            "mip_terminate": 0,
            "ms_enable": 1,
            "numthreads": 16,
            "ms_numthreads": 16,
            "ms_maxsolves": 4,
            "mip_numthreads": 16,
        }
    opti.solver(solver, plugin_options, {**solver_defaults, **(solver_options or {})})
    try:
        opti_solution = opti.solve()
        stats = opti_solution.stats()
        solve_time = stats["t_wall_total"]
    except RuntimeError as e:
        if "KN_RC_MIP_TERM_FEAS" in str(e):
            print("Knitro found a feasible but not necessarily optimal solution.")
            opti_solution = opti.debug
            stats = opti_solution.stats()  # CHECKME
            solve_time = stats["t_wall_total"]
        else:
            # Handle other types of failures
            print(f"Solver failed with error: {e}")
            raise

    optimal_positions = np.array(opti_solution.value(positions)).T
    optimal_inputs = np.array(opti_solution.value(inputs)).T
    optimal_velocities = (
        np.array(opti_solution.value(velocities)).T
        if params.control_mode == "force"
        else np.vstack([np.zeros((1, DIM)), optimal_inputs / dt])
    )
    return dict(
        positions=optimal_positions,
        velocities=optimal_velocities,
        inputs=optimal_inputs,
        triangle_of_knot=np.argmax(
            np.array(opti_solution.value(knot_in_triangle)), axis=0
        ),
        cost=float(opti_solution.value(opti.f)),
        solve_time=solve_time,
    )
