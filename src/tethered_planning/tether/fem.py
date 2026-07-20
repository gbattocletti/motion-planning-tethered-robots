import numpy as np

from tethered_planning.env import env_2d
from tethered_planning.utils import curves


class TetherFEM2D:

    # Env parameters
    g: float = 9.81  # gravity acceleration
    # CHECKME/TODO: how to deal with case of planar movement where g does not deform the
    # tether shape? Add flag for this case?

    # Tether parameters
    diameter: float = 0.01  # diameter
    rho_cable: float = 1300.0  # kg/m^3 (cable material, incl. voids)
    area = np.pi * diameter**2 / 4.0  # cross-sectional area
    E: float = 5.0e8  # Young's modulus [Pa]
    EA = E * area  # axial stiffness
    EI = 0.0  # bending rigidity [N m^2]
    c_internal: float = 5.0  # internal axial damping coeff [N s]
    c_struct: float = 0.05  # structural viscous damping (C matrix) [N s/m]
    no_compression: bool = True  # wether the tether is able to react to compression

    # Medium parameters
    rho_water: float = 1025.0  # water density
    rho_air: float = 1.225  # air density
    Cd_normal: float = 1.2  # normal drag coefficient
    Cd_tangent: float = 0.02  # tangential (skin friction) drag coefficient
    Ca: float = 1.0  # added-mass coefficient (water only)

    # Removed
    # compression_taut_only: bool = True,       # cable can't push (no compression)

    def __init__(
        self,
        env: env_2d.Env2D,  # anchor, obstacles
        n_nodes: int,
        state: np.ndarray,
        input_mode: str = "position",  # {"force", "position"}
        dt: float = 0.001,
        medium: str = "water",  # {"water", "air", "none"}
        water_current: np.ndarray = np.zeros(2),
        wind: np.ndarray = np.zeros(2),
        # TODO
    ):
        """
        2D lumped-mass finite element model of a slack tether.

        The cable is discretized into n nodes connected by (n-1) linear elements.
        The state is an (n x 6) numpy array where each row is [x, y, vx, vy, ax, ay].

        Node 0 is clamped at the anchor point. Node n-1 (corresponding to the robot
        position) is the controlled endpoint. Depending on input_mode, the control input
        u is either a force [Fx, Fy] applied at that node (input_mode="force"), or an
        imposed position [x, y] for that node (input_mode="position"). In the position
        case the command must move smoothly between steps (no large jumps).

        Equations of motion of a single node:
            M * a = F_internal(x) - C * v + F_external(x, v, t) + u
        where
            M           lumped (diagonal) mass matrix, including added mass if the
                        medium is water;
            F_internal  axial spring-damper forces from the two adjacent elements
                        (+ optional discrete bending stiffness);
            C           viscous structural damping (diagonal);
            F_external  gravity/buoyancy, water-current drag, wind drag (depends on the
                        medium, disabled if medium=none).

        The numerical integration is performed with semi-implicit Euler:
            a(k)   = M^{-1} F(x(k), v(k))
            v(k+1) = v(k) + a(k) * dt
            x(k+1) = x(k) + v(k+1) * dt

        Args:
            env (env_2d.Env2D): environment object, containing the obstacles
            n_nodes (int): number of nodes in the FEM model
            state (np.ndarray): initial tether state. The tether state must be a
                (2, n_nodes) numpy array such that state[0, :] is the anchor point and
                state[-1, :] the initial robot location. The initial state is assumed
                to have no intersection with the obstacles.
            input_mode (str, optional): the input mode with which the robot is
                controlled. Can be "force" or "position".
            dt (float, optional): the time step to use in the numerical integration.
            medium (str): the medium where the tether moves (water, air, none)
            water_current (np.ndarray): uniform water current field (if medium is water)
            wind (np.ndarray): uniform wind field (if medium is air)

        Returns:
            None
        """
        # Initialize model properties
        self.env: env_2d.Env2D = env
        self.n: int = int(n_nodes)
        if self.n < 3:
            raise ValueError("FEM model needs at least 3 nodes.")
        if state.shape[0] != self.n:
            raise ValueError(
                "Number of nodes in tether state does not match the FEM "
                f"number of nodes {state.shape[0]} != {self.n}"
            )
        self.state: np.ndarray = np.hstack([state, np.zeros([self.n, 4])])  # add v, a
        self.anchor: np.ndarray = state[0, :]  # anchor point (initial tether node)
        self.l: float = curves.measure_length(self.state)  # compute initial length
        self.l_el: float = self.l / (self.n - 1)  # unstretched element length
        if input_mode not in ("force", "position"):
            raise ValueError("input_mode must be 'force' or 'position'")
        self.input_mode: str = input_mode
        self.dt: float = dt
        self.current: np.ndarray = water_current
        self.wind: np.ndarray = wind

        # Compute nodes properties
        m_elem = self.rho_cable * self.area * self.l_el  # lumped mass of each node
        self.m_node = np.full(self.n, m_elem)  # vector of node masses
        self.m_node[0] *= 0.5  # initial and final nodes only have 1/2 mass
        self.m_node[-1] *= 0.5

        # Compute nodes properties due to medium
        self.medium: str = medium
        self.m_added: np.ndarray = np.zeros(self.n)  # vecrtor of nodal added masses
        rho_forces: float  # effective density
        self.rho: float  # density of medium
        self.flow: np.ndarray  # flow in medium
        match self.medium:
            case "water":
                # activates buoyancy, added mass
                ma_elem = self.Ca * self.rho_water * self.area * self.l_el
                self.m_added = np.full(self.n, ma_elem)
                self.m_added[0] *= 0.5
                self.m_added[-1] *= 0.5
                rho_forces = self.rho_water
                self.rho = self.rho_water
                self.flow = self.current
            case "air":
                rho_forces = 0.0
                self.rho = self.rho_air
                self.flow = self.wind
            case "none":
                rho_forces = 0.0
                self.rho = 0.0
                self.flow = np.array([0, 0])
            case _:
                raise ValueError("medium must be 'water' or 'air' or 'none'.")
        self.m_eff = self.m_node + self.m_added  # effective mass (diagonal of M)

        # Compute nodes weight (net) due to gravity and buoyancy
        w_elem = (self.rho_cable - rho_forces) * self.area * self.l_el * self.g
        self.w_node = np.full(self.n, w_elem)  # vector of nodal net weight
        self.w_node[0] *= 0.5
        self.w_node[-1] *= 0.5

        # Friction with obstacles (0 = free slide, 1 = full stop due to friction)
        self.obs_friction: float = 0.0  # [0, 1]

    def step(
        self,
        u: np.ndarray,
    ) -> None:
        """
        Advances one time step of the FEM dynamics.

        Args:
            u (np.ndarray): (2,) control input at the free end node:
                - a force [Fx, Fy] if input_mode == "force",
                - a target position [x, y] if input_mode == "position"

        Returns:
            None
        """
        # Ensure correct dimension of control input
        u = np.asarray(u, dtype=float)  # ensure correct data type
        if u.shape != (2,):
            raise ValueError(f"u must have shape (2,) (got {u.shape}).")

        # Compute nodal forces and accelerations
        F = self.forces(u)
        acc = F / self.m_eff[:, None]

        # Boundary condition at anchor point
        acc[0] = 0.0

        # Numerical integration with semi-implicit Euler
        state_new = self.state.copy()
        state_new[:, 4:6] = acc
        state_new[:, 2:4] = self.state[:, 2:4] + acc * self.dt
        state_new[:, 0:2] = self.state[:, 0:2] + state_new[:, 2:4] * self.dt

        # Position-controlled free endpoint: impose the commanded position and derive
        # consistent velocity and acceleration (note: input must be smooth to avoid
        # sudden forces, since force is computed at next time step)
        # If pos control is used, cache endpoint pos and vel to restore them after the
        # obstacle contact resolution.
        if self.input_mode == "position":
            pos_endpoint_old = self.state[-1, 0:2].copy()
            vel_endpoint_old = self.state[-1, 2:4].copy()
            v_new = (u - pos_endpoint_old) / self.dt
            state_new[-1, 0:2] = u
            state_new[-1, 2:4] = v_new
            state_new[-1, 4:6] = (v_new - vel_endpoint_old) / self.dt

        # Obstacle contact: project penetrating nodes back to the boundary
        state_new = self._resolve_collisions(state_new)

        # Enforce obstacle contact at free endpoint (position control can override it)
        if self.input_mode == "position":
            state_new[-1, 0:2] = u
            state_new[-1, 2:4] = (u - pos_endpoint_old) / self.dt

        # Enforce anchor point
        state_new[0, 0:2] = self.anchor
        state_new[0, 2:6] = 0.0

        # Update the tether state
        self.state = state_new

    def _axial_forces(self) -> np.ndarray:
        """
        Spring-damper tension in each element, lumped onto the nodes.
        """
        # Initialize nodal forces due to tension
        f_nodes = np.zeros((self.n, 2))

        # Compute element length and tangential unit vectors
        elements = self.state[1:, :2] - self.state[:-1, :2]  # (n-1, 2)
        l_elements = np.linalg.norm(elements, axis=1)
        l_elements = np.where(l_elements < 1e-12, 1e-12, l_elements)
        t_elements = elements / l_elements[:, None]  # tangents to elements (axial vec)

        # Compute strain on each element
        strain = (l_elements - self.l_el) / self.l_el
        dvel = self.state[1:, 2:4] - self.state[:-1, 2:4]
        strain_rate = np.einsum("ij,ij->i", dvel, t_elements) / self.l_el

        # Compute tension along elements
        tension = self.EA * strain + self.c_internal * strain_rate
        if self.no_compression:
            tension = np.maximum(tension, 0.0)  # cable cannot push
        f_elem = tension[:, None] * t_elements  # force along elements due to tension

        # Compute resulting forces on the nodes
        f_nodes[:-1] += f_elem  # force on node i due to element i+1
        f_nodes[1:] -= f_elem  # force on node i due to element i-1
        return f_nodes

    def _bending_forces(self) -> np.ndarray:
        """
        Discrete bending force between adjacent elements. The resulting force penalizes
        curvature at interior nodes. Uses the standard discrete-beam approximation
            F_b ~ -EI * d^4 r / ds^4
        evaluated with finite differences on the node positions.
        """
        # Initialize forces on nodes due to bending
        f_nodes = np.zeros((self.n, 2))

        # Elastic constant
        if self.EI <= 0.0:
            return f_nodes
        k = self.EI / self.l_el**3

        # Second difference (proportional to curvature) at interior nodes (n-2, 2)
        curvature = (
            self.state[:-2, :2] - 2.0 * self.state[1:-1, :2] + self.state[2:, :2]
        )

        # Restoring forces acting on the nodes
        # F = -grad of bending energy (EI/2L0^3) * sum |curv|^2
        f_nodes[1:-1] += 2.0 * k * curvature
        f_nodes[:-2] -= k * curvature
        f_nodes[2:] -= k * curvature
        return f_nodes

    # TODO from here

    def _drag_forces(
        self,
    ) -> np.ndarray:
        """
        Quadratic fluid drag (Morison-type), split between normal and tangential force.
        """
        # Tangents at nodes
        elements = self.state[1:, :2] - self.state[:-1, :2]
        l_elements = np.linalg.norm(elements, axis=1)
        l_elements = np.where(l_elements < 1e-12, 1e-12, l_elements)
        t_elements = elements / l_elements[:, None]  # tangents to elements (axial vec)
        t_node = np.zeros((self.n, 2))  # tangents to nodes (average between elements)
        t_node[0] = t_elements[0]
        t_node[-1] = t_elements[-1]
        t_node[1:-1] = t_elements[:-1] + t_elements[1:]
        norms = np.linalg.norm(t_node, axis=1)
        t_node /= np.where(norms < 1e-12, 1.0, norms)[:, None]

        # Fluid velocity w.r.t. the tether
        v_rel = self.flow[None, :] - self.state[:, 2:4]
        v_t = np.einsum("ij,ij->i", v_rel, t_node)[:, None] * t_node  # tangential
        v_n = v_rel - v_t  # normal

        # Exposed length per node
        Ln = np.full(self.n, self.l_el)
        Ln[0] *= 0.5
        Ln[-1] *= 0.5

        # Normal and tangential forces
        Fn = (
            0.5
            * self.rho
            * self.Cd_normal
            * self.diameter
            * Ln[:, None]
            * np.linalg.norm(v_n, axis=1)[:, None]
            * v_n
        )
        Ft = (
            0.5
            * self.rho
            * self.Cd_tangent
            * np.pi
            * self.diameter
            * Ln[:, None]
            * np.linalg.norm(v_t, axis=1)[:, None]
            * v_t
        )
        return Fn + Ft

    def forces(self, u: np.ndarray) -> np.ndarray:
        """
        Total nodal force vector, shape (n, 2).

        Collects internal (axial + bending), environmental (weight, buoyancy, drag),
        and damping forces. In input_mode == "force" the control force u is added at the
        last node; in "position" mode u is ignored here (the end node is driven
        kinematically in step(), and the reaction force can be retrieved afterwards).
        """
        F: np.ndarray  # (2, n) nodal force
        F = self._axial_forces()
        F += self._bending_forces()
        F += self._drag_forces()
        F[:, 1] -= self.w_node  # net weight (down = -y)
        F -= self.c_struct * self.state[:, 2:4]  # structural damping C * v
        if self.input_mode == "force":
            F[-1] += np.asarray(u, dtype=float)  # control force at free endpoint
        return F

    @staticmethod
    def _point_in_polygon(
        point: np.ndarray,
        polygon: np.ndarray,
    ) -> bool:
        """
        Efficient check to determine if a point is inside or outside a simple polygon.
        The method is based on the ray-casting-inside-test.
        """
        # TODO: move to env class (env method taking into account only the point)
        x, y = point
        inside = False
        j = len(polygon) - 1
        for i in range(len(polygon)):
            xi, yi = polygon[i]
            xj, yj = polygon[j]
            if (yi > y) != (yj > y):
                x_cross = (xj - xi) * (y - yi) / (yj - yi) + xi
                if x < x_cross:
                    inside = not inside
            j = i
        return inside

    @staticmethod
    def _closest_point_on_boundary(p: np.ndarray, poly: np.ndarray):
        """
        Nearest point to p on a polygon's boundary (projection of p on the boundary).
        The method also returns the distance from p to the projected point.
        """
        # TODO: consider moving to env class
        best_q = None
        best_d2 = np.inf
        m = len(poly)
        for i in range(m):
            a = poly[i]
            b = poly[(i + 1) % m]
            ab = b - a
            denom = ab @ ab
            t = 0.0 if denom < 1e-16 else np.clip((p - a) @ ab / denom, 0.0, 1.0)
            q = a + t * ab
            d2 = (p - q) @ (p - q)
            if d2 < best_d2:
                best_d2 = d2
                best_q = q
        return best_q, np.sqrt(best_d2)

    def _resolve_collisions(
        self,
        state: np.ndarray,
    ) -> np.ndarray:
        """
        Geometric contact handling, applied as a postporcessing step after the
        integration step. For every node that ended up inside an obstacle:
            1. project it to the nearest point on the polygon boundary
            2. remove the inward (normal) component of its velocity
            3. optionally scale down the tangential component (friction).
        Modifies pos and vel in place. Valid for time steps small enough
        that a node cannot tunnel across an obstacle in a single step.

        Args:
            state (np.ndarray): (n, 6) tether state [x, y, vx, vy, ax, ay] to compute
                the obstacle collisions and resolution on.

        Returns:
            np.ndarray: (n, 6) corrected copy of the state, with penetrating nodes
                projected onto the obstacle boundaries and their velocities adjusted.

        Notes:
            - The time steps must be small enough that a node cannot tunnel across an
              obstacle in a single step.
            - Node 0 (the anchor) is fixed.
        """
        state = state.copy()  # avoid modifying the input
        eps = 1e-6
        poly: np.ndarray
        for poly in self.env.obstacle_vertices:
            # cheap bounding-box rejection
            lo = poly.min(axis=0) - eps
            hi = poly.max(axis=0) + eps
            for i in range(1, self.n):  # node 0 is clamped
                p = state[i, :2]
                if np.any(p < lo) or np.any(p > hi):
                    continue
                if not self._point_in_polygon(p, poly):
                    continue
                q, dist = self._closest_point_on_boundary(p, poly)
                # outward normal: from the interior point toward the boundary
                if dist > 1e-12:
                    n_hat = (q - p) / dist
                else:  # exactly on the boundary
                    n_hat = p - poly.mean(axis=0)
                    n_hat /= max(np.linalg.norm(n_hat), 1e-12)

                # 1. project node on obstacle boundary
                state[i, :2] = q + eps * n_hat

                # 2. cancel inward normal velocity
                v_n = state[i, 2:4] @ n_hat
                if v_n < 0.0:
                    state[i, 2:4] -= v_n * n_hat

                # 3. add tangential friction
                if self.obs_friction > 0.0:
                    v_t = state[i, 2:4] - (state[i, 2:4] @ n_hat) * n_hat
                    state[i, 2:4] -= self.obs_friction * v_t

        return state

    def reaction_force_endpoint(self) -> np.ndarray:
        """Force required at the driven end to realize its current motion.

        Only meaningful in input_mode == 'position'. Uses Newton's law at the end node:
            R = m_eff * a_end - F_model,
        where F_model collects internal, environmental, and damping forces (no control
        input).
        """
        F_model = self.forces(np.zeros(2))[-1]
        return self.m_eff[-1] * self.state[-1, 4:6] - F_model
