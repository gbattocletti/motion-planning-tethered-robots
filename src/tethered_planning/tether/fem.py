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
        water_current: np.ndarray | None = None,
        wind: np.ndarray | None = None,
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
        rho_forces: float = 0.0  # effective density
        match self.medium:
            case "water":
                # activates buoyancy, added mass
                ma_elem = self.Ca * self.rho_water * self.area * self.l_el
                self.m_added = np.full(self.n, ma_elem)
                self.m_added[0] *= 0.5
                self.m_added[-1] *= 0.5
                rho_forces = self.rho_water
            case "air":
                pass
            case "none":
                pass
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
        input: np.ndarray,
        dt: float | None = None,
    ) -> None:
        pass  # TODO

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
        tang = elements / l_elements[:, None]

        # Compute strain on each element
        strain = (l_elements - self.l_el) / self.l_el
        dvel = self.state[1:, 2:4] - self.state[:-1, 2:4]
        strain_rate = np.einsum("ij,ij->i", dvel, tang) / self.l_el

        # Compute tension along elements
        tension = self.EA * strain + self.c_internal * strain_rate
        if self.no_compression:
            tension = np.maximum(tension, 0.0)  # cable cannot push
        f_elem = tension[:, None] * tang  # force along elements due to tension

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
        curvature = self.state[:-2, :2] - 2.0 * self.state[1:-1:2] + self.state[2:, :2]

        # Restoring forces acting on the nodes
        # F = -grad of bending energy (EI/2L0^3) * sum |curv|^2
        f_nodes[1:-1] += 2.0 * k * curvature
        f_nodes[:-2] -= k * curvature
        f_nodes[2:] -= k * curvature
        return f_nodes

    # TODO from here

    def _drag_forces(self, pos: np.ndarray, vel: np.ndarray) -> np.ndarray:
        """Quadratic fluid drag (Morison-type), split normal/tangential."""
        rho = self.rho_water if self.medium == "water" else self.rho_air
        flow = self.current if self.medium == "water" else self.wind

        # tangents at nodes (average of adjacent element tangents)
        seg = pos[1:] - pos[:-1]
        l = np.linalg.norm(seg, axis=1)
        l = np.where(l < 1e-12, 1e-12, l)
        t_el = seg / l[:, None]
        t_node = np.zeros((self.n, 2))
        t_node[0] = t_el[0]
        t_node[-1] = t_el[-1]
        t_node[1:-1] = t_el[:-1] + t_el[1:]
        norms = np.linalg.norm(t_node, axis=1)
        t_node /= np.where(norms < 1e-12, 1.0, norms)[:, None]

        v_rel = flow[None, :] - vel  # fluid velocity rel. to cable
        v_t = np.einsum("ij,ij->i", v_rel, t_node)[:, None] * t_node
        v_n = v_rel - v_t

        # exposed length per node
        Ln = np.full(self.n, self.L0)
        Ln[0] *= 0.5
        Ln[-1] *= 0.5

        Fn = (
            0.5
            * rho
            * self.Cdn
            * self.d
            * Ln[:, None]
            * np.linalg.norm(v_n, axis=1)[:, None]
            * v_n
        )
        Ft = (
            0.5
            * rho
            * self.Cdt
            * np.pi
            * self.d
            * Ln[:, None]
            * np.linalg.norm(v_t, axis=1)[:, None]
            * v_t
        )
        return Fn + Ft

    @staticmethod
    def _point_in_polygon(p: np.ndarray, poly: np.ndarray) -> bool:
        """Ray-casting inside test for a simple polygon (any orientation)."""
        x, y = p
        inside = False
        j = len(poly) - 1
        for i in range(len(poly)):
            xi, yi = poly[i]
            xj, yj = poly[j]
            if (yi > y) != (yj > y):
                x_cross = (xj - xi) * (y - yi) / (yj - yi) + xi
                if x < x_cross:
                    inside = not inside
            j = i
        return inside

    @staticmethod
    def _closest_point_on_boundary(p: np.ndarray, poly: np.ndarray):
        """Nearest point on the polygon's boundary to p, and its distance."""
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
        pos: np.ndarray,
        vel: np.ndarray,
    ):
        """
        Geometric contact handling, applied as a postporcessing step after the
        integration step. For every node that ended up inside an obstacle:
            1. project it to the nearest point on the polygon boundary
            2. remove the inward (normal) component of its velocity
            3. optionally scale down the tangential component (friction).
        Modifies pos and vel in place. Valid for time steps small enough
        that a node cannot tunnel across an obstacle in a single step.
        """
        eps = 1e-6
        for poly in self.env.obstacles_vertices:
            # cheap bounding-box rejection
            lo = poly.min(axis=0) - eps
            hi = poly.max(axis=0) + eps
            for i in range(1, self.n):  # node 0 is clamped
                p = pos[i]
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
                pos[i] = q + eps * n_hat

                # 2. cancel inward normal velocity
                v_n = vel[i] @ n_hat
                if v_n < 0.0:
                    vel[i] -= v_n * n_hat

                # 3. add tangential friction
                if self.obs_friction > 0.0:
                    v_t = vel[i] - (vel[i] @ n_hat) * n_hat
                    vel[i] -= self.obs_friction * v_t
