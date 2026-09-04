# -*- coding: utf-8 -*-
"""MPC velocity-command controller for Crazyflie rectangular flight.

Self-contained.  Does NOT modify main.py / controller.py.  Reuses the
CrazyflieLink only through the standard send_velocity_world_setpoint() and
get_state() interface, so it can run without the Nokov SDK.

Design
------
Model (planar + altitude, world frame):
    p[k+1] = p[k] + dt * v[k]
    v[k+1] = beta * v[k] + (1-beta) * u[k],   beta=exp(-dt/tau)

State : s = [x, y, z, vx, vy, vz]         (6)
Input : u = [vx_cmd, vy_cmd, vz_cmd]       (3)   -> send_velocity_world_setpoint

Cost  : J = sum_k (s_k-ref_k)^T Q (s_k-ref_k)
             + u_k^T R u_k + (u_k-u_{k-1})^T S (u_k-u_{k-1})
Box   : |u_xy| <= vmax, -vz_lo <= u_z <= vz_hi
Solve : dense box-constrained QP by accelerated projected gradient (FISTA).

The controller replaces the old waypoint/PID setpoint logic.  The state is
taken from the Crazyflie onboard EKF (link.get_state()); it does not need the
Nokov motion-capture SDK for the control law.
"""

import math

import numpy as np


# --------------------------------------------------------------------------- #
# Box-constrained QP solver (FISTA / accelerated projected gradient)
# --------------------------------------------------------------------------- #
def solve_box_qp(h, g, lo, hi, z0=None, max_iter=250, tol=1e-7, seed=0):
    """Minimize  0.5*z'H*z + g'z  s.t.  lo <= z <= hi.

    H must be symmetric positive semi-definite.  Uses accelerated projected
    gradient with a power-iteration step-size estimate and warm start.
    Returns (z, cost, iters).
    """
    g = np.asarray(g, dtype=float).reshape(-1)
    h = np.asarray(h, dtype=float)
    lo = np.asarray(lo, dtype=float).reshape(-1)
    hi = np.asarray(hi, dtype=float).reshape(-1)
    n = g.size

    # Step size from a crude largest-eigenvalue estimate of H.
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(n)
    v /= np.linalg.norm(v) + 1e-12
    lam = 1.0
    for _ in range(80):
        w = h @ v
        lam = float(np.dot(v, w))
        nv = float(np.linalg.norm(w))
        if nv < 1e-12:
            break
        v = w / nv
    l_h = max(abs(lam), 1e-8)
    step = 1.0 / (1.5 * l_h + 1e-6)

    z = np.clip(lo, lo, hi) if z0 is None else np.clip(z0, lo, hi)
    y = z.copy()
    t = 1.0
    prev_cost = None
    iters = 0
    for it in range(1, max_iter + 1):
        grad = h @ y + g
        z_new = np.clip(y - step * grad, lo, hi)
        t_new = (1.0 + math.sqrt(1.0 + 4.0 * t * t)) / 2.0
        y = z_new + ((t - 1.0) / t_new) * (z_new - z)
        z = z_new
        t = t_new
        iters = it
        if it % 4 == 0:
            cost = 0.5 * float(z @ (h @ z)) + float(g @ z)
            if prev_cost is not None and abs(prev_cost - cost) < tol:
                break
            prev_cost = cost

    cost = 0.5 * float(z @ (h @ z)) + float(g @ z)
    return z, cost, iters


# --------------------------------------------------------------------------- #
# MPC velocity planner
# --------------------------------------------------------------------------- #
class MpcVelocityPlanner:
    """Discrete-time position+velocity model, solved as a box QP."""

    def __init__(self, dt=0.10, horizon=20, tau=0.30,
                 vmax=0.9, vz_up=0.6, vz_dn=0.6,
                 q_pos=60.0, q_vel=4.0, r_ctrl=0.4, r_slew=10.0):
        self.dt = float(dt)
        self.n = int(horizon)
        self.tau = float(tau)
        self.vmax = float(vmax)
        self.vz_up = float(vz_up)
        self.vz_dn = float(vz_dn)

        beta = math.exp(-self.dt / self.tau)
        a = 1.0 - beta
        self.a = a
        self.beta = beta

        # A (6x6), B (6x3)
        self.A = np.array([
            [1, 0, 0, self.dt, 0, 0],
            [0, 1, 0, 0, self.dt, 0],
            [0, 0, 1, 0, 0, self.dt],
            [0, 0, 0, beta, 0, 0],
            [0, 0, 0, 0, beta, 0],
            [0, 0, 0, 0, 0, beta]], dtype=float)
        self.B = np.zeros((6, 3))
        self.B[3, 0] = a
        self.B[4, 1] = a
        self.B[5, 2] = a

        # Weights.
        self.Q = np.diag([q_pos, q_pos, q_pos, q_vel, q_vel, q_vel])
        self.R = np.eye(3) * r_ctrl
        self.S = np.eye(3) * r_slew

        self._build_prediction()
        self._build_constant_cost()
        self.last_solve = {}

    # -- prediction matrices ------------------------------------------------ #
    def _build_prediction(self):
        nx, nu = 6, 3
        self._apow = [np.linalg.matrix_power(self.A, k) for k in range(self.n + 1)]
        self.M = []  # M[k] : 6 x (3N), maps z=[u_0..u_{N-1}] -> s_k contribution
        for k in range(self.n + 1):
            mk = np.zeros((nx, nu * self.n))
            for i in range(k):
                mk[:, 3 * i:3 * i + 3] = self._apow[k - 1 - i] @ self.B
            self.M.append(mk)

    def _build_constant_cost(self):
        nu = 3
        nc = nu * self.n
        H = np.zeros((nc, nc))
        for k in range(1, self.n + 1):
            Mk = self.M[k]
            H += 2.0 * Mk.T @ self.Q @ Mk
        # control-magnitude
        rblk = np.kron(np.eye(self.n), self.R)
        H += 2.0 * rblk
        # slew
        D = self._slew_matrix()
        sblk = np.kron(np.eye(self.n), self.S)
        H += 2.0 * D.T @ sblk @ D
        self.H = H
        self._slew_D = D
        self._slew_S = sblk
        # box bounds
        lo = np.tile([-self.vmax, -self.vmax, -self.vz_dn], self.n)
        hi = np.tile([self.vmax, self.vmax, self.vz_up], self.n)
        self.lo = lo
        self.hi = hi

    def _slew_matrix(self):
        nu = 3
        n = self.n
        D = np.zeros((nu * n, nu * n))
        for k in range(n):
            D[nu * k:nu * k + nu, nu * k:nu * k + nu] = np.eye(nu)
            if k > 0:
                D[nu * k:nu * k + nu, nu * (k - 1):nu * k] = -np.eye(nu)
        return D

    # -- solve -------------------------------------------------------------- #
    def gradient_terms(self, s0, ref, u_prev):
        """Build g for  J = 0.5 z'H z + g'z  given current state and refs."""
        nc = 3 * self.n
        s0 = np.asarray(s0, dtype=float).reshape(6)
        u_prev = np.asarray(u_prev, dtype=float).reshape(3)
        g = np.zeros(nc)
        for k in range(1, self.n + 1):
            fk = self._apow[k] @ s0 - np.asarray(ref[k - 1] if k - 1 < len(ref)
                                                else ref[-1], dtype=float).reshape(6)
            Mk = self.M[k]
            g += 2.0 * Mk.T @ (self.Q @ fk)
        # slew:  d0 = [-u_prev; 0; ...; 0]
        d0 = np.zeros(nc)
        d0[:3] = -u_prev
        g += 2.0 * self._slew_D.T @ (self._slew_S @ d0)
        return g

    def solve(self, s0, ref, u_prev, warm=None):
        g = self.gradient_terms(s0, ref, u_prev)
        z0 = None
        if warm is not None and len(warm) == 3 * self.n:
            # shift left one block and repeat last block
            z0 = np.concatenate([warm[3:], warm[-3:]])
        z, cost, iters = solve_box_qp(self.H, g, self.lo, self.hi, z0=z0)
        u_star = z[0:3].copy()
        self.last_solve = {"cost": cost, "iters": iters, "z": z}
        return u_star, z

    def predict_states(self, s0, z):
        """Return the predicted state sequence s_1..s_N for a solved input z."""
        z = np.asarray(z, dtype=float).reshape(-1)
        s = np.asarray(s0, dtype=float).reshape(6).copy()
        out = []
        for k in range(self.n):
            u_k = z[3 * k:3 * k + 3]
            s = self.A @ s + self.B @ u_k
            out.append(s.copy())
        return out


# --------------------------------------------------------------------------- #
# Mission reference (takeoff -> hover -> rectangle -> land)
# --------------------------------------------------------------------------- #
class MissionRef:
    """Piecewise reference in world frame on a single mission clock.

    sample(t) -> (pos_ref(3), vel_ref(3), phase).
    The rectangle is the closed square (0,0)->(s,0)->(s,s)->(0,s)->(0,0)
    at height hover_z, flown once at cruise speed.
    """

    def __init__(self, size=0.6, hover_z=1.0, takeoff_s=4.0, hover_s=3.0,
                 cruise=0.5, land_s=2.5, landing_z=0.12,
                 unlock_s=3.0, rect_enabled=True):
        self.size = float(size)
        self.hover_z = float(hover_z)
        self.takeoff_s = float(takeoff_s)
        self.hover_s = float(hover_s)
        self.cruise = float(cruise)
        self.land_s = float(land_s)
        self.landing_z = float(landing_z)
        self.unlock_s = float(unlock_s)
        self.rect_enabled = bool(rect_enabled)
        self.perimeter = 4.0 * self.size
        self.rect_s = self.perimeter / self.cruise
        self.t_takeoff_end = self.unlock_s + self.takeoff_s
        self.t_hover_end = self.t_takeoff_end + self.hover_s
        if self.rect_enabled:
            self.t_rect_end = self.t_hover_end + self.rect_s
        else:
            self.t_rect_end = self.t_hover_end
        self.t_land_end = self.t_rect_end + self.land_s

    def _rect_point(self, arc):
        arc = arc % self.perimeter
        s = self.size
        if arc < s:
            f = arc / s
            return (s * f, 0.0, self.hover_z), (self.cruise, 0.0, 0.0)
        if arc < 2 * s:
            f = (arc - s) / s
            return (s, s * f, self.hover_z), (0.0, self.cruise, 0.0)
        if arc < 3 * s:
            f = (arc - 2 * s) / s
            return (s * (1.0 - f), s, self.hover_z), (-self.cruise, 0.0, 0.0)
        f = (arc - 3 * s) / s
        return (0.0, s * (1.0 - f), self.hover_z), (0.0, -self.cruise, 0.0)

    def phase_of(self, t):
        if t < self.unlock_s:
            return "unlock"
        if t < self.t_takeoff_end:
            return "takeoff"
        if t < self.t_hover_end:
            return "hover"
        if t < self.t_rect_end:
            return "rect"
        if t < self.t_land_end:
            return "land"
        return "done"

    def sample(self, t):
        t = max(0.0, float(t))
        if t < self.unlock_s:
            return (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), "unlock"
        if t < self.t_takeoff_end:
            f = (t - self.unlock_s) / self.takeoff_s
            z = f * self.hover_z
            return (0.0, 0.0, z), (0.0, 0.0, self.hover_z / self.takeoff_s), "takeoff"
        if t < self.t_hover_end:
            return (0.0, 0.0, self.hover_z), (0.0, 0.0, 0.0), "hover"
        if t < self.t_rect_end:
            arc = (t - self.t_hover_end) * self.cruise
            (px, py, pz), (vx, vy, vz) = self._rect_point(arc)
            return (px, py, pz), (vx, vy, vz), "rect"
        if t < self.t_land_end:
            f = (t - self.t_rect_end) / self.land_s
            z = (1.0 - f) * self.hover_z + f * self.landing_z
            return (0.0, 0.0, z), (0.0, 0.0, (self.landing_z - self.hover_z) / self.land_s), "land"
        return (0.0, 0.0, self.landing_z), (0.0, 0.0, 0.0), "done"

    def ref_state(self, t):
        p, v, ph = self.sample(t)
        return np.array([p[0], p[1], p[2], v[0], v[1], v[2]]), ph


# --------------------------------------------------------------------------- #
# Small state observer (position -> velocity estimate, used when the Crazyflie
# link does not expose velocity directly).
# --------------------------------------------------------------------------- #
class VelocityObserver:
    def __init__(self, alpha=0.35):
        self.alpha = float(alpha)
        self._prev_p = None
        self._prev_t = None
        self.v = np.zeros(3)

    def update(self, pos, t):
        pos = np.asarray(pos, dtype=float).reshape(3)
        if self._prev_p is not None and self._prev_t is not None:
            dt = max(1e-3, t - self._prev_t)
            v_raw = (pos - self._prev_p) / dt
            self.v = (1.0 - self.alpha) * self.v + self.alpha * v_raw
        self._prev_p = pos.copy()
        self._prev_t = t
        return self.v.copy()
