
from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional, Callable, Dict
from DMP_helper import *


def clamp(x: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    return np.minimum(np.maximum(x, lo), hi)

from dataclasses import dataclass
from typing import Tuple
import numpy as np

@dataclass
class FR3Params:
    urdf_path: str = "/home/asus/ros2_ws/src/franka_description/robots/fr3/fr3.urdf"
    gravity: np.ndarray = np.array([0.0, 0.0, -9.81])
    n_joints: int = 7
    ee_frame_name: str = "fr3_hand_tcp"
    finger_joint_names: Tuple[str, str] = ("fr3_finger_joint1", "fr3_finger_joint2")

class FR3Model:
    """Pinocchio-backed FR3 model reduced to 7 DoF by locking finger joints."""
    def __init__(self, params: FR3Params):
        self.p = params
        try:
            import pinocchio as pin
        except Exception as e:
            raise ImportError("Pinocchio is required for FR3Model. Install `pip install pin`.") from e
        self.pin = pin

        # Load the full model
        full_model = self.pin.buildModelFromUrdf(self.p.urdf_path)
        self.full_model = full_model
        self.full_data = self.full_model.createData()

        # Cache joint counts
        self.n_full = self.full_model.nq
        self.n_arm = self.p.n_joints

        # End-effector frame ID
        self.ee_fid = self.full_model.getFrameId(self.p.ee_frame_name)

    # Internal helpers to pad 7→full nq
    def _pad_q(self, q7: np.ndarray) -> np.ndarray:
        q_full = np.zeros(self.n_full)
        q_full[:self.n_arm] = q7
        return q_full

    def _pad_dq(self, dq7: np.ndarray) -> np.ndarray:
        dq_full = np.zeros(self.full_model.nv)
        dq_full[:self.n_arm] = dq7
        return dq_full

    # ---- Kinematics ----
    def fk_position(self, q7: np.ndarray) -> np.ndarray:
        pin = self.pin
        q_full = self._pad_q(q7)
        pin.forwardKinematics(self.full_model, self.full_data, q_full)
        pin.updateFramePlacements(self.full_model, self.full_data)
        oMf = self.full_data.oMf[self.ee_fid]
        return np.asarray(oMf.translation).reshape(3)

    def jacobian_linear(self, q7: np.ndarray) -> np.ndarray:
        pin = self.pin
        q_full = self._pad_q(q7)
        pin.computeJointJacobians(self.full_model, self.full_data, q_full)
        pin.updateFramePlacements(self.full_model, self.full_data)
        J6 = pin.getFrameJacobian(self.full_model, self.full_data, self.ee_fid, pin.ReferenceFrame.WORLD)
        return np.asarray(J6[:3, :self.n_arm])

    def jacobian_dot_linear(self, q7: np.ndarray, dq7: np.ndarray) -> np.ndarray:
        pin = self.pin
        q_full = self._pad_q(q7)
        dq_full = self._pad_dq(dq7)
        try:
            if hasattr(pin, 'computeJointJacobiansTimeVariation'):
                pin.computeJointJacobians(self.full_model, self.full_data, q_full)
                pin.computeJointJacobiansTimeVariation(self.full_model, self.full_data, q_full, dq_full)
                dJ6 = pin.getFrameJacobianTimeVariation(self.full_model, self.full_data, self.ee_fid, pin.ReferenceFrame.WORLD)
                dJv = np.asarray(dJ6[:3, :self.n_arm])
            else:
                raise AttributeError
        except Exception:
            eps = 1e-6
            q_fd = q7 + eps * dq7
            J0 = self.jacobian_linear(q7)
            J1 = self.jacobian_linear(q_fd)
            dJv = (J1 - J0) / eps
        return dJv

    # ---- Dynamics ----
    def M(self, q7: np.ndarray) -> np.ndarray:
        pin = self.pin
        q_full = self._pad_q(q7)
        M_full = pin.crba(self.full_model, self.full_data, q_full)
        M_full = (M_full + M_full.T) * 0.5
        return np.asarray(M_full)[:self.n_arm, :self.n_arm]

    def C(self, q7: np.ndarray, dq7: np.ndarray) -> np.ndarray:
        pin = self.pin
        q_full = self._pad_q(q7)
        dq_full = self._pad_dq(dq7)
        C_full = pin.computeCoriolisMatrix(self.full_model, self.full_data, q_full, dq_full)
        return np.asarray(C_full)[:self.n_arm, :self.n_arm]

    def g(self, q7: np.ndarray) -> np.ndarray:
        pin = self.pin
        q_full = self._pad_q(q7)
        g_full = pin.computeGeneralizedGravity(self.full_model, self.full_data, q_full)
        return np.asarray(g_full).reshape(-1)[:self.n_arm]

    def nle(self, q7: np.ndarray, dq7: np.ndarray) -> np.ndarray:
        pin = self.pin
        q_full = self._pad_q(q7)
        dq_full = self._pad_dq(dq7)
        nle_full = pin.nonLinearEffects(self.full_model, self.full_data, q_full, dq_full)
        return np.asarray(nle_full).reshape(-1)[:self.n_arm]

    def tau_fric(self, dq7: np.ndarray) -> np.ndarray:
        Kv = 0.02  # Identify on hardware later
        return Kv * dq7


@dataclass
class Theta:
    """Fictitious scale parameters for online adaptation via UKF.
    Each is a 7-vector scaling the corresponding term elementwise.
    """
    sM: np.ndarray  # scales M^{-1} via (I + diag(sM))
    sC: np.ndarray  # scales C
    sg: np.ndarray  # scales g
    sf: np.ndarray  # scales tau_fric

    @staticmethod
    def zeros(n: int = 7) -> "Theta":
        return Theta(sM=np.zeros(n), sC=np.zeros(n), sg=np.zeros(n), sf=np.zeros(n))

    def as_vector(self) -> np.ndarray:
        return np.concatenate([self.sM, self.sC, self.sg, self.sf])

    @staticmethod
    def from_vector(v: np.ndarray) -> "Theta":
        n = v.size // 4
        return Theta(sM=v[0:n], sC=v[n:2*n], sg=v[2*n:3*n], sf=v[3*n:4*n])



class FR3TaskDynamics:
    def __init__(self, model: FR3Model, cfg: DynConfig):
        self.mdl = model
        self.cfg = cfg
        self.nx = 20
        self.nu = 7
        self.n_theta = 28  # 4*7
        self.tau_limits = np.array([87, 87, 87, 87, 12, 12, 12], dtype=float)  # Nm
        self.dq_max = 8.0     # rad/s   (clip joint velocities when evaluating dynamics)
        self.ddq_max = 100.0  # rad/s^2 (clip joint accelerations result)
        self.a_max = 10.0     # m/s^2   (clip Cartesian linear accel)
        self.lam0 = 1e-3      # base mass-matrix regularization

        # ---- helper ----
    def _solve_M_ddq(self, M, rhs, dq):
        dq_norm_sq = float(np.dot(dq, dq))
        lam = self.lam0 * (1.0 + dq_norm_sq)
        Mreg = 0.5 * (M + M.T) + lam * np.eye(7)
        try:
            return np.linalg.solve(Mreg, rhs)
        except np.linalg.LinAlgError:
            w, V = np.linalg.eigh(Mreg)
            w = np.clip(w, 1e-6, None)
            return V @ ((V.T @ rhs) / w)

    def _derivatives(self, x: np.ndarray, u_tau: np.ndarray, theta_vec: np.ndarray) -> np.ndarray:
        th = Theta.from_vector(theta_vec)

        # ---- unpack & clamp inputs ----
        p  = x[0:3]
        v  = x[3:6]
        q  = x[6:13]
        dq = x[13:20]
        # torque clamp (per-joint)
        u_tau = np.clip(u_tau, -self.tau_limits, self.tau_limits)
        # velocity clamp (keeps Coriolis bounded)
        dq = np.clip(dq, -self.dq_max, self.dq_max)

        # ---- model calls ----
        M  = self.mdl.M(q)
        C  = self.mdl.C(q, dq)
        g  = self.mdl.g(q)
        J  = self.mdl.jacobian_linear(q)
        try:
            dJ = self.mdl.jacobian_dot_linear(q, dq)
        except Exception:
            # extremely conservative fallback if time variation is unstable
            dJ = np.zeros_like(J)

        tau_f = self.mdl.tau_fric(dq)

        # ---- scaled terms ----
        I7 = np.eye(7)
        C_scaled   = (I7 + np.diag(th.sC)) @ C
        g_scaled   = (I7 + np.diag(th.sg)) @ g
        tauf_scaled= (I7 + np.diag(th.sf)) @ tau_f

        rhs = u_tau - (C_scaled @ dq) - g_scaled - tuf_scaled if False else u_tau - (C_scaled @ dq) - g_scaled - tauf_scaled  # keep original name

        # ---- solve for ddq with regularized M (no explicit inverse) ----
        ddq = self._solve_M_ddq(M, rhs, dq)

        # ddq clamp
        ddq = np.clip(ddq, -self.ddq_max, self.ddq_max)

        # ---- Cartesian linear accel ----
        a = J @ ddq + dJ @ dq
        a = np.clip(a, -self.a_max, self.a_max)
        # --- DEBUG CHECKS ---
        if np.any(np.isnan(u_tau)) or np.any(np.isinf(u_tau)):
            print("u_tau NaN/Inf", u_tau)
        if np.linalg.cond(M) > 1e6:
            print("M badly conditioned", np.linalg.cond(M))
        if np.max(np.abs(u_tau)) > 500:
            print("High torque input:", np.max(np.abs(u_tau)))

        # compute dynamics
        # try:
        #     ddq = self._solve_M_ddq(M, rhs, dq)
        #     ddq = np.clip(ddq, -self.ddq_max, self.ddq_max)

        #     # ---- Cartesian linear accel ----
        #     a = J @ ddq + dJ @ dq
        #     a = np.clip(a, -self.a_max, self.a_max)
        #     a = J @ ddq + dJ @ dq
        # except Exception as e:
        #     print("ERROR in dynamics:", e)
        #     print("M:", M)
        #     print("C:", C)
        #     print("g:", g)
        #     print("u_tau:", u_tau)
        #     raise

        if np.any(np.isnan(ddq)) or np.any(np.isinf(ddq)):
            print("ddq invalid: ", ddq)
        if np.any(np.isnan(a)) or np.any(np.isinf(a)):
            print("a invalid: ", a)

        dx = np.zeros_like(x)
        dx[0:3] = v
        dx[3:6] = a
        dx[6:13] = dq
        dx[13:20] = ddq
        if not np.isfinite(dx).all():
        # zero out the offending pieces to keep integrator alive
            print("dx contains NaN/Inf, zeroing out")
            dx = np.nan_to_num(dx, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

        return dx


    # def step(self, x: np.ndarray, u_tau: np.ndarray, theta_vec: np.ndarray) -> np.ndarray:
    #     dt = self.cfg.dt

    #     # Runge-Kutta 4 integration
    #     k1 = self._derivatives(x, u_tau, theta_vec)
    #     k2 = self._derivatives(x + 0.5 * dt * k1, u_tau, theta_vec)
    #     k3 = self._derivatives(x + 0.5 * dt * k2, u_tau, theta_vec)
    #     k4 = self._derivatives(x + dt * k3, u_tau, theta_vec)

    #     x_next = x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    #     return x_next
    def step(self, x: np.ndarray, u_tau: np.ndarray, theta_vec: np.ndarray) -> np.ndarray:
        dt = self.cfg.dt

        # keep torque bounded through the whole RK4 step
        u_tau = np.clip(u_tau, -self.tau_limits, self.tau_limits)

        k1 = self._derivatives(x,                u_tau, theta_vec)
        k2 = self._derivatives(x + 0.5*dt*k1,    u_tau, theta_vec)
        k3 = self._derivatives(x + 0.5*dt*k2,    u_tau, theta_vec)
        k4 = self._derivatives(x + dt*k3,        u_tau, theta_vec)

        x_next = x + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)

        # small post-step sanity clamp on joint velocities (prevents runaway)
        x_next[13:20] = np.clip(x_next[13:20], -self.dq_max, self.dq_max)

        # FK part stays whatever integration produced; if still non-finite, repair
        if not np.isfinite(x_next).all():
            x_next = np.nan_to_num(x_next, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

        return x_next

    def output(self, x: np.ndarray) -> np.ndarray:
        return x[0:3]  # end-effector position

@dataclass
class DynConfig:
    dt: float = 0.002
    process_noise_p: float = 1e-5
    process_noise_q: float = 1e-5

@dataclass
class AuxUKFTheta:  # θ for eq. (20): we keep it simple: 3 blocks of 3 scalings
    s_v: np.ndarray  # scales the v ← v + dt*u (velocity integration)
    s_u: np.ndarray  # scales the control effectiveness in B
    s_x: np.ndarray  # (optional) scales the x ← x + dt*v part (usually zeros)

    @staticmethod
    def zeros() -> "AuxUKFTheta":
        return AuxUKFTheta(s_v=np.zeros(3), s_u=np.zeros(3), s_x=np.zeros(3))

    @staticmethod
    def from_vector(v: np.ndarray) -> "AuxUKFTheta":
        # expect len=9 -> [s_x(3), s_v(3), s_u(3)] in that order
        assert v.size >= 9, "AuxUKFTheta expects at least 9 params"
        return AuxUKFTheta(s_x=v[0:3], s_v=v[3:6], s_u=v[6:9])
class AuxDynamics:
    """
    Discrete auxiliary model used **inside MPC**:
        x_aux = [X; V] ∈ R^6,  u_aux ∈ R^3
        X_{k+1} = X_k + dt*(I+diag(s_x))*V_k + 0.5*dt^2*(I+diag(s_u))*u_k
        V_{k+1} = (I+diag(s_v))*V_k + dt*(I+diag(s_u))*u_k

    When θ=0 (all scales = 0), this is a clean double integrator (eq. 11).
    """
    def __init__(self, dt: float):
        self.dt = float(dt)
        self.nx = 6   # [X(3), V(3)]
        self.nu = 3   # u_aux (3)

        # position selector E: takes the first 3 states as position
        self.E = np.hstack([np.eye(3), np.zeros((3,3))])  # (3x6)

    def AB(self, theta_vec: Optional[np.ndarray]):
        dt = self.dt
        if theta_vec is None:
            th = AuxUKFTheta.zeros()
        else:
            th = AuxUKFTheta.from_vector(theta_vec)

        Sx = np.eye(3) + np.diag(th.s_x)
        Sv = np.eye(3) + np.diag(th.s_v)
        Su = np.eye(3) + np.diag(th.s_u)

        A = np.block([
            [np.eye(3), dt * Sx],
            [np.zeros((3,3)), Sv]
        ])  # shape (6,6)

        B = np.vstack([
            0.5 * (dt ** 2) * Su,
            dt * Su
        ])  # shape (6,3)

        return A, B

    def step(self, x_aux: np.ndarray, u_aux: np.ndarray, theta_vec: Optional[np.ndarray]) -> np.ndarray:
        A, B = selfAB = self.AB(theta_vec)
        return A @ x_aux + B @ u_aux


def aux_to_tau(mdl: FR3Model, q: np.ndarray, dq: np.ndarray,
               Xdot: np.ndarray, u_aux: np.ndarray) -> np.ndarray:
    """
    τ = J^T ( Λ u_aux + μ + p )  (+ optional joint damping/friction)
    with:
      Λ   = (J M^{-1} J^T)^{-1}
      μ   = Λ [ J M^{-1} (C dq) - dJ dq ]
      p   = Λ [ J M^{-1} g ]
    """
    M  = mdl.M(q)
    C  = mdl.C(q, dq)
    g  = mdl.g(q)
    J  = mdl.jacobian_linear(q)
    dJ = mdl.jacobian_dot_linear(q, dq)

    # Regularize inverses
    Minv = np.linalg.inv(M + 1e-6*np.eye(7))
    JMJT = J @ Minv @ J.T
    JMJT_inv = np.linalg.inv(JMJT + 1e-9*np.eye(3))  # Λ

    # Task-space bias terms
    mu = JMJT_inv @ ( J @ (Minv @ (C @ dq)) - dJ @ dq )
    p  = JMJT_inv @ ( J @ (Minv @ g) )

    F_task = JMJT_inv @ u_aux + mu + p  # This equals Mx*u + Cx*Xdot + Gx
    tau = J.T @ F_task

    # optional viscous joint damping
    tau -= 0.2 * dq
    return tau

@dataclass
class UKFConfig:
    alpha: float = 1e-3
    beta: float = 2.0
    kappa: float = 0.0
    Rx: float = 1e-6  # process noise base
    Ry: float = 1e-6  # measurement noise base

class UKF:
    """Minimal UKF scaffold for augmented state x_e = [x; θ].
    Replace sigma-point details with a library if you prefer.
    """
    def __init__(self, dyn: FR3TaskDynamics, ukf_cfg: UKFConfig):
        self.dyn = dyn
        self.cfg = ukf_cfg
        self.nx = dyn.nx + dyn.n_theta
        self.ny = 3  # y = p
        self.P = np.eye(self.nx) * 1e-3
        self.xe = np.zeros(self.nx)
        self.R = np.eye(self.ny) * self.cfg.Ry
        self.Q = np.eye(self.nx) * self.cfg.Rx

    def init(self, x0: np.ndarray, theta0: Theta, P0: Optional[np.ndarray] = None):
        self.xe = np.concatenate([x0, theta0.as_vector()])
        if P0 is not None:
            self.P = P0.copy()

    # ---- Sigma points ----
    def _sigma_points(self, m: np.ndarray, P: np.ndarray):
        n = m.size
        lam = self.cfg.alpha**2 * (n + self.cfg.kappa) - n
        P = 0.5 * (P + P.T)
        P += 1e-9 * np.eye(P.shape[0])
        S = np.linalg.cholesky((n + lam) * P)
        Xi = np.zeros((n, 2*n + 1))
        Xi[:, 0] = m
        for i in range(n):
            Xi[:, i+1]     = m + S[:, i]
            Xi[:, i+1+n]   = m - S[:, i]
        Wm = np.full(2*n + 1, 1.0 / (2*(n + lam)))
        Wc = Wm.copy()
        Wm[0] = lam / (n + lam)
        Wc[0] = Wm[0] + (1 - self.cfg.alpha**2 + self.cfg.beta)
        return Xi, Wm, Wc

    # ---- Predict ----
    def predict(self, u: np.ndarray) -> None:
        n = self.xe.size
        x = self.xe
        P = self.P + self.Q
        Xi, Wm, Wc = self._sigma_points(x, P)

        # Propagate through dynamics for state part only
        X_pred = np.zeros_like(Xi)
        for j in range(Xi.shape[1]):
            xj = Xi[:, j]
            x_state = xj[0:self.dyn.nx]
            theta = xj[self.dyn.nx:]
            x_next = self.dyn.step(x_state, u, theta)
            # θ is random walk
            X_pred[:, j] = np.concatenate([x_next, theta])

        # Mean/covariance
        x_pred = np.sum(X_pred * Wm, axis=1)
        P_pred = np.zeros((n, n))
        for j in range(Xi.shape[1]):
            d = (X_pred[:, j] - x_pred).reshape(-1, 1)
            P_pred += Wc[j] * (d @ d.T)
        self.xe = x_pred
        self.P = P_pred

    # ---- Update with y = p ----
    def update(self, y_meas: np.ndarray) -> None:
        n = self.xe.size
        Xi, Wm, Wc = self._sigma_points(self.xe, self.P)
        Y = np.zeros((self.ny, Xi.shape[1]))
        for j in range(Xi.shape[1]):
            x_state = Xi[0:self.dyn.nx, j]
            yj = self.dyn.output(x_state)
            Y[:, j] = yj
        y_pred = np.sum(Y * Wm, axis=1)
        Py = np.zeros((self.ny, self.ny))
        Pxy = np.zeros((self.nx, self.ny))
        for j in range(Xi.shape[1]):
            dy = (Y[:, j] - y_pred).reshape(-1, 1)
            dx = (Xi[:, j] - self.xe).reshape(-1, 1)
            Py += Wc[j] * (dy @ dy.T)
            Pxy += Wc[j] * (dx @ dy.T)
        Py += self.R
        K = Pxy @ np.linalg.inv(Py)
        self.xe = self.xe + (K @ (y_meas - y_pred))
        self.P = self.P - K @ Py @ K.T



from dataclasses import dataclass, asdict
import numpy as np
import pandas as pd

# -------------------- CONFIG --------------------

@dataclass
class DMPConfig:
    alpha_s: float = 4.0
    alpha_z: float = 25.0
    beta_ratio: float = 0.25
    n_basis: int = 150
    ridge: float = 1e-6

@dataclass
class DMPModelXYZ:
    alpha_s: float
    alpha_z: float
    beta_z: float
    tau: float
    centers: np.ndarray
    widths: np.ndarray
    w_x: np.ndarray
    w_y: np.ndarray
    w_z: np.ndarray
    x0: np.ndarray
    g: np.ndarray

    def to_dict(self):
        d = asdict(self)
        for k, v in list(d.items()):
            if isinstance(v, np.ndarray):
                d[k] = v.tolist()
        return d


@dataclass
class MPCConfig:
    
    S: int = 15               # horizon steps
    Qp: float = 500.0         # position error weight
    Rtau: float = 1e-2        # torque weight
    Qterm: float = 1000.0     # terminal position weight
    # tau_min: float = -85.0    # Nm (per joint)
    # tau_max: float = 85.0
    dt: float = 0.002
    tau_limits = np.array([87, 87, 87, 87, 12, 12, 12])  
    tau_min = -tau_limits
    tau_max =  tau_limits
# --- Add at top of file once ---
import casadi as ca

# def _fd_jacobian(f, x, u, theta, eps=1e-6):
#     """Finite-difference linearization: returns (A,B,c) for x+ = f(x,u,theta)."""
#     n = x.size
#     m = u.size
#     fxu = f(x, u, theta)
#     A = np.zeros((n, n))
#     B = np.zeros((n, m))
#     # Columns of A
#     for i in range(n):
#         dx = np.zeros(n); dx[i] = eps
#         A[:, i] = (f(x + dx, u, theta) - fxu) / eps
#     # Columns of B
#     for j in range(m):
#         du = np.zeros(m); du[j] = eps
#         B[:, j] = (f(x, u + du, theta) - fxu) / eps
#     c = fxu - A @ x - B @ u
#     return A, B, c

class UMPC:
    """
    MPC on the **auxiliary linear model** (Ẍ = u_aux),
    with optional θ-deformation from UKF (eq. 20).
    Decision vars: u_aux[k] ∈ R^3.
    Cost: position tracking (task-space), torque-rate-like smoothing on u_aux,
          plus small control effort.
    Output: τ (7,) for the current step via feedback linearization mapping.
    """
    def __init__(self, dyn: FR3TaskDynamics, cfg: MPCConfig):
        self.plant = dyn            # used only for torque mapping context (q,dq)
        self.cfg = cfg
        self.aux = AuxDynamics(cfg.dt)

        # gradient descent hyperparams (active-set can replace this later)
        self.max_iters = 20
        self.alpha_step = 5e-2
        self.bt_beta = 0.5
        self.bt_c = 1e-4

        self.w_Ru  = cfg.Rtau    # small effort penalty (on u_aux, not torque)
        self.w_Rdu = 1e-3        # rate smoothing on u_aux

        # conservative bounds on u_aux (m/s^2); you can tighten/loosen
        self.u_min = -5.0 * np.ones(3)
        self.u_max =  5.0 * np.ones(3)

    # ---------- linear rollout ----------
    def _rollout(self, x_aux0: np.ndarray, U: np.ndarray, theta_vec: Optional[np.ndarray]):
        S = U.shape[0]
        X = np.zeros((S+1, 6))
        X[0] = x_aux0
        for k in range(S):
            X[k+1] = self.aux.step(X[k], U[k], theta_vec)
        return X

    # ---------- cost & gradient (adjoint) ----------
    def _cost_and_grad(self, X: np.ndarray, U: np.ndarray, d_seq: np.ndarray) -> tuple[float, np.ndarray]:
        """
        X: (S+1,6), U: (S,3), d_seq: (S,3) desired positions
        """
        S = U.shape[0]
        E = self.aux.E  # (3x6)

        # cost
        cost = 0.0
        for k in range(S):
            e = E @ X[k] - d_seq[k]                 # position error
            cost += self.cfg.Qp * (e @ e) + self.w_Ru * (U[k] @ U[k])
            if k > 0:
                du = U[k] - U[k-1]
                cost += self.w_Rdu * (du @ du)
        # terminal
        eT = E @ X[S] - d_seq[-1]
        cost += self.cfg.Qterm * (eT @ eT)

        # gradient via adjoint (costate) recursion
        A, B = self.aux.AB(None)  # NOTE: use θ-deformed AB at solve() time; filled there
        # place-holders; overwritten by caller with AB(θ) before call
        # We'll return just the structure here; in solve() we pass in A,B

        return cost, None  # we compute grad in solve() using A,B and X,U

    def _grad_adj(self, A: np.ndarray, B: np.ndarray, X: np.ndarray, U: np.ndarray, d_seq: np.ndarray) -> np.ndarray:
        S = U.shape[0]
        E = self.aux.E
        grad = np.zeros_like(U)

        # Costate recursion
        lam = np.zeros((S+1, 6))
        lam[S] = 2.0 * self.cfg.Qterm * (E.T @ (E @ X[S] - d_seq[-1]))  # terminal

        for k in reversed(range(S)):
            # gradient wrt u_k
            gk = 2.0 * self.w_Ru * U[k]
            if k > 0:
                gk += 2.0 * self.w_Rdu * (U[k] - U[k-1])
            if k < S-1:
                gk -= 2.0 * self.w_Rdu * (U[k+1] - U[k])
            grad[k] = gk + B.T @ lam[k+1]

            # backprop costate
            ek = E @ X[k] - d_seq[k]
            lam[k] = (A.T @ lam[k+1]) + 2.0 * self.cfg.Qp * (E.T @ ek)

        return grad

    # ---------- public solve ----------
    def solve(self,
              x_full: np.ndarray,          # full plant state: [p,v,q,dq] (20,)
              theta_vec: Optional[np.ndarray],  # UKF θ (at least 9-dim for aux)
              d_seq: np.ndarray,           # (S,3) desired positions
              u_prev: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Returns: τ_cmd (7,) for current step, computed from u_aux[0] via feedback linearization.
        """
        S = self.cfg.S
        dt = self.cfg.dt

        # 1) Build **auxiliary** initial state from current plant state
        p = x_full[0:3]
        v = x_full[3:6]
        x_aux0 = np.hstack([p, v])    # [X, V]

        # 2) Get θ-deformed (A,B)
        A, B = self.aux.AB(theta_vec)  # θ from UKF; pass None for nominal

        # 3) Warm start
        if u_prev is None:
            U = np.zeros((S, 3))
        else:
            U = np.tile(u_prev[:3], (S, 1))
        U = np.clip(U, self.u_min, self.u_max)

        # 4) Simple projected gradient with Armijo
        for _ in range(self.max_iters):
            # rollout
            X = np.zeros((S+1, 6))
            X[0] = x_aux0
            for k in range(S):
                X[k+1] = A @ X[k] + B @ U[k]

            # cost + grad (use adjoint)
            # (we compute cost again here for Armijo; ok for small S)
            cost = 0.0
            E = self.aux.E
            for k in range(S):
                e = E @ X[k] - d_seq[k]
                cost += self.cfg.Qp * (e @ e) + self.w_Ru * (U[k] @ U[k])
                if k > 0:
                    du = U[k] - U[k-1]
                    cost += self.w_Rdu * (du @ du)
            eT = E @ X[S] - d_seq[-1]
            cost += self.cfg.Qterm * (eT @ eT)

            grad = self._grad_adj(A, B, X, U, d_seq)

            # Armijo
            step = self.alpha_step
            base = cost
            while True:
                U_trial = np.clip(U - step * grad, self.u_min, self.u_max)

                # rollout for trial cost
                Xtr = np.zeros((S+1, 6)); Xtr[0] = x_aux0
                for k in range(S):
                    Xtr[k+1] = A @ Xtr[k] + B @ U_trial[k]
                cost_tr = 0.0
                for k in range(S):
                    e = E @ Xtr[k] - d_seq[k]
                    cost_tr += self.cfg.Qp * (e @ e) + self.w_Ru * (U_trial[k] @ U_trial[k])
                    if k > 0:
                        du = U_trial[k] - U_trial[k-1]
                        cost_tr += self.w_Rdu * (du @ du)
                eT = E @ Xtr[S] - d_seq[-1]
                cost_tr += self.cfg.Qterm * (eT @ eT)

                if cost_tr <= base - self.bt_c * step * np.sum(grad * grad):
                    U = U_trial
                    break
                step *= self.bt_beta
                if step < 1e-8:
                    break

        # 5) Map u_aux[0] to τ via feedback linearization (paper’s eq. 10)
        q  = x_full[6:13]
        dq = x_full[13:20]
        Xdot = v  # from x_full
        
        u0 = U[0]
        tau_cmd = aux_to_tau(self.plant.mdl, q, dq, Xdot, u0)

        # clamp to actuator limits
        tau_cmd = np.clip(tau_cmd, self.cfg.tau_min, self.cfg.tau_max)

        return tau_cmd


    
import numpy as np
import csv
import time
import matplotlib.pyplot as plt

def main():
    # --- Setup components ---
    params = FR3Params()
    mdl = FR3Model(params)

    # Plant (nonlinear FR3 dynamics) still used to integrate truth:
    dyn = FR3TaskDynamics(mdl, DynConfig(dt=0.01))

    # UKF (kept; if you want to run nominal-only, set use_ukf=False below)
    ukf = UKF(dyn, UKFConfig())

    # NEW: UMPC now optimizes u_aux on linear aux model and maps to torque
    mpc = UMPC(dyn, MPCConfig(S=10, dt=0.01))

    # === Load and train DMP from demo ===
    file_path = "/home/asus/Documents/DMP_LfD/paper_roboface_demonstrated_joint_positions(t,j1-j7,x,y,z).csv"
    trajectories = load_and_augment_trajectories(file_path, n_aug=0)
    t_demo = trajectories[0]["t"].values
    x_demo = trajectories[0]["x"].values
    y_demo = trajectories[0]["y"].values
    z_demo = trajectories[0]["z"].values

    dmp_model = train_dmp_xyz(t_demo, x_demo, y_demo, z_demo, DMPConfig())

    # --- Initial state (plant state) ---
    q0 = np.zeros(7)
    dq0 = np.zeros(7)
    p0 = mdl.fk_position(q0)
    v0 = mdl.jacobian_linear(q0) @ dq0
    x0 = np.concatenate([p0, v0, q0, dq0])  # [X, V, q, dq] ∈ R^20

    # UKF init (kept)
    theta0 = Theta.zeros().as_vector()
    ukf.init(x0, Theta.zeros())
    tau_demo = t_demo[-1] - t_demo[0]
    print(f"tau_demo:{tau_demo}")
    # --- DMP rollout reference trajectory (task-space) ---
    # You can change tau=5.0 to match your demo duration
    # T_rollout, Xr, Yr, Zr = dmp_rollout(dmp_model, dt=dyn.cfg.dt, tau=tau_demo)
    T_rollout, Xr, Yr, Zr = dmp_rollout_on_time(dmp_model, t_demo)

    
    ref_traj = np.vstack([Xr, Yr, Zr]).T  # (N, 3)
    print(f"ref_traj shape: {ref_traj.shape}")
    plot_reconstruction(t_demo, x_demo, y_demo, z_demo, T_rollout, Xr, Yr, Zr)
    dmp_data = np.hstack([T_rollout.reshape(-1, 1), ref_traj])
    header = "time,x_desired,y_desired,z_desired"
    np.savetxt("desired_dmp_traj.csv", dmp_data, delimiter=",", header=header, comments='')

    print(f"Saved desired trajectory to desired_dmp_traj.csv with shape {dmp_data.shape}")

    # Warm-starts:
    u_prev_tau  = np.zeros(7)  # previous torque sent to plant (for UKF.predict)
    u_prev_uaux = None         # previous u_aux for MPC warm-start (3-D)

    steps = len(ref_traj) - mpc.cfg.S - 1

    # === Logging ===
    torque_log = []
    time_log = []
    t0 = time.time()
    ee_log = [] 

    use_ukf = True  # set False to run pure nominal aux model

    # --- Simulation loop ---
    for t in range(steps):
        if t % 50 == 0:
            print(f"Step {t}/{steps}")

        # Horizon slice from DMP reference
        d_seq = ref_traj[t : t + mpc.cfg.S]  # (S, 3)

        # UKF predict / update (kept; uses previous torque)
        if use_ukf:
            ukf.predict(u_prev_tau)
            y_meas = mdl.fk_position(x0[6:13])  # FK from current q
            ukf.update(y_meas)
            xe_hat = ukf.xe
            x_hat = xe_hat[0:dyn.nx]
            theta_hat = xe_hat[dyn.nx:]  # original 28-dim vector in your scaffold
            # Extract the first 9 elements for aux deformation: [s_x(3), s_v(3), s_u(3)]
            theta_aux = theta_hat[:9]
        else:
            x_hat = x0
            theta_hat = Theta.zeros().as_vector()  # not used by plant here
            theta_aux = None  # nominal double integrator in MPC

        # === UMPC solve on auxiliary model ===
        # Returns τ_cmd (7,) computed from u_aux[0] via feedback linearization
        tau_cmd = mpc.solve(x_hat, theta_aux, d_seq, u_prev=u_prev_uaux)

        # === Integrate the **plant** with torque τ_cmd ===
        # We can ignore plant-theta scales for stability (use zeros)
        x0 = dyn.step(x0, tau_cmd, Theta.zeros().as_vector())

        # Update warm-starts
        u_prev_tau = tau_cmd
        # We don't recover u_aux[0] from inside UMPC; warm-start nominally (None) or keep last guess if you expose it
        u_prev_uaux = None

        # --- Log torques and time ---
        torque_log.append(tau_cmd.copy())
        time_log.append(T_rollout[t])
        # ee_pos = mdl.fk_position(x0[6:13])
        ee_pos = x0[:3].copy()
        ee_log.append(ee_pos.copy())
    print("Simulation finished.")

    # === Save torque log to CSV ===
    # torque_log = np.array(torque_log)
    # time_log = np.array(time_log).reshape(-1, 1)
    # data_to_save = np.hstack([time_log, torque_log])
    # header = "time," + ",".join([f"tau_j{i+1}" for i in range(7)])
    # np.savetxt("torque_log.csv", data_to_save, delimiter=",", header=header, comments='')
    # print("Torque log saved to torque_log_ukf.csv")
    torque_log = np.array(torque_log)
    ee_log = np.array(ee_log)
    time_log = np.array(time_log).reshape(-1, 1)

    # Save torques
    data_to_save = np.hstack([time_log, torque_log])
    header = "time," + ",".join([f"tau_j{i+1}" for i in range(7)])
    np.savetxt("torque_log.csv", data_to_save, delimiter=",", header=header, comments='')

    # Save end-effector tracking
    ee_data = np.hstack([time_log, ee_log])
    np.savetxt("ee_tracking.csv", ee_data, delimiter=",", header="time,x,y,z", comments='')

    print("Torque log saved to torque_log.csv")
    print("End-effector tracking saved to ee_tracking.csv")
    # === Plot tracking results ===
    plt.figure(figsize=(10,6))
    plt.plot(ref_traj[:len(ee_log),0], ref_traj[:len(ee_log),1], 'r--', label="Desired (DMP)")
    plt.plot(ee_log[:,0], ee_log[:,1], 'b-', label="Actual (MPC)")
    plt.xlabel("X [m]")
    plt.ylabel("Y [m]")
    plt.title("End-Effector Trajectory (XY Plane)")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()

    # Optionally, plot position vs time
    plt.figure(figsize=(10,6))
    plt.subplot(3,1,1)
    plt.plot(time_log, ref_traj[:len(ee_log),0], 'r--', label="x_des")
    plt.plot(time_log, ee_log[:,0], 'b-', label="x_act")
    plt.legend(); plt.grid()
    plt.subplot(3,1,2)
    plt.plot(time_log, ref_traj[:len(ee_log),1], 'r--', label="y_des")
    plt.plot(time_log, ee_log[:,1], 'b-', label="y_act")
    plt.legend(); plt.grid()
    plt.subplot(3,1,3)
    plt.plot(time_log, ref_traj[:len(ee_log),2], 'r--', label="z_des")
    plt.plot(time_log, ee_log[:,2], 'b-', label="z_act")
    plt.legend(); plt.grid()
    plt.suptitle("End-Effector Position Tracking vs Time")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()