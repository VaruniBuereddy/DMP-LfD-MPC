
from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional, Callable, Dict
from DMP_helper import *

# ==========================
# Utility helpers
# ==========================

def clamp(x: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    return np.minimum(np.maximum(x, lo), hi)

# ==========================
# Robot model placeholders
# ==========================

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

# ==========================
# Uncertainty parameterization (θ)
# ==========================

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

# ==========================
# Discrete dynamics f(x, u, θ)
# ==========================

@dataclass
class DynConfig:
    dt: float = 0.002
    process_noise_p: float = 1e-5
    process_noise_q: float = 1e-5

class FR3TaskDynamics:
    def __init__(self, model: FR3Model, cfg: DynConfig):
        self.mdl = model
        self.cfg = cfg
        self.nx = 20
        self.nu = 7
        self.n_theta = 28  # 4*7

    def _derivatives(self, x: np.ndarray, u_tau: np.ndarray, theta_vec: np.ndarray) -> np.ndarray:
        th = Theta.from_vector(theta_vec)

        p = x[0:3]
        v = x[3:6]
        q = x[6:13]
        dq = x[13:20]

        M = self.mdl.M(q)
        C = self.mdl.C(q, dq)
        g = self.mdl.g(q)
        J = self.mdl.jacobian_linear(q)
        dJ = self.mdl.jacobian_dot_linear(q, dq)
        tau_f = self.mdl.tau_fric(dq)

        # Apply fictitious scales
        # Minv = np.linalg.inv(M)
        Minv = np.linalg.inv(M + 1e-6*np.eye(M.shape[0]))
        Minv_scaled = (np.eye(7) + np.diag(th.sM)) @ Minv
        C_scaled = (np.eye(7) + np.diag(th.sC)) @ C
        g_scaled = (np.eye(7) + np.diag(th.sg)) @ g
        tau_f_scaled = (np.eye(7) + np.diag(th.sf)) @ tau_f
        
        ddq = Minv_scaled @ (u_tau - C_scaled @ dq - g_scaled - tau_f_scaled)
        if not np.all(np.isfinite(ddq)):
            print("NaN in ddq at runtime!")
        print(f"‖dq‖={np.linalg.norm(dq):.2e}, ‖u_tau‖={np.linalg.norm(u_tau):.2e}, ‖ddq‖={np.linalg.norm(ddq):.2e}")

        a = J @ ddq + dJ @ dq

        dx = np.zeros_like(x)
        dx[0:3] = v
        dx[3:6] = a
        dx[6:13] = dq
        dx[13:20] = ddq
        return dx

    def step(self, x: np.ndarray, u_tau: np.ndarray, theta_vec: np.ndarray) -> np.ndarray:
        dt = self.cfg.dt

        # Runge-Kutta 4 integration
        k1 = self._derivatives(x, u_tau, theta_vec)
        k2 = self._derivatives(x + 0.5 * dt * k1, u_tau, theta_vec)
        k3 = self._derivatives(x + 0.5 * dt * k2, u_tau, theta_vec)
        k4 = self._derivatives(x + dt * k3, u_tau, theta_vec)

        x_next = x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        return x_next

    def output(self, x: np.ndarray) -> np.ndarray:
        return x[0:3]  # end-effector position

# ==========================
# UKF skeleton for x^e = [x; θ]
# ==========================

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


# ==========================
# DMP (position-only, 3D) placeholder
# ==========================

from dataclasses import dataclass, asdict
import numpy as np
import pandas as pd

# -------------------- CONFIG --------------------

@dataclass
class DMPConfig:
    alpha_s: float = 4.0
    alpha_z: float = 25.0
    beta_ratio: float = 0.25
    n_basis: int = 25
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



# ==========================
# UMPC scaffold
# ==========================

@dataclass
class MPCConfig:
    S: int = 15               # horizon steps
    Qp: float = 500.0         # position error weight
    Rtau: float = 1e-2        # torque weight
    Qterm: float = 1000.0     # terminal position weight
    tau_min: float = -85.0    # Nm (per joint)
    tau_max: float = 85.0
    dt: float = 0.002
# --- Add at top of file once ---
import casadi as ca

class UMPC:
    """
    Lightweight shooting-based NMPC:
      - projected gradient with Armijo backtracking
      - position tracking (task space)
      - terminal cost
      - torque and torque-rate regularization
    Replace with CasADi+IPOPT later if you like; API stays the same.
    """
    def __init__(self, dyn: FR3TaskDynamics, cfg: MPCConfig):
        self.dyn = dyn
        self.cfg = cfg
        # Extra hyperparameters for stability
        self.max_iters = 30          # gradient iters per solve
        self.alpha_step = 5e-3       # initial step size
        self.bt_beta = 0.5           # backtracking reduction
        self.bt_c = 1e-4             # Armijo constant
        self.Rdu = 1e-3              # input rate penalty

    def _rollout_cost(self, x0: np.ndarray, theta: np.ndarray,
                      u_seq: np.ndarray, d_seq: np.ndarray) -> float:
        """Roll forward dynamics and compute total cost with terminal term."""
        x = x0.copy()
        cost = 0.0
        for k in range(self.cfg.S):
            p = x[0:3]
            e = p - d_seq[k]
            cost += self.cfg.Qp * (e @ e) + self.cfg.Rtau * (u_seq[k] @ u_seq[k])
            if k > 0:
                du = u_seq[k] - u_seq[k-1]
                cost += self.Rdu * (du @ du)
            x = self.dyn.step(x, u_seq[k], theta)
        # terminal cost
        eT = x[0:3] - d_seq[-1]
        cost += self.cfg.Qterm * (eT @ eT)
        return cost

    def solve(self,
              x0: np.ndarray,
              theta: np.ndarray,
              d_seq: np.ndarray,
              u_prev: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Args:
          x0: state (20,)
          theta: θ vector (28,)
          d_seq: desired positions over horizon (S,3)
          u_prev: previous torque (7,) for warm-start
        Returns:
          u0: torque (7,)
        """
        S, nu = self.cfg.S, self.dyn.nu
        tau_lo = np.full(nu, self.cfg.tau_min)
        tau_hi = np.full(nu, self.cfg.tau_max)

        # warm start
        if u_prev is None:
            u = np.zeros((S, nu))
        else:
            u = np.tile(u_prev, (S, 1))
        u = clamp(u, tau_lo, tau_hi)

        for _ in range(self.max_iters):
            # forward rollout (store p and J for gradient shaping)
            x = x0.copy()
            p_list, J_list = [], []
            for k in range(S):
                p_list.append(x[0:3].copy())
                J_list.append(self.dyn.mdl.jacobian_linear(x[6:13]))  # (3x7)
                x = self.dyn.step(x, u[k], theta)
            pT = x[0:3]

            # cost
            cost = 0.0
            for k in range(S):
                e = p_list[k] - d_seq[k]
                cost += self.cfg.Qp * (e @ e) + self.cfg.Rtau * (u[k] @ u[k])
                if k > 0:
                    du_k = u[k] - u[k-1]
                    cost += self.Rdu * (du_k @ du_k)
            eT = pT - d_seq[-1]
            cost += self.cfg.Qterm * (eT @ eT)

            # gradient wrt controls (J^T shaping + regularizers)
            grad = np.zeros_like(u)
            for k in range(S):
                e = p_list[k] - d_seq[k]
                J = J_list[k]
                grad[k] += 2.0 * self.cfg.Qp   * (J.T @ e)      # tracking
                grad[k] += 2.0 * self.cfg.Rtau * u[k]           # effort
                if k > 0:
                    grad[k]   += 2.0 * self.Rdu * (u[k] - u[k-1])  # rate forward
                    grad[k-1] -= 2.0 * self.Rdu * (u[k] - u[k-1])  # rate backward
            # terminal shaping on last control using terminal Jacobian
            J_T = self.dyn.mdl.jacobian_linear(x[6:13])
            grad[-1] += 2.0 * self.cfg.Qterm * (J_T.T @ eT)

            # backtracking line search (projected)
            step = self.alpha_step
            base_cost = cost
            while True:
                u_trial = clamp(u - step * grad, tau_lo, tau_hi)
                trial_cost = self._rollout_cost(x0, theta, u_trial, d_seq)
                # Armijo condition
                if trial_cost <= base_cost - self.bt_c * step * np.sum(grad * grad):
                    u = u_trial
                    break
                step *= self.bt_beta
                if step < 1e-8:
                    break  # give up backtracking this iter

        return u[0]

# ==========================
# Example offline loop (no ROS2)
# ==========================
import numpy as np
import csv
import time
import matplotlib.pyplot as plt

def main():
    # --- Robot model ---
    params = FR3Params()
    mdl = FR3Model(params)
    dyn = FR3TaskDynamics(mdl, DynConfig(dt=0.1))  # slower dt is fine here
    mpc = UMPC(dyn, MPCConfig(S=20, dt=0.1, tau_min=-50.0, tau_max=50.0))

    # === Load and train DMP from demo ===
    file_path = "/home/asus/Documents/DMP_LfD/paper_roboface_demonstrated_joint_positions(t,j1-j7,x,y,z).csv"
    trajectories = load_and_augment_trajectories(file_path, n_aug=0)
    t_demo = trajectories[0]["t"].values
    x_demo = trajectories[0]["x"].values
    y_demo = trajectories[0]["y"].values
    z_demo = trajectories[0]["z"].values
    dmp_model = train_dmp_xyz(t_demo, x_demo, y_demo, z_demo, DMPConfig())

    # --- Initial state ---
    q0 = np.zeros(7)
    dq0 = np.zeros(7)
    p0 = mdl.fk_position(q0)
    v0 = mdl.jacobian_linear(q0) @ dq0
    x0 = np.concatenate([p0, v0, q0, dq0])

    # --- DMP rollout reference trajectory ---
    T_rollout, Xr, Yr, Zr = dmp_rollout(dmp_model, dt=dyn.cfg.dt)
    ref_traj = np.vstack([Xr, Yr, Zr]).T

    # --- Simulation ---
    num_steps = len(ref_traj) - mpc.cfg.S - 1
    u_prev = np.zeros(7)
    torque_log = []

    for t in range(num_steps):
        # if t % 20 == 0:
        print(f"Step {t}/{num_steps}")

        d_seq = ref_traj[t : t + mpc.cfg.S]

        # Solve MPC (no UKF, true state)
        u_cmd = mpc.solve(x0, np.zeros(28), d_seq, u_prev=u_prev)

        # Integrate true dynamics
        x0 = dyn.step(x0, u_cmd, np.zeros(28))

        # Log torque command
        torque_log.append(np.hstack([t * dyn.cfg.dt, u_cmd]))
        u_prev = u_cmd

    # --- Save torque log ---
    torque_log = np.array(torque_log)
    columns = ["time"] + [f"tau_{i+1}" for i in range(7)]
    pd.DataFrame(torque_log, columns=columns).to_csv("torque_log.csv", index=False)
    print(f"Simulation finished. Saved torque log to torque_log.csv")



if __name__ == "__main__":
    main()