
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
    Discrete auxiliary model used inside MPC (clean double integrator):
        x_aux = [X; V] ∈ R^6,  u_aux ∈ R^3
        X_{k+1} = X_k + dt * V_k + 0.5 * dt^2 * u_k
        V_{k+1} = V_k + dt * u_k
    """
    def __init__(self, dt: float):
        self.dt = float(dt)
        self.nx = 6   # [X(3), V(3)]
        self.nu = 3   # u_aux (3)
        # Selectors
        self.E  = np.hstack([np.eye(3), np.zeros((3,3))])  # position
        self.Ev = np.hstack([np.zeros((3,3)), np.eye(3)])  # velocity

        # Precompute A, B for the fixed dt
        I3 = np.eye(3)
        Z3 = np.zeros((3,3))
        dt = self.dt

        self.A = np.block([
            [I3, dt * I3],
            [Z3, I3]
        ])  # (6x6)

        self.B = np.vstack([
            0.5 * (dt ** 2) * I3,
            dt * I3
        ])  # (6x3)

    def AB(self):
        """Return (A, B) of the discrete double-integrator."""
        return self.A, self.B

    def step(self, x_aux: np.ndarray, u_aux: np.ndarray) -> np.ndarray:
        """One-step propagate x_{k+1} = A x_k + B u_k."""
        return self.A @ x_aux + self.B @ u_aux


from dataclasses import dataclass, asdict
import numpy as np
import pandas as pd

# -------------------- CONFIG --------------------

@dataclass
class DMPConfig:
    alpha_s: float = 4.0
    alpha_z: float = 25.0
    beta_ratio: float = 0.25
    n_basis: int = 100
    ridge: float = 1e-6

class DMPConfig:
    alpha_s: float = 4.0
    alpha_z: float = 25.0
    beta_ratio: float = 0.25
    n_basis: int = 50
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


class MPC_AUX:
    """
    MPC on the auxiliary linear model (Ẍ = u_aux).
    Decision vars: u_aux[k] ∈ R^3.
    Cost: position tracking (+ optional velocity tracking) + small effort on u_aux + rate smoothing.
    OUTPUT: u_aux_cmd (3,)  <-- end-effector acceleration command at current step.
            (optionally) predicted X (S+1,6) and U (S,3) for plotting/debug.
    """
    def __init__(self, cfg: MPCConfig):
        self.cfg = cfg
        self.aux = AuxDynamics(cfg.dt)

        # projected-gradient hyperparams
        self.max_iters  = 50      # Increased for better convergence
        self.alpha_step = 1e-1    # Larger initial step size
        self.bt_beta    = 0.5
        self.bt_c       = 1e-4

        # weights
        self.w_Ru  = cfg.Rtau       # effort on u_aux
        self.w_Rdu = 1e-3           # rate smoothing on u_aux
        self.w_Qv  = getattr(cfg, "Qv", 0.0)  # OPTIONAL velocity tracking weight

        # per-axis acceleration bounds (m/s^2)
        self.u_min = -2.0 * np.ones(3)
        self.u_max =  2.0 * np.ones(3)

    # ---------- internal: adjoint gradient ----------
    def _grad_adj(self, A: np.ndarray, B: np.ndarray,
                  X: np.ndarray, U: np.ndarray,
                  d_seq: np.ndarray, v_seq: Optional[np.ndarray]) -> np.ndarray:
        S = U.shape[0]
        E = self.aux.E  # (3x6) position selector
        Ev = self.aux.Ev  # (3x6) velocity selector (define in AuxDynamics: Ev = [0 I])

        grad = np.zeros_like(U)
        lam = np.zeros((S+1, 6))

        # terminal costate
        eT = E @ X[S] - d_seq[-1]
        lam[S] = 2.0 * self.cfg.Qterm * (E.T @ eT)
        # (no terminal vel term by default; add if you want)

        # backward pass
        for k in reversed(range(S)):
            # dJ/du_k: effort + rate regularization
            gk = 2.0 * self.w_Ru * U[k]
            if k > 0:
                gk += 2.0 * self.w_Rdu * (U[k] - U[k-1])
            if k < S-1:
                gk -= 2.0 * self.w_Rdu * (U[k+1] - U[k])

            # dynamics sensitivity
            grad[k] = gk + B.T @ lam[k+1]

            # propagate costate (position + optional velocity tracking)
            ek_p = E @ X[k] - d_seq[k]
            lam_term = 2.0 * self.cfg.Qp * (E.T @ ek_p)

            if (v_seq is not None) and (self.w_Qv > 0.0):
                ek_v = Ev @ X[k] - v_seq[k]
                lam_term += 2.0 * self.w_Qv * (Ev.T @ ek_v)

            lam[k] = (A.T @ lam[k+1]) + lam_term

        return grad

    # ---------- public solve ----------
    def solve(self,
              x_aux: np.ndarray,                 # [X, V] ∈ R^6
              d_seq: np.ndarray,                 # (S,3) desired EE positions
              v_seq: Optional[np.ndarray] = None,# (S,3) desired EE velocities (optional)
              u_prev: Optional[np.ndarray] = None,
              return_rollout: bool = False,
              verbose: bool = False):
        """
        Returns:
            u_aux_cmd : np.ndarray, shape (3,)   # EE acceleration command at current step
            (X_pred, U_opt) if return_rollout=True
        """
        S = self.cfg.S
        x_aux0 = x_aux.copy()  # (6,)
        A, B = self.aux.AB()   # nominal; no θ
        E = self.aux.E

        # warm start
        if u_prev is None:
            U = np.zeros((S, 3))
        else:
            U = np.tile(u_prev[:3], (S, 1))
        U = np.clip(U, self.u_min, self.u_max)

        # projected gradient with Armijo
        cost_history = []
        for iter_num in range(self.max_iters):
            # rollout
            X = np.zeros((S+1, 6))
            X[0] = x_aux0
            for k in range(S):
                X[k+1] = A @ X[k] + B @ U[k]

            # cost
            cost = 0.0
            for k in range(S):
                e_p = E @ X[k] - d_seq[k]
                cost += self.cfg.Qp * (e_p @ e_p) + self.w_Ru * (U[k] @ U[k])
                if (v_seq is not None) and (self.w_Qv > 0.0):
                    e_v = self.aux.Ev @ X[k] - v_seq[k]
                    cost += self.w_Qv * (e_v @ e_v)
                if k > 0:
                    du = U[k] - U[k-1]
                    cost += self.w_Rdu * (du @ du)
            eT = E @ X[S] - d_seq[-1]
            cost += self.cfg.Qterm * (eT @ eT)

            cost_history.append(cost)

            # Check convergence
            if iter_num > 0:
                cost_reduction = cost_history[-2] - cost_history[-1]
                if verbose and iter_num % 10 == 0:
                    print(f"  Iter {iter_num}: cost={cost:.6f}, reduction={cost_reduction:.6e}")
                if cost_reduction < 1e-6 and cost_reduction >= 0:
                    if verbose:
                        print(f"  Converged at iteration {iter_num}")
                    break

            # gradient
            grad = self._grad_adj(A, B, X, U, d_seq, v_seq)

            # Armijo
            step = self.alpha_step
            base_quad = np.sum(grad * grad)
            while True:
                U_trial = np.clip(U - step * grad, self.u_min, self.u_max)

                # rollout trial & cost
                Xtr = np.zeros((S+1, 6)); Xtr[0] = x_aux0
                for k in range(S):
                    Xtr[k+1] = A @ Xtr[k] + B @ U_trial[k]

                cost_tr = 0.0
                for k in range(S):
                    e_p = E @ Xtr[k] - d_seq[k]
                    cost_tr += self.cfg.Qp * (e_p @ e_p) + self.w_Ru * (U_trial[k] @ U_trial[k])
                    if (v_seq is not None) and (self.w_Qv > 0.0):
                        e_v = self.aux.Ev @ Xtr[k] - v_seq[k]
                        cost_tr += self.w_Qv * (e_v @ e_v)
                    if k > 0:
                        du = U_trial[k] - U_trial[k-1]
                        cost_tr += self.w_Rdu * (du @ du)
                eT = E @ Xtr[S] - d_seq[-1]
                cost_tr += self.cfg.Qterm * (eT @ eT)

                if cost_tr <= cost - self.bt_c * step * base_quad:
                    U = U_trial
                    X = Xtr
                    break
                step *= self.bt_beta
                if step < 1e-8:
                    break

        u_aux_cmd = U[0].copy()  # end-effector acceleration for this step

        if return_rollout:
            return u_aux_cmd, X, U
        return u_aux_cmd


    
import numpy as np
import csv
import time
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def main():
    # -------------------------
    # 1) Load demo & train DMP
    # -------------------------
    file_path = "/home/iisc-hiro-lab-phd-2/Documents/DMP_LfD/STT_tube_ee_trajectory_adjusted.csv"
    trajectories = load_and_augment_trajectories(file_path, n_aug=0)
    t_demo = trajectories[0]["t"].values.astype(float)
    x_demo = trajectories[0]["x"].values.astype(float)
    y_demo = trajectories[0]["y"].values.astype(float)
    z_demo = trajectories[0]["z"].values.astype(float)
    

    # dt to MATCH DMP rollout
    dt = float(np.mean(np.diff(t_demo)))
    # Train DMP (task-space)
    dmp_model = train_dmp_xyz(t_demo, x_demo, y_demo, z_demo, DMPConfig())

    # Rollout DMP with same dt and duration
    tau_demo = float(t_demo[-1] - t_demo[0])
    T_ref, Xr, Yr, Zr = dmp_rollout_on_time(dmp_model, t_demo)
    ref_pos = np.vstack([Xr, Yr, Zr]).T                      # (N,3)
    # Desired velocities for tracking/plots
    ref_vel = np.gradient(ref_pos, dt, axis=0)               # (N,3)
    N = ref_pos.shape[0]
    
    print("t_demo[0], t_demo[-1] =", t_demo[0], t_demo[-1])
    print("T_ref[0], T_ref[-1]     =", T_ref[0], T_ref[-1])
    # Save desired traj (pos + vel)
    # desired_out = np.hstack([T_ref.reshape(-1,1), ref_pos, ref_vel])
    # np.savetxt("desired_dmp_traj.csv", desired_out, delimiter=",",
    #            header="time,x_des,y_des,z_des,vx_des,vy_des,vz_des", comments='')
    # print(f"[save] desired_dmp_traj.csv  shape={desired_out.shape}")

    # # ------------------------------------------------
    # 2) Build MPC on the auxiliary linear model only
    # ------------------------------------------------
    # Config: horizon, weights, dt
    S = 30  # Increased horizon for better trajectory preview
    mpc_cfg = MPCConfig(
        S=S,
        dt=dt,
        Qp=10000.0,      # Much higher position tracking weight
        Qterm=15000.0,   # Much higher terminal position weight
        Rtau=1e-6        # Very small control effort weight for aggressive tracking
    )
    # Instantiate auxiliary-only MPC (the class that RETURNS u_aux, not torque)
    mpc = MPC_AUX(mpc_cfg)

    # Set more relaxed accel bounds (m/s^2)
    mpc.u_min = -8.0 * np.ones(3)
    mpc.u_max =  8.0 * np.ones(3)

    # Enable velocity tracking for better trajectory following
    mpc.w_Qv = 500.0    # Higher velocity tracking weight
    mpc.w_Rdu = 1e-6    # Very low rate smoothing for aggressive tracking

    # -----------------------------
    # 3) Receding-horizon simulation
    # -----------------------------
    # Initial aux state: position & velocity from DMP start
    x_aux = np.hstack([ref_pos[0], ref_vel[0]])  # [X0,V0] ∈ R^6

    # Logs
    t_log = []
    x_log = []    # executed positions
    v_log = []    # executed velocities
    u_log = []    # applied accelerations

    steps = len(T_ref) - 1  # We run until len(T_ref)-1 because we look ahead
    tracking_errors = []

    # Get A, B matrices once
    if hasattr(mpc.aux, "AB"):
        A, B = mpc.aux.AB()
    elif hasattr(mpc.aux, "A") and hasattr(mpc.aux, "B"):
        A, B = mpc.aux.A, mpc.aux.B
    else:
        raise RuntimeError("Aux dynamics must expose AB() or A,B.")

    for k in range(steps):
        if k % 100 == 0:
            if tracking_errors:
                avg_error = np.mean(tracking_errors[-100:])
                print(f"Step {k}/{steps}, Avg tracking error (last 100 steps): {avg_error:.6f} m")

        # Current desired state: we want to be at ref_pos[k] now
        current_error = np.linalg.norm(x_aux[:3] - ref_pos[k])

        # Horizon slice (positions and velocities for tracking)
        # d_seq[0] is where we want to be now, d_seq[1] is next step, etc.
        d_seq = ref_pos[k : k + S]
        v_seq = ref_vel[k : k + S]

        # Pad if necessary
        if d_seq.shape[0] < S:
            d_seq = np.vstack([d_seq, np.repeat(d_seq[-1:], S - d_seq.shape[0], axis=0)])
            v_seq = np.vstack([v_seq, np.repeat(v_seq[-1:], S - v_seq.shape[0], axis=0)])

        # Solve MPC: find control that minimizes tracking error over horizon
        verbose = (k == 0)  # Print convergence info for first step only
        u0 = mpc.solve(x_aux, d_seq, v_seq=v_seq, u_prev=(u_log[-1] if len(u_log) else None), verbose=verbose)

        # Log current state before applying control
        t_log.append(T_ref[k])
        x_log.append(x_aux[:3].copy())
        v_log.append(x_aux[3:].copy())
        u_log.append(u0.copy())
        tracking_errors.append(current_error)

        # Apply control: move to next state
        x_aux = A @ x_aux + B @ u0

    # Log final state
    t_log.append(T_ref[-1])
    x_log.append(x_aux[:3].copy())
    v_log.append(x_aux[3:].copy())
    u_log.append(u0.copy())  # Same control as last step
    final_error = np.linalg.norm(x_aux[:3] - ref_pos[-1])
    tracking_errors.append(final_error)

    t_log = np.array(t_log)
    x_log = np.array(x_log)
    v_log = np.array(v_log)
    u_log = np.array(u_log)
    tracking_errors = np.array(tracking_errors)

    # Print tracking statistics
    print("\n" + "="*60)
    print("TRACKING PERFORMANCE SUMMARY")
    print("="*60)
    print(f"Mean tracking error:    {np.mean(tracking_errors):.6f} m")
    print(f"Max tracking error:     {np.max(tracking_errors):.6f} m")
    print(f"Std tracking error:     {np.std(tracking_errors):.6f} m")
    print(f"Final tracking error:   {tracking_errors[-1]:.6f} m")
    print("="*60 + "\n")

    # Save DMP desired trajectory (not MPC executed)
    # Compute desired accelerations from velocities
    ref_acc = np.gradient(ref_vel, dt, axis=0)  # (N,3)

    results_out = np.hstack([
        T_ref.reshape(-1, 1),   # time
        ref_pos,                # x, y, z (DMP desired)
        ref_vel,                # xdot, ydot, zdot (DMP desired)
        ref_acc                 # xddot, yddot, zddot (DMP desired)
    ])

    np.savetxt(
        "./W_results/dmp_umpc_results.csv",
        results_out,
        delimiter=",",
        header="time,x,y,z,xdot,ydot,zdot,xddot,yddot,zddot",
        comments=''
    )

    print(f"[save] dmp_umpc_results.csv (DMP desired trajectory)  shape={results_out.shape}")
    # -----------------
    # 5) Quick plotting
    # -----------------
    # Align desired arrays for plotting (truncate to logged length)
    ref_pos_plot = ref_pos[:len(x_log)]
    ref_vel_plot = ref_vel[:len(v_log)]
    print('-----------------------------------------------')

    print("t_demo[0], t_demo[-1] =", t_demo[0], t_demo[-1])
    print("T_log[0], T_log[-1]     =", t_log[0], t_log[-1])
    t_demo0 = t_demo - t_demo[0]
    t_log0  = t_log - t_log[0]
    print("t_demo[0], t_demo[-1] =", t_demo[0], t_demo[-1])
    print("T_log[0], T_log[-1]     =", t_log0[0], t_log0[-1])

    # ------------------------------------------
    # Position tracking (XYZ vs time) + Demo traj
    # ------------------------------------------
    plt.figure(figsize=(10, 8))

    # X-axis
    ax = plt.subplot(3, 1, 1)
    ax.plot(t_demo0, x_demo, 'g:', label='x_demo')            # demonstrated
    ax.plot(t_log0, ref_pos_plot[:, 0], 'r--', label='x_des') # desired from DMP
    ax.plot(t_log0, x_log[:, 0], 'b-', label='x_mpc')         # MPC executed
    ax.legend(); ax.grid(True)

    # Y-axis
    ax = plt.subplot(3, 1, 2)
    ax.plot(t_demo0, y_demo, 'g:', label='y_demo')
    ax.plot(t_log0, ref_pos_plot[:, 1], 'r--', label='y_des')
    ax.plot(t_log0, x_log[:, 1], 'b-', label='y_mpc')
    ax.legend(); ax.grid(True)

    # Z-axis
    ax = plt.subplot(3, 1, 3)
    ax.plot(t_demo, z_demo, 'g:', label='z_demo')
    ax.plot(t_log0, ref_pos_plot[:, 2], 'r--', label='z_des')
    ax.plot(t_log0, x_log[:, 2], 'b-', label='z_mpc')
    ax.legend(); ax.grid(True)

    plt.suptitle("End-Effector Position Tracking (Demo vs DMP vs MPC)")
    plt.tight_layout()
    plt.show()

    # ------------------------------------------
    # 3D Trajectory Comparison: DMP vs Reference STT
    # ------------------------------------------
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Plot reference STT trajectory
    ax.plot(x_demo, y_demo, z_demo, 'g-', linewidth=2, label='Reference STT Trajectory')
    ax.scatter(x_demo[0], y_demo[0], z_demo[0], c='green', marker='o', s=100, label='STT Start')
    ax.scatter(x_demo[-1], y_demo[-1], z_demo[-1], c='darkgreen', marker='x', s=100, label='STT End')

    # Plot DMP-generated trajectory
    ax.plot(ref_pos_plot[:, 0], ref_pos_plot[:, 1], ref_pos_plot[:, 2], 'r--', linewidth=2, label='DMP Desired Trajectory')
    ax.scatter(ref_pos_plot[0, 0], ref_pos_plot[0, 1], ref_pos_plot[0, 2], c='red', marker='o', s=100, label='DMP Start')
    ax.scatter(ref_pos_plot[-1, 0], ref_pos_plot[-1, 1], ref_pos_plot[-1, 2], c='darkred', marker='x', s=100, label='DMP End')

    # Plot MPC-executed trajectory
    ax.plot(x_log[:, 0], x_log[:, 1], x_log[:, 2], 'b-', linewidth=1.5, alpha=0.7, label='MPC Executed Trajectory')
    ax.scatter(x_log[0, 0], x_log[0, 1], x_log[0, 2], c='blue', marker='o', s=100, label='MPC Start')
    ax.scatter(x_log[-1, 0], x_log[-1, 1], x_log[-1, 2], c='darkblue', marker='x', s=100, label='MPC End')

    ax.set_xlabel('X (m)', fontsize=12)
    ax.set_ylabel('Y (m)', fontsize=12)
    ax.set_zlabel('Z (m)', fontsize=12)
    ax.set_title('3D End-Effector Trajectory Comparison', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True)

    # Set equal aspect ratio for better visualization
    max_range = np.array([x_demo.max()-x_demo.min(),
                          y_demo.max()-y_demo.min(),
                          z_demo.max()-z_demo.min()]).max() / 2.0

    mid_x = (x_demo.max()+x_demo.min()) * 0.5
    mid_y = (y_demo.max()+y_demo.min()) * 0.5
    mid_z = (z_demo.max()+z_demo.min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()