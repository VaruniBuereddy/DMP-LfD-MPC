import numpy as np
import pandas as pd
from dataclasses import dataclass, asdict

import pandas as pd
import numpy as np

# Reuse the augmentation function from earlier
def load_and_augment_trajectories(file_path: str, n_aug: int = 5, noise_std: float = 0.002, shift_std: float = 0.01, seed: int = 0):
    rng = np.random.default_rng(seed)
    df = pd.read_csv(file_path)
    tcol = df.columns[0]
    xcol, ycol, zcol = df.columns[-3], df.columns[-2], df.columns[-1]
    base = df[[tcol, xcol, ycol, zcol]].copy()
    base.columns = ["t", "x", "y", "z"]
    t = base["t"].values
    xyz = base[["x", "y", "z"]].values

    trajectories = [base]
    for _ in range(n_aug):
        noise = rng.normal(0, noise_std, xyz.shape)
        shift = rng.normal(0, shift_std, (1, 3))
        aug_xyz = xyz + noise + shift
        aug_df = pd.DataFrame({"t": t, "x": aug_xyz[:,0], "y": aug_xyz[:,1], "z": aug_xyz[:,2]})
        trajectories.append(aug_df)

    return trajectories


# Load original + augmented trajectories
file_path = "/home/asus/Documents/DMP_LfD/paper_roboface_demonstrated_joint_positions(t,j1-j7,x,y,z).csv"
trajectories = load_and_augment_trajectories(file_path, n_aug=10, noise_std=0.001, shift_std=0.02)

# Extract t, x, y, z for the FIRST trajectory (original)
t = trajectories[0]["t"].values
x = trajectories[0]["x"].values
y = trajectories[0]["y"].values
z = trajectories[0]["z"].values

print("t shape:", t.shape)
print("x shape:", x.shape)
print("Example values:\n", trajectories[0].head())


# ======= Discrete DMP (Cartesian x,y,z) – FIT ONLY =======

@dataclass
class DMPModelXYZ:
    # Gains & timing
    alpha_s: float
    alpha_z: float
    beta_z: float
    tau: float  # duration (s)

    # Basis
    centers: np.ndarray  # (M,)
    widths:  np.ndarray  # (M,)

    # Weights per axis
    w_x: np.ndarray      # (M,)
    w_y: np.ndarray
    w_z: np.ndarray

    # Start & goal
    x0: np.ndarray       # (3,)
    g:  np.ndarray       # (3,)

    def to_dict(self):
        """Portable (json-friendly) dict."""
        d = asdict(self)
        for k, v in list(d.items()):
            if isinstance(v, np.ndarray):
                d[k] = v.tolist()
        return d


def _compute_s_from_time(t: np.ndarray, alpha_s: float, tau: float) -> np.ndarray:
    """Canonical system for discrete DMP: ds/dt = -alpha_s * s / tau, s(0)=1."""
    t = np.asarray(t) - t[0]
    return np.exp(-alpha_s * t / tau)

def _finite_diff(y: np.ndarray, t: np.ndarray) -> np.ndarray:
    """d/dt using numpy.gradient (supports non-uniform t)."""
    return np.gradient(y, t, edge_order=2)

def _make_rbfs(s: np.ndarray, M: int) -> tuple[np.ndarray, np.ndarray]:
    """Gaussian basis in canonical space (centers from 1→0.01, widths for good overlap)."""
    c = np.linspace(1.0, 0.01, M)
    dc = np.diff(c).mean()
    h = np.ones(M) * (1.0 / (2.0 * (dc**2)))  # neighbor overlap ~0.5
    return c, h

def _design_matrix(s: np.ndarray, centers: np.ndarray, widths: np.ndarray) -> np.ndarray:
    """Normalized activations φ_i(s) = (ψ_i(s) * s) / Σ_j ψ_j(s) (enforces f(0)=0)."""
    psi = np.exp(-widths * (s[:, None] - centers[None, :])**2)
    denom = np.sum(psi, axis=1, keepdims=True) + 1e-12
    return (psi / denom) * s[:, None]

def train_dmp_xyz(t: np.ndarray, x: np.ndarray, y: np.ndarray, z: np.ndarray,
                  n_basis: int = 25,
                  alpha_s: float = 4.0,
                  alpha_z: float = 25.0,
                  beta_ratio: float = 0.25,
                  ridge: float = 1e-6) -> DMPModelXYZ:
    """
    Fit a discrete DMP in Cartesian space (x,y,z) to a single demonstration.

    Returns:
        DMPModelXYZ (weights, centers, widths, gains, tau, start, goal)
    """
    t = np.asarray(t, float)
    x = np.asarray(x, float); y = np.asarray(y, float); z = np.asarray(z, float)

    tau = float(t[-1] - t[0])
    if tau <= 0:
        raise ValueError("Time vector must be strictly increasing.")

    beta_z = alpha_z * beta_ratio
    s = _compute_s_from_time(t, alpha_s, tau)

    # Derivatives
    xd = _finite_diff(x, t); yd = _finite_diff(y, t); zd = _finite_diff(z, t)
    xdd = _finite_diff(xd, t); ydd = _finite_diff(yd, t); zdd = _finite_diff(zd, t)

    # Start & goal
    x0 = np.array([x[0], y[0], z[0]])
    g  = np.array([x[-1], y[-1], z[-1]])

    # Forcing targets from discrete DMP dynamics:
    # τ² ẍ = αz (βz (g − x) − τ ẋ) + f(s)  ⇒  f_target = τ² ẍ − αz (βz (g − x) − τ ẋ)
    fx = tau**2 * xdd - alpha_z * (beta_z * (g[0] - x) - tau * xd)
    fy = tau**2 * ydd - alpha_z * (beta_z * (g[1] - y) - tau * yd)
    fz = tau**2 * zdd - alpha_z * (beta_z * (g[2] - z) - tau * zd)

    centers, widths = _make_rbfs(s, n_basis)
    Phi = _design_matrix(s, centers, widths)  # (N, M)

    # Ridge-regularized least squares: w = (ΦᵀΦ + λI)⁻¹ Φᵀ f
    def _solve_w(f_target: np.ndarray) -> np.ndarray:
        A = Phi.T @ Phi + ridge * np.eye(Phi.shape[1])
        b = Phi.T @ f_target
        return np.linalg.solve(A, b)

    w_x = _solve_w(fx)
    w_y = _solve_w(fy)
    w_z = _solve_w(fz)

    return DMPModelXYZ(
        alpha_s=alpha_s, alpha_z=alpha_z, beta_z=beta_z, tau=tau,
        centers=centers, widths=widths,
        w_x=w_x, w_y=w_y, w_z=w_z,
        x0=x0, g=g
    )

import numpy as np
import matplotlib.pyplot as plt

def dmp_rollout(model, dt, goal=None, tau=None, n_steps=None):
    """
    Roll out a discrete DMP in Cartesian space (x,y,z).
    Args:
        model: trained DMPModelXYZ
        dt: time step
        goal: optional new goal [x,y,z]
        tau: optional new duration (time scaling)
        n_steps: number of rollout steps (if None, based on model.tau/dt)
    Returns:
        t, x, y, z arrays of shape (n_steps,)
    """
    # Unpack model
    alpha_s = model.alpha_s
    alpha_z = model.alpha_z
    beta_z = model.beta_z
    centers = model.centers
    widths = model.widths
    w_x, w_y, w_z = model.w_x, model.w_y, model.w_z
    x0 = model.x0
    g = model.g if goal is None else np.array(goal)
    tau = model.tau if tau is None else tau

    if n_steps is None:
        n_steps = int(np.round(tau / dt)) + 1

    # Init states
    s = 1.0
    x = x0.copy()
    v = np.zeros(3)

    # Trajectory arrays
    T = np.linspace(0, tau, n_steps)
    X = np.zeros((n_steps, 3))
    X[0] = x

    def forcing_term(s_val, w):
        psi = np.exp(-widths * (s_val - centers)**2)
        psi_sum = np.sum(psi) + 1e-12
        f = (np.dot(psi, w) * s_val) / psi_sum
        return f

    for i in range(1, n_steps):
        # Forcing terms
        fx = forcing_term(s, w_x)
        fy = forcing_term(s, w_y)
        fz = forcing_term(s, w_z)
        f = np.array([fx, fy, fz])

        # Transformation system
        a = alpha_z * (beta_z * (g - x) - v) + f
        v += (a * dt) / tau
        x += (v * dt) / tau

        # Canonical system
        s += -alpha_s * s * dt / tau

        X[i] = x

    return T, X[:, 0], X[:, 1], X[:, 2]


def plot_reconstruction(t_demo, x_demo, y_demo, z_demo, t_recon, x_recon, y_recon, z_recon):
    """Plot demonstration vs DMP reconstruction in x/y/z subplots."""
    fig, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    axs[0].plot(t_demo, x_demo, 'k-', label='demo')
    axs[0].plot(t_recon, x_recon, 'r--', label='recon')
    axs[1].plot(t_demo, y_demo, 'k-')
    axs[1].plot(t_recon, y_recon, 'r--')
    axs[2].plot(t_demo, z_demo, 'k-')
    axs[2].plot(t_recon, z_recon, 'r--')
    axs[0].set_ylabel('X (m)')
    axs[1].set_ylabel('Y (m)')
    axs[2].set_ylabel('Z (m)')
    axs[2].set_xlabel('Time (s)')
    axs[0].legend()
    plt.tight_layout()
    plt.show()


# t, x, y, z are from your original demonstration
model = train_dmp_xyz(t, x, y, z, n_basis=150)

# Roll out with original goal
dt = np.mean(np.diff(t))
t_recon, x_recon, y_recon, z_recon = dmp_rollout(model, dt)
ref_traj = np.vstack([x_recon, y_recon, z_recon]).T  # (N, 3)
print(f"ref_traj shape: {ref_traj.shape}")
# Plot
plot_reconstruction(t, x, y, z, t_recon, x_recon, y_recon, z_recon)

# # Retimed (slower, 2x time)
# t_recon2, x_recon2, y_recon2, z_recon2 = dmp_rollout(model, dt, tau=model.tau*2)
# plot_reconstruction(t, x, y, z, t_recon2, x_recon2, y_recon2, z_recon2)

# # New goal position
# new_goal = model.g + np.array([0.05, 0.02, 0])  # shift 5cm in x, 2cm in y
# t_recon3, x_recon3, y_recon3, z_recon3 = dmp_rollout(model, dt, goal=new_goal)
# plot_reconstruction(t, x, y, z, t_recon3, x_recon3, y_recon3, z_recon3)


dmp_data = np.hstack([t_recon.reshape(-1, 1), ref_traj])
header = "time,x_desired,y_desired,z_desired"
np.savetxt("desired_dmp_traj.csv", dmp_data, delimiter=",", header=header, comments='')

print(f"Saved desired trajectory to desired_dmp_traj.csv with shape {dmp_data.shape}")
