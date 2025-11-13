from dataclasses import dataclass, asdict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# -------------------- HELPER FUNCTIONS --------------------

@dataclass
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

# def load_and_augment_trajectories(file_path: str, n_aug: int = 5, noise_std: float = 0.002, shift_std: float = 0.01, seed: int = 0):
#     rng = np.random.default_rng(seed)
#     df = pd.read_csv(file_path)
#     tcol = df.columns[0]
#     xcol, ycol, zcol = df.columns[-3], df.columns[-2], df.columns[-1]
#     base = df[[tcol, xcol, ycol, zcol]].copy()
#     base.columns = ["t", "x", "y", "z"]
#     t = base["t"].values
#     xyz = base[["x", "y", "z"]].values

#     trajectories = [base]
#     for _ in range(n_aug):
#         noise = rng.normal(0, noise_std, xyz.shape)
#         shift = rng.normal(0, shift_std, (1, 3))
#         aug_xyz = xyz + noise + shift
#         aug_df = pd.DataFrame({"t": t, "x": aug_xyz[:,0], "y": aug_xyz[:,1], "z": aug_xyz[:,2]})
#         trajectories.append(aug_df)

#     return trajectories

def load_and_augment_trajectories(
    file_path: str,
    n_aug: int = 5,
    noise_std: float = 0.002,
    shift_std: float = 0.01,
    seed: int = 0,
):
    """
    Reads a trajectory CSV file and generates n_aug noisy variants.
    Expects columns named like: time, t, x, y, z (case-insensitive).
    """
    rng = np.random.default_rng(seed)
    df = pd.read_csv(file_path)
    cols = [c.lower().strip() for c in df.columns]

    # ---- Identify time and xyz columns robustly ----
    # Accept common naming variants
    time_candidates = ["time", "t", "timestamp"]
    x_candidates = ["x", "pos_x", "position_x"]
    y_candidates = ["y", "pos_y", "position_y"]
    z_candidates = ["z", "pos_z", "position_z"]

    def find_col(candidates):
        for c in candidates:
            for existing in cols:
                if existing == c or existing.endswith(c):
                    return df.columns[cols.index(existing)]
        raise KeyError(f"Could not find any of {candidates} in CSV header {df.columns.tolist()}")

    tcol = find_col(time_candidates)
    xcol = find_col(x_candidates)
    ycol = find_col(y_candidates)
    zcol = find_col(z_candidates)

    # ---- Build base trajectory ----
    base = df[[tcol, xcol, ycol, zcol]].copy()
    base.columns = ["t", "x", "y", "z"]
    t = base["t"].values
    xyz = base[["x", "y", "z"]].values

    # ---- Augment ----
    trajectories = [base]
    for _ in range(n_aug):
        noise = rng.normal(0, noise_std, xyz.shape)
        shift = rng.normal(0, shift_std, (1, 3))
        aug_xyz = xyz + noise + shift
        aug_df = pd.DataFrame({"t": t, "x": aug_xyz[:, 0], "y": aug_xyz[:, 1], "z": aug_xyz[:, 2]})
        trajectories.append(aug_df)

    print(f"[INFO] Loaded trajectory with columns: t={tcol}, x={xcol}, y={ycol}, z={zcol}")
    print(f"[INFO] Generated {n_aug} augmented trajectories")
    return trajectories



def _compute_s_from_time(t: np.ndarray, alpha_s: float, tau: float) -> np.ndarray:
    t = np.asarray(t) - t[0]
    return np.exp(-alpha_s * t / tau)

def _finite_diff(y: np.ndarray, t: np.ndarray) -> np.ndarray:
    return np.gradient(y, t, edge_order=2)

def _make_rbfs(M: int) -> tuple[np.ndarray, np.ndarray]:
    c = np.linspace(1.0, 0.01, M)
    dc = np.diff(c).mean()
    h = np.ones(M) * (1.0 / (2.0 * (dc ** 2)))
    return c, h

def _design_matrix(s: np.ndarray, centers: np.ndarray, widths: np.ndarray) -> np.ndarray:
    psi = np.exp(-widths * (s[:, None] - centers[None, :])**2)
    denom = np.sum(psi, axis=1, keepdims=True) + 1e-12
    return (psi / denom) * s[:, None]

# -------------------- TRAINING --------------------

def train_dmp_xyz(t: np.ndarray, x: np.ndarray, y: np.ndarray, z: np.ndarray, cfg: DMPConfig) -> DMPModelXYZ:
    t = np.asarray(t, float)
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    z = np.asarray(z, float)

    tau = float(t[-1] - t[0])
    if tau <= 0:
        raise ValueError("Time vector must be strictly increasing.")

    alpha_s = cfg.alpha_s
    alpha_z = cfg.alpha_z
    beta_z = alpha_z * cfg.beta_ratio

    s = _compute_s_from_time(t, alpha_s, tau)

    xd = _finite_diff(x, t); yd = _finite_diff(y, t); zd = _finite_diff(z, t)
    xdd = _finite_diff(xd, t); ydd = _finite_diff(yd, t); zdd = _finite_diff(zd, t)

    x0 = np.array([x[0], y[0], z[0]])
    g  = np.array([x[-1], y[-1], z[-1]])

    fx = tau**2 * xdd - alpha_z * (beta_z * (g[0] - x) - tau * xd)
    fy = tau**2 * ydd - alpha_z * (beta_z * (g[1] - y) - tau * yd)
    fz = tau**2 * zdd - alpha_z * (beta_z * (g[2] - z) - tau * zd)

    centers, widths = _make_rbfs(cfg.n_basis)
    Phi = _design_matrix(s, centers, widths)

    def _solve_w(f_target):
        A = Phi.T @ Phi + cfg.ridge * np.eye(Phi.shape[1])
        b = Phi.T @ f_target
        return np.linalg.solve(A, b)

    w_x = _solve_w(fx)
    w_y = _solve_w(fy)
    w_z = _solve_w(fz)

    return DMPModelXYZ(alpha_s, alpha_z, beta_z, tau, centers, widths, w_x, w_y, w_z, x0, g)

# -------------------- ROLLOUT --------------------

def dmp_rollout(model: DMPModelXYZ, dt: float, goal: np.ndarray = None, tau: float = None, n_steps: int = None):
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

    s = 1.0
    x = x0.copy()
    v = np.zeros(3)
    T = np.linspace(0, tau, n_steps)
    X = np.zeros((n_steps, 3))
    X[0] = x

    def forcing_term(s_val, w):
        psi = np.exp(-widths * (s_val - centers)**2)
        psi_sum = np.sum(psi) + 1e-12
        return (np.dot(psi, w) * s_val) / psi_sum

    for i in range(1, n_steps):
        f = np.array([forcing_term(s, w_x),
                      forcing_term(s, w_y),
                      forcing_term(s, w_z)])
        a = alpha_z * (beta_z * (g - x) - v) + f
        v += (a * dt) / tau
        x += (v * dt) / tau
        s += -alpha_s * s * dt / tau
        X[i] = x

    return T, X[:, 0], X[:, 1], X[:, 2]

def dmp_rollout_on_time(model: DMPModelXYZ, t: np.ndarray, goal: np.ndarray = None, tau: float = None):
    """
    Roll out the DMP using the original time vector t (variable dt each step).
    The output length and timing match the demonstration exactly.
    """
    t = np.asarray(t, float)
    assert np.all(np.diff(t) > 0), "t must be strictly increasing"

    alpha_s = model.alpha_s
    alpha_z = model.alpha_z
    beta_z  = model.beta_z
    centers = model.centers
    widths  = model.widths
    w_x, w_y, w_z = model.w_x, model.w_y, model.w_z
    x0 = model.x0
    g  = model.g if goal is None else np.array(goal, float)
    tau = float(t[-1] - t[0]) if tau is None else float(tau)

    # init
    s = 1.0
    x = x0.copy()
    v = np.zeros(3)

    T = t - t[0]
    X = np.zeros((t.size, 3))
    X[0] = x

    def forcing_term(s_val, w):
        psi = np.exp(-widths * (s_val - centers) ** 2)
        psi_sum = np.sum(psi) + 1e-12
        return (np.dot(psi, w) * s_val) / psi_sum

    for i in range(1, t.size):
        dt_i = t[i] - t[i-1]
        f = np.array([forcing_term(s, w_x),
                      forcing_term(s, w_y),
                      forcing_term(s, w_z)])
        a = alpha_z * (beta_z * (g - x) - v) + f
        v += (a * dt_i) / tau
        x += (v * dt_i) / tau
        s += -alpha_s * s * dt_i / tau
        X[i] = x

    return T, X[:,0], X[:,1], X[:,2]

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