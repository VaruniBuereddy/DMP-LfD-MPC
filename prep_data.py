import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def load_and_augment_trajectories(
    file_path: str,
    n_aug: int = 5,
    noise_std: float = 0.002,
    shift_std: float = 0.01,   # ~1 cm lateral shift
    seed: int = 0
):
    """
    Reads a CSV with columns [t, ... , x, y, z], extracts t,x,y,z,
    generates `n_aug` noisy + shifted trajectories, and plots them.

    Returns:
        trajectories: list of DataFrames, original first then augmented ones.
    """
    rng = np.random.default_rng(seed)

    # Load CSV and extract relevant columns
    df = pd.read_csv(file_path)
    tcol = df.columns[0]
    xcol, ycol, zcol = df.columns[-3], df.columns[-2], df.columns[-1]
    traj = df[[tcol, xcol, ycol, zcol]].copy()
    traj.columns = ["t", "x", "y", "z"]
    t = traj["t"].values
    xyz = traj[["x", "y", "z"]].values

    trajectories = [traj]

    # Augment trajectories with noise and shift
    for _ in range(n_aug):
        # Add random noise
        noise = rng.normal(0, noise_std, xyz.shape)
        aug_xyz = xyz + noise

        # Add lateral shift (translation of whole trajectory)
        shift = rng.normal(0, shift_std, (1, 3))
        aug_xyz = aug_xyz + shift

        aug_df = pd.DataFrame({"t": t, "x": aug_xyz[:, 0], "y": aug_xyz[:, 1], "z": aug_xyz[:, 2]})
        trajectories.append(aug_df)

    # Plot all trajectories
    fig, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    labels = ["Original"] + [f"Aug {i+1}" for i in range(n_aug)]
    colors = plt.cm.viridis(np.linspace(0, 1, len(trajectories)))

    for idx, tr in enumerate(trajectories):
        axs[0].plot(tr["t"], tr["x"], label=labels[idx], color=colors[idx])
        axs[1].plot(tr["t"], tr["y"], color=colors[idx])
        axs[2].plot(tr["t"], tr["z"], color=colors[idx])

    axs[0].set_ylabel("X (m)")
    axs[1].set_ylabel("Y (m)")
    axs[2].set_ylabel("Z (m)")
    axs[2].set_xlabel("Time (s)")
    axs[0].legend()
    plt.tight_layout()
    plt.show()

    return trajectories

trajectories = load_and_augment_trajectories(
    "/home/asus/Documents/DMP_LfD/paper_roboface_demonstrated_joint_positions(t,j1-j7,x,y,z).csv",
    n_aug=10,
    noise_std=0.001,
    shift_std=0.02   # ~2 cm shifts
)
