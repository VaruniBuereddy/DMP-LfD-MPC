import numpy as np
import matplotlib.pyplot as plt

# === Load torque log ===
data = np.loadtxt("torque_log.csv", delimiter=",", skiprows=1)
time = data[:, 0]
torques = data[:, 1:]  # shape (N, 7)

# === Plot each joint in a subplot ===
num_joints = torques.shape[1]
fig, axes = plt.subplots(num_joints, 1, figsize=(10, 12), sharex=True)

for i in range(num_joints):
    axes[i].plot(time, torques[:, i], label=f'Joint {i+1}', color='b')
    axes[i].axhline(0, color='k', linestyle='--', linewidth=0.8)
    axes[i].set_ylabel(f'τ{ i+1 } [Nm]')
    axes[i].grid(True)
    axes[i].legend(loc='upper right')

axes[-1].set_xlabel('Time [s]')
fig.suptitle('Joint Torques over Time', fontsize=14)
fig.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()
