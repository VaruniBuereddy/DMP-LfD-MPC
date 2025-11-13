import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import numpy as np

# === File paths ===
# nominal_path = 'outputs/ee_pose.csv'
dmp_path = '/home/asus/Documents/DMP_LfD/ee_tracking.csv'
kmp_path = '/home/asus/Documents/KMP-LQR/mu_star.csv'
demo_path = '/home/asus/Documents/DMP_LfD/paper_roboface_demonstrated_joint_positions(t,j1-j7,x,y,z).csv'
stt_path = '/home/asus/ros2_ws/src/franka_example_controllers/scripts/paper_roboface_fb_learned_end_effector_positions(t(ignore),x,y,z).csv'

# === Load nominal and mass trajectories ===
# df_ = pd.read_csv(demo_path)
df_dmp = pd.read_csv(dmp_path)
df_kmp = pd.read_csv(kmp_path)
# x_nom = df_nom['ee_x'].to_numpy()
# y_nom = df_nom['ee_y'].to_numpy()
# z_nom = df_nom['ee_z'].to_numpy()

x_dmp = df_dmp['x_desired'].to_numpy()
y_dmp = df_dmp['y_desired'].to_numpy()
z_dmp = df_dmp['z_desired'].to_numpy()

# print(df_kmp.columns)
x_kmp = df_kmp['s1'].to_numpy()
y_kmp = df_kmp['s2'].to_numpy()
z_kmp = df_kmp['s3'].to_numpy()

# === Load demonstration trajectory ===
df_demo = pd.read_csv(demo_path, header=None)
x_demo = df_demo.iloc[:, -3].to_numpy()
y_demo = df_demo.iloc[:, -2].to_numpy()
z_demo = df_demo.iloc[:, -1].to_numpy()

# === Load STT trajectory ===
df_stt = pd.read_csv(stt_path, header=None)
x_stt = df_stt.iloc[:, 1].to_numpy()
y_stt = df_stt.iloc[:, 2].to_numpy()
z_stt = df_stt.iloc[:, 3].to_numpy()

# === Tube around nominal trajectory ===
tube_radius = 0.008  # 5 mm tolerance
num_circle_points = 20

points = np.vstack((x_demo, y_demo, z_demo)).T
directions = np.diff(points, axis=0)
directions = np.vstack((directions, directions[-1]))  # match length
norms = np.linalg.norm(directions, axis=1, keepdims=True)
directions = directions / norms

ref = np.array([0, 0, 1])
theta = np.linspace(0, 2 * np.pi, num_circle_points)
tube_X, tube_Y, tube_Z = [], [], []

for i, center in enumerate(points):
    t_vec = directions[i]
    if np.allclose(np.cross(ref, t_vec), 0):  # handle parallel
        ref = np.array([0, 1, 0])
    n_vec = np.cross(ref, t_vec)
    n_vec /= np.linalg.norm(n_vec)
    b_vec = np.cross(t_vec, n_vec)
    circle_pts = center + tube_radius * (np.outer(np.cos(theta), n_vec) + np.outer(np.sin(theta), b_vec))
    tube_X.append(circle_pts[:, 0])
    tube_Y.append(circle_pts[:, 1])
    tube_Z.append(circle_pts[:, 2])

tube_X = np.array(tube_X)
tube_Y = np.array(tube_Y)
tube_Z = np.array(tube_Z)
# === Plot ===
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# --- Demonstration tube ---
ax.plot_surface(
    tube_X, tube_Y, tube_Z,
    color='blue', alpha=0.15, linewidth=0, shade=True, label='_nolegend_'  # translucent tube
)

# Demonstration (centerline)
ax.plot(x_demo, y_demo, z_demo, color='green', linestyle='-.', label='Demonstration', linewidth=2)

# DMP + MPC trajectory
ax.plot(x_dmp, y_dmp, z_dmp, color='orange',linestyle='-.', label='DMP+UMPC', linewidth=2)
ax.plot(x_kmp, y_kmp, z_kmp, color='purple', label='KMP+LQR', linewidth=2)

# STT (feedback learned)
ax.plot(x_stt, y_stt, z_stt, color='red', label='STT Learned', linewidth=2)

# Start and end markers
# ax.scatter(x_demo[0], y_demo[0], z_demo[0], color='darkgreen', s=40, marker='^', label='Start')
# ax.scatter(x_demo[-1], y_demo[-1], z_demo[-1], color='darkred', s=40, marker='^', label='End')
# ax.scatter(x_stt[0], y_stt[0], z_stt[0], color='maroon', s=40, marker='s', label='Start STT')
# ax.scatter(x_stt[-1], y_stt[-1], z_stt[-1], color='black', s=40, marker='s', label='End STT')

# Labels and layout
ax.set_xlabel('X [m]')
ax.set_ylabel('Y [m]')
ax.set_zlabel('Z [m]')
ax.set_title('End Effector Trajectories')
ax.legend()
ax.grid(True)
plt.tight_layout()
plt.show()
