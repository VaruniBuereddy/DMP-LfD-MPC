import pinocchio as pin
import numpy as np
from os.path import join

# Path to your URDF file
urdf_path = "/home/asus/ros2_ws/src/franka_description/robots/fr3/fr3.urdf"

# Load the model
model, collision_model, visual_model = pin.buildModelsFromUrdf(urdf_path)

# Print basic info
print(f"Robot has {model.njoints} joints (including universe).")

# List all joint names (skip index 0 because it's 'universe')
print("\n--- Joint Names ---")
for idx, joint in enumerate(model.names):
    if idx == 0:
        continue  # skip universe
    print(f"{idx}: {joint}")

# End-effector = usually last joint in chain
ee_joint_id = model.getJointId(model.names[-1])
print(f"\n--- End-effector ---")
print(f"Joint ID: {ee_joint_id}")
print(f"Joint Name: {model.names[-1]}")

# list frames (if you want end-effector frame)
print("\n--- Frames ---")
for idx, frame in enumerate(model.frames):
    print(f"{idx}: {frame.name} ({frame.type})")

# # End-effector can also be defined as last frame:
# ee_frame = model.frames[-1]
# print(f"\nEnd-effector Frame: {ee_frame.name}, ID: {ee_frame.id}")

ee_frame_id = model.getFrameId("fr3_hand_tcp")
print("End-effector frame ID:", ee_frame_id)

ee_frame = model.frames[ee_frame_id]
print("End-effector frame name:", ee_frame.name)


# from urdfpy import URDF

# robot = URDF.load(urdf_path)
# tau_limits = []
# for joint in robot.joints:
#     if joint.limit is not None:
#         tau_limits.append(joint.limit.effort)
# tau_limits = np.array(tau_limits)
# print(tau_limits)
import os
import trimesh
from urdfpy import URDF

urdf_path = "/home/asus/ros2_ws/src/franka_description/robots/fr3/fr3.urdf"

# Where your franka_description package actually lives
package_map = {
    "franka_description": "/home/asus/ros2_ws/src/franka_description"
}

def resolve_package_path(uri):
    # Remove any accidental absolute prefix before package://
    if "package://" in uri:
        uri = uri[uri.find("package://"):]  # strip preceding junk
    if uri.startswith("package://"):
        pkg_name, rel_path = uri.replace("package://", "").split("/", 1)
        if pkg_name in package_map:
            return os.path.join(package_map[pkg_name], rel_path)
    return uri

# Patch trimesh.load
_orig_load = trimesh.load
def patched_load(file_obj, *args, **kwargs):
    if isinstance(file_obj, str) and "package://" in file_obj:
        file_obj = resolve_package_path(file_obj)
    return _orig_load(file_obj, *args, **kwargs)

trimesh.load = patched_load

robot = URDF.load(urdf_path)
print("URDF loaded successfully.")

tau_limits = []
for joint in robot.joints:
    if joint.limit is not None:
        tau_limits.append(joint.limit.effort)
tau_limits = np.array(tau_limits)
print(tau_limits)