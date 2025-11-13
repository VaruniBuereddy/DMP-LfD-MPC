import numpy as np
from UMPC import FR3Model, FR3Params

params = FR3Params(
    urdf_path="/home/asus/ros2_ws/src/franka_description/robots/fr3/fr3.urdf",
    ee_frame_name="fr3_hand_tcp"
)
model = FR3Model(params)

# for i, body in enumerate(model.inertias):
#     print(f"Link {i}: mass = {body.mass}")

q = np.zeros(7)

dq = np.zeros(7)

print("FK Position:", model.fk_position(q))
print("Jacobian shape:", model.jacobian_linear(q).shape)
print("Mass matrix shape:", model.M(q).shape)
print("Coriolis matrix shape:", model.C(q,dq).shape)
print("Gravity torque:", model.g(q))

q_test = np.array([0.5, -0.3, 0.2, -0.5, 0.1, 0.3, -0.2])
print("FK Position:", model.fk_position(q_test))
print("Gravity torque:", model.g(q_test))
