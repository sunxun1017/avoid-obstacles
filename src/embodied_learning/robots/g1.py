"""Published Click-and-Traverse body-site order for Unitree G1.

Source: GalaxyGeneralRobotics/Click-and-Traverse, commit 866ba392,
data/assets/unitree_g1/g1_mjx_feetonly_torque.xml. Native mjlab uses
29 position-controlled joints; the source task's action layout differs.
"""

G1_BODY_SCHEMA = (
    "head",
    "imu_in_pelvis",
    "imu_in_torso",
    "left_foot",
    "right_foot",
    "left_palm",
    "right_palm",
    "left_knee",
    "right_knee",
    "left_shoulder",
    "right_shoulder",
)
# Exact source site offsets for sites absent from native mjlab G1.
G1_ADDED_SITES = {
    "head": ("torso_link", (0.0, 0.0, 0.4)),
    "left_knee": ("left_knee_link", (0.0, 0.0, 0.0)),
    "right_knee": ("right_knee_link", (0.0, 0.0, 0.0)),
    "left_shoulder": ("left_shoulder_pitch_link", (0.0, 0.0, 0.0)),
    "right_shoulder": ("right_shoulder_pitch_link", (0.0, 0.0, 0.0)),
}
