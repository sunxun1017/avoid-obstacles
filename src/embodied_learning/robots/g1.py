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
G1_ADDED_SITES = {  # 这些就是相对link的位置
    "head": ("torso_link", (0.0, 0.0, 0.4)),
    "left_knee": ("left_knee_link", (0.0, 0.0, 0.0)),
    "right_knee": ("right_knee_link", (0.0, 0.0, 0.0)),
    "left_shoulder": ("left_shoulder_pitch_link", (0.0, 0.0, 0.0)),
    "right_shoulder": ("right_shoulder_pitch_link", (0.0, 0.0, 0.0)),
}

# Source constants.py, Apache-2.0, commit 866ba392. Ordered by names, not native IDs.
CAT_JOINT_NAMES = (
    tuple(
        f"{side}_{name}_joint"
        for side in ("left", "right")
        for name in (
            "hip_pitch",
            "hip_roll",
            "hip_yaw",
            "knee",
            "ankle_pitch",
            "ankle_roll",
        )
    )
    + ("waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint")
    + tuple(
        f"{side}_{name}_joint"
        for side in ("left", "right")
        for name in (
            "shoulder_pitch",
            "shoulder_roll",
            "shoulder_yaw",
            "elbow",
            "wrist_roll",
            "wrist_pitch",
            "wrist_yaw",
        )
    )
)
CAT_ACTION_NAMES = CAT_JOINT_NAMES[:12]
CAT_OBS_NAMES = CAT_JOINT_NAMES[:19] + CAT_JOINT_NAMES[22:26]
CAT_DEFAULT_JOINTS = (
    (-0.1, 0, 0, 0.3, -0.2, 0) * 2
    + (0, 0, 0)
    + (0.2, 0.3, 0, 1.28, 0, 0, 0)
    + (0.2, -0.3, 0, 1.28, 0, 0, 0)
)
CAT_KP = (
    (100, 100, 100, 200, 80, 20) * 2
    + (300, 300, 300)
    + (90, 60, 20, 60, 20, 20, 20) * 2
)
CAT_KD = (2, 2, 2, 4, 2, 1) * 2 + (10, 10, 10) + (2, 2, 1, 1, 1, 1, 1) * 2
CAT_TORQUE_LIMIT = (
    (88, 139, 88, 139, 50, 50) * 2 + (88, 50, 50) + (25, 25, 25, 25, 25, 5, 5) * 2
)
CAT_LEG_BODIES = (
    "left_knee_link",
    "left_ankle_roll_link",
    "right_knee_link",
    "right_ankle_roll_link",
)
