import numpy as np


ROBOT_NAME = "go2"
OBJECT_NAME = "go2"
P_IMU_IN_VICON = np.array([ 0.0,  0.0, 0.0])
R_IMU_IN_VICON = np.array(
    [[ 1, 0, 0],
    [ 0,  1, 0],
    [ 0,  0,  1],]
    )

STAND_UP_JOINT_POS = np.array([
    0.00571868, 0.608813, -1.21763, -0.00571868, 0.608813, -1.21763,
    0.00571868, 0.608813, -1.21763, -0.00571868, 0.608813, -1.21763
], dtype=float)

STAND_DOWN_JOINT_POS = np.array([
    0.0473455, 1.22187, -2.44375, -0.0473455, 1.22187, -2.44375, 0.0473455,
    1.22187, -2.44375, -0.0473455, 1.22187, -2.44375
],dtype=float)

P_IMU_IN_BASE = np.array([0,0,0])
R_IMU_IN_BASE = np.eye(3)

CONTROL_FREQ = 80
Kp = 30
Kd = 1.8

# Multiply gain on the real robot
scale_gains = 1.5