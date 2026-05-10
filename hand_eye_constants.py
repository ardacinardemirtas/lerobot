"""
Ground-truth hand-eye calibration result.
Independently verified by two methods (PARK/HORAUD/DANIILIDIS solver + external solver).
Agreement to 6 decimal places.

T_EE_CAM : 4x4 rigid transform — camera pose expressed in end-effector frame.
           Apply as:  p_ee = T_EE_CAM @ p_cam
"""

import numpy as np

# fmt: off
T_EE_CAM = np.array([
    [-0.998972,  0.003967,  0.045150, -0.006058],
    [-0.022214, -0.911169, -0.411433,  0.068949],
    [ 0.039507, -0.412013,  0.910321, -0.057824],
    [ 0.000000,  0.000000,  0.000000,  1.000000],
], dtype=np.float64)
# fmt: on

R_EE_CAM = T_EE_CAM[:3, :3]   # 3x3 rotation
t_EE_CAM = T_EE_CAM[:3,  3]   # 3-vector, metres

# Inverse: EE pose in camera frame  (p_cam = T_CAM_EE @ p_ee)
T_CAM_EE = np.linalg.inv(T_EE_CAM)
R_CAM_EE = T_CAM_EE[:3, :3]
t_CAM_EE = T_CAM_EE[:3,  3]
