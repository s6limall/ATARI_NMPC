# Foot sensors
int16[4] foot_force

# Motors
float32[12] motor_q
float32[12] motor_dq
float32[12] motor_ddq
float32[12] motor_tau_est

# IMU
float32[4] imu_quaternion
float32[3] imu_gyroscope
float32[3] imu_accelerometer
float32[3] imu_rpy
