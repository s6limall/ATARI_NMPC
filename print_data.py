import numpy as np
import os

file_path = "lowstate_data/lowstate_20260127_1512.npy"

# Load the data
data = np.load(file_path)

# Build column names
columns = ["timestamp"]
columns += [f"foot{i}" for i in range(4)]
for arr_name in ["q", "dq", "ddq", "tau_est"]:
    columns += [f"{arr_name}{i}" for i in range(12)]
columns += [f"imu_quat{i}" for i in range(4)]
columns += [f"imu_gyro{i}" for i in range(3)]
columns += [f"imu_acc{i}" for i in range(3)]
columns += [f"imu_rpy{i}" for i in range(3)]

# Print summary
print(f"Data shape: {data.shape}")
print(f"Number of frames: {data.shape[0]}")
print(f"Row length: {data.shape[1]}")
print(f"Columns: {columns}")

# Print first 5 rows
print("First 5 rows:")
for i, row in enumerate(data[:5]):
    print(f"Frame {i}: ")
    print({col: val for col, val in zip(columns, row)})
