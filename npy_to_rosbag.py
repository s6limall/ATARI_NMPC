import numpy as np
import sys
import rclpy
from rclpy.node import Node
from rclpy.serialization import serialize_message
from rosbag2_py import SequentialWriter, StorageOptions, ConverterOptions, TopicMetadata
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowState_, MotorState_, ImuState_

print("Starting npy_to_rosbag conversion...")

# Path to .npy file and output rosbag
if len(sys.argv) < 3:
    print("Usage: python npy_to_rosbag.py <npy_file> <rosbag_output_path>")
    sys.exit(1)

npy_file = sys.argv[1]
rosbag_path = sys.argv[2]

# Load numpy data
data = np.load(npy_file)
print(f"Loaded {data.shape[0]} frames from {npy_file}")

# Initialize ROS 2
rclpy.init()
node = Node("npy_to_rosbag")

# Setup rosbag writer
storage_options = StorageOptions(uri=rosbag_path, storage_id="sqlite3")
converter_options = ConverterOptions(
    input_serialization_format="cdr",
    output_serialization_format="cdr"
)

writer = SequentialWriter()
writer.open(storage_options, converter_options)

# Create topic
topic_name = "/rt/lowstate"
topic_metadata = TopicMetadata(
    name=topic_name,
    type="unitree_go_interfaces/msg/LowState",
    serialization_format="cdr"
)
writer.create_topic(topic_metadata)

# Helper function to convert a row to LowState message
def row_to_lowstate(row):
    msg = LowState_()
    idx = 0
    msg.timestamp = row[idx]; idx += 1

    # Foot force
    msg.foot_force = row[idx:idx+4].tolist(); idx += 4

    # Motors
    msg.motor_state = [MotorState_() for _ in range(12)]
    for i in range(12):
        msg.motor_state[i].q = row[idx]; idx += 1
    for i in range(12):
        msg.motor_state[i].dq = row[idx]; idx += 1
    for i in range(12):
        msg.motor_state[i].ddq = row[idx]; idx += 1
    for i in range(12):
        msg.motor_state[i].tau_est = row[idx]; idx += 1

    # IMU
    imu = ImuState_()
    imu.quaternion = row[idx:idx+4].tolist(); idx += 4
    imu.gyroscope = row[idx:idx+3].tolist(); idx += 3
    imu.accelerometer = row[idx:idx+3].tolist(); idx += 3
    imu.rpy = row[idx:idx+3].tolist(); idx += 3
    msg.imu_state = imu

    return msg

# Write each row to the bag
for i, row in enumerate(data):
    msg = row_to_lowstate(row)
    ts = int(row[0] * 1e9)  # timestamp in nanoseconds
    writer.write(topic_name, serialize_message(msg), ts)

writer.close()
print(f"ROS 2 bag saved at: {rosbag_path}")

# Shutdown ROS 2 cleanly
node.destroy_node()
rclpy.shutdown()
