import time
import os
import numpy as np
from datetime import datetime
from unitree_sdk2py.core.channel import ChannelSubscriber, ChannelFactoryInitialize
from sdk_controller.topics import TOPIC_HIGHSTATE
from unitree_sdk2py.idl.unitree_go.msg.dds_ import SportModeState_
from sdk_controller.vicon_publisher import ViconHighStatePublisher

VICON_IP = "131.220.7.195:801"
OBJECT_NAME = "Go2"
NETWORK_IFACE = "enp5s0"

start_time = time.time()

folder_name = "/data/highstate_data"
os.makedirs(folder_name, exist_ok=True)
start_time_str = datetime.fromtimestamp(start_time).strftime("%Y%m%d_%H%M")
npy_path = os.path.join(folder_name, f"highstate_{start_time_str}.npy")

ROW_LENGTH = 1 + 3 + 3 + 4 + 3  # timestamp + position + velocity + quaternion + gyroscope
data_buffer = []
backup_interval = 500
backup_count = 0

def save_npy(partial=False):
    global backup_count
    if data_buffer:
        filename = npy_path
        if partial:
            backup_count += 1
            filename = npy_path.replace(".npy", f"_part{backup_count}.npy")
        np.save(filename, np.array(data_buffer, dtype=np.float64))
        if partial:
            print(f"Backed up {len(data_buffer)} frames to {filename}")
        else:
            print(f"Saved {len(data_buffer)} frames to {filename}")

# Callback to handle received highstate messages
def highstate_callback(msg: SportModeState_):
    row = [time.time()]

    # Position, velocity
    row.extend(msg.position)
    row.extend(msg.velocity)

    # IMU
    row.extend(msg.imu_state.quaternion)
    row.extend(msg.imu_state.gyroscope)

    data_buffer.append(row)

    # Periodic backup
    if len(data_buffer) % backup_interval == 0:
        save_npy(partial=True)
        data_buffer.clear()

import atexit
atexit.register(save_npy)

# Initialize network channel
ChannelFactoryInitialize(0, NETWORK_IFACE)

# Start the Vicon publisher
vicon_publisher = ViconHighStatePublisher(VICON_IP, OBJECT_NAME)

# Set up the subscriber
subscriber = ChannelSubscriber(TOPIC_HIGHSTATE, SportModeState_)
subscriber.Init(highstate_callback, 10)

# Keep script running
try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    print("Stopping...")
