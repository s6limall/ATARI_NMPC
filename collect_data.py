import sys
import time
import os
import numpy as np
from datetime import datetime
from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowState_

start_time = time.time()

folder_name = "/data/lowstate_data"
os.makedirs(folder_name, exist_ok=True)
start_time_str = datetime.fromtimestamp(start_time).strftime("%Y%m%d_%H%M")
npy_path = os.path.join(folder_name, f"lowstate_{start_time_str}.npy")

ROW_LENGTH = 1 + 4 + 12*4 + 4 + 3 + 3 + 3  # timestamp + foot + motors + imu
data_buffer = []
backup_interval = 2000
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

def handle_lowstate(msg: LowState_):
    row = [time.time()]

    # Foot force
    row.extend(msg.foot_force[i] for i in range(4))

    # Motors: q, dq, ddq, tau_est
    for arr_name in ["q", "dq", "ddq", "tau_est"]:
        row.extend(getattr(msg.motor_state[i], arr_name) for i in range(12))

    # IMU
    row.extend(msg.imu_state.quaternion[i] for i in range(4))
    row.extend(msg.imu_state.gyroscope[i] for i in range(3))
    row.extend(msg.imu_state.accelerometer[i] for i in range(3))
    row.extend(msg.imu_state.rpy[i] for i in range(3))

    data_buffer.append(row)

    # Periodic backup
    if len(data_buffer) % backup_interval == 0:
        save_npy(partial=True)
        data_buffer.clear()

import atexit
atexit.register(save_npy)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        ChannelFactoryInitialize(0, sys.argv[1])
    else:
        ChannelFactoryInitialize(0, "enx503eaadfddca")

    subscriber = ChannelSubscriber("rt/lowstate", LowState_)
    subscriber.Init(handle_lowstate, 10)

    while True:
        time.sleep(1)
