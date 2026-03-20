import os
import time
import numpy as np
from datetime import datetime
from threading import Lock
from atexit import register

from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowState_, SportModeState_
from sdk_controller.vicon_publisher import ViconHighStatePublisher
from sdk_controller.topics import TOPIC_HIGHSTATE

# ------------------- Configuration -------------------
LOWSTATE_IFACE = "enx503eaadfddca"   # Interface for low-level robot channel
HIGHSTATE_IFACE = "enp5s0"           # Interface for high-level/Vicon channel
VICON_IP = "131.220.7.195:801"
OBJECT_NAME = "Go2"

BACKUP_INTERVAL = 3000  # frames
LOWSTATE_FOLDER = "/data/lowstate_data"
HIGHSTATE_FOLDER = "/data/highstate_data"
os.makedirs(LOWSTATE_FOLDER, exist_ok=True)
os.makedirs(HIGHSTATE_FOLDER, exist_ok=True)

start_time = time.time()
start_time_str = datetime.fromtimestamp(start_time).strftime("%Y%m%d_%H%M")

# File paths
lowstate_path = os.path.join(LOWSTATE_FOLDER, f"lowstate_{start_time_str}.npy")
highstate_path = os.path.join(HIGHSTATE_FOLDER, f"highstate_{start_time_str}.npy")

# ------------------- Buffers -------------------
lowstate_buffer = []
highstate_buffer = []
lowstate_lock = Lock()
highstate_lock = Lock()
lowstate_backup_count = 0
highstate_backup_count = 0

# ------------------- LowState Callback -------------------
def handle_lowstate(msg: LowState_):
    global lowstate_backup_count
    try:
        timestamp = time.time() - start_time
        row = [timestamp]

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

        print(f"[DEBUG] LowState trying to acquire lock at {time.time()}")
        with lowstate_lock:
            print(f"[DEBUG] LowState acquired lock at {time.time()}")
            lowstate_buffer.append(row)
            if len(lowstate_buffer) >= BACKUP_INTERVAL:
                # Copy buffer and clear it to avoid blocking callbacks
                buffer_copy = lowstate_buffer.copy()
                lowstate_buffer.clear()
                lowstate_backup_count += 1
                filename = lowstate_path.replace(".npy", f"_part{lowstate_backup_count}.npy")
                try:
                    start_save = time.time()
                    np.save(filename, np.array(buffer_copy, dtype=np.float32))
                    end_save = time.time()
                    print(f"[DEBUG] Save took {end_save - start_save:.2f} seconds")
                    print(f"LowState backup saved to {filename}")
                except Exception as e:
                    print(f"Failed to save LowState backup: {e}")
    except Exception as e:
        print(f"Error in LowState callback: {e}")

# ------------------- HighState Callback -------------------
def handle_highstate(msg: SportModeState_):
    global highstate_backup_count
    try:
        timestamp = time.time() - start_time
        row = [timestamp]

        # Position, velocity
        row.extend(msg.position)
        row.extend(msg.velocity)

        # IMU
        row.extend(msg.imu_state.quaternion)
        row.extend(msg.imu_state.gyroscope)

        print(f"[DEBUG] HighState trying to acquire lock at {time.time()}")
        with highstate_lock:
            print(f"[DEBUG] HighState acquired lock at {time.time()}")
            highstate_buffer.append(row)
            if len(highstate_buffer) >= BACKUP_INTERVAL:
                buffer_copy = highstate_buffer.copy()
                highstate_buffer.clear()
                highstate_backup_count += 1
                filename = highstate_path.replace(".npy", f"_part{highstate_backup_count}.npy")
                try:
                    start_save = time.time()
                    np.save(filename, np.array(buffer_copy, dtype=np.float32))
                    end_save = time.time()
                    print(f"[DEBUG] Save took {end_save - start_save:.2f} seconds")
                    print(f"HighState backup saved to {filename}")
                except Exception as e:
                    print(f"Failed to save HighState backup: {e}")
    except Exception as e:
        print(f"Error in HighState callback: {e}")
# ------------------- Exit Saving -------------------
def save_on_exit():
    with lowstate_lock:
        save_npy(lowstate_buffer, lowstate_path)
    with highstate_lock:
        save_npy(highstate_buffer, highstate_path)

register(save_on_exit)

# ------------------- Main -------------------
def main():
    # Initialize network channel separately for lowstate
    ChannelFactoryInitialize(0, LOWSTATE_IFACE)
    low_sub = ChannelSubscriber("rt/lowstate", LowState_)
    low_sub.Init(handle_lowstate, 200)

    # Initialize network channel separately for highstate
    ChannelFactoryInitialize(0, HIGHSTATE_IFACE)
    vicon_publisher = ViconHighStatePublisher(VICON_IP, OBJECT_NAME)
    high_sub = ChannelSubscriber(TOPIC_HIGHSTATE, SportModeState_)
    high_sub.Init(handle_highstate, 200)
    print("HighState subscriber initialized.")

    # Keep script alive
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Exiting...")

if __name__ == "__main__":
    main()
