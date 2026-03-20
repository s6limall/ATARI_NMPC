import sys
import os
import numpy as np
import matplotlib.pyplot as plt

os.makedirs("/data/figures", exist_ok=True)


def load_merged(path: str):
    """Load a merged .npz file produced by merge_data.py."""
    data = np.load(path, allow_pickle=True)
    timestamps = data["timestamps"]
    lowstate = data["lowstate"]
    highstate = data["highstate"]
    lowstate_cols = list(data["lowstate_cols"])
    print(f"lowstate_cols: {lowstate_cols}")
    highstate_cols = list(data["highstate_cols"])

    # Make time relative (start at 0) for readable plots
    time = timestamps - timestamps[0]

    print(f"Loaded {path}")
    print(f"  Duration : {time[-1]:.2f} s  |  Frames: {len(time)}")
    print(f"  Lowstate : {lowstate.shape}  cols: {len(lowstate_cols)}")
    print(f"  Highstate: {highstate.shape}  cols: {len(highstate_cols)}")

    return time, lowstate, highstate, lowstate_cols, highstate_cols


def visualize_lowstate(time: np.ndarray, data: np.ndarray, cols: list, prefix: str):
    """Visualize lowstate data: joint positions, velocities, accelerations, foot forces."""
    if data.size == 0:
        return

    # Build column index lookup
    col_idx = {name: i for i, name in enumerate(cols)}

    plt.figure(figsize=(18, 16))

    # Joint positions (q)
    plt.subplot(4, 1, 1)
    for i in range(12):
        plt.plot(time, data[:, col_idx[f"motor_{i}_q"]], label=f"Motor {i}")
    plt.title("Joint Positions (q)")
    plt.ylabel("q [rad]")
    plt.legend(ncol=4, fontsize=8)

    # Joint velocities (dq)
    plt.subplot(4, 1, 2)
    for i in range(12):
        plt.plot(time, data[:, col_idx[f"motor_{i}_dq"]], label=f"Motor {i}")
    plt.title("Joint Velocities (dq)")
    plt.ylabel("dq [rad/s]")
    plt.legend(ncol=4, fontsize=8)

    # # Joint accelerations (ddq)
    # plt.subplot(4, 1, 3)
    # for i in range(12):
    #     plt.plot(time, data[:, col_idx[f"motor_{i}_ddq"]], label=f"Motor {i}")
    # plt.title("Joint Accelerations (ddq)")
    # plt.ylabel("ddq [rad/s²]")
    # plt.legend(ncol=4, fontsize=8)

    # Joint accelerations (ddq)
    plt.subplot(4, 1, 3)
    for i in range(12):
        plt.plot(time, data[:, col_idx[f"motor_{i}_tau_est"]], label=f"Motor {i}")
    plt.title("Joint Torques (tau_est)")
    plt.ylabel("Torque [Nm]")
    plt.legend(ncol=4, fontsize=8)

    # Foot forces
    plt.subplot(4, 1, 4)
    for name in ["foot_force_FR", "foot_force_FL", "foot_force_RR", "foot_force_RL"]:
        plt.plot(time, data[:, col_idx[name]], label=name)
    plt.title("Foot Force Sensors")
    plt.xlabel("Time [s]")
    plt.ylabel("Force [N]")
    plt.legend(ncol=4, fontsize=8)

    plt.tight_layout()
    plt.savefig(f"/data/figures/{prefix}_lowstate.png")
    plt.show()


def visualize_highstate(time: np.ndarray, data: np.ndarray, cols: list, prefix: str):
    """Visualize highstate data: positions, velocities, quaternion, gyroscope."""
    if data.size == 0:
        return

    col_idx = {name: i for i, name in enumerate(cols)}

    plt.figure(figsize=(18, 12))

    # Positions
    plt.subplot(3, 1, 1)
    for i in range(3):
        plt.plot(time, data[:, col_idx[f"pos_{i}"]], label=f"pos_{i}")
    plt.title("HighState Positions")
    plt.ylabel("Position [m]")
    plt.legend(ncol=3, fontsize=8)

    # Velocities
    plt.subplot(3, 1, 2)
    for i in range(3):
        plt.plot(time, data[:, col_idx[f"vel_{i}"]], label=f"vel_{i}")
    plt.title("HighState Velocities")
    plt.ylabel("Velocity [m/s]")
    plt.legend(ncol=3, fontsize=8)

    # IMU: quaternion + gyroscope
    plt.subplot(3, 1, 3)
    for i in range(4):
        plt.plot(time, data[:, col_idx[f"quat_{i}"]], label=f"quat_{i}")
    for i in range(3):
        plt.plot(time, data[:, col_idx[f"gyro_{i}"]], label=f"gyro_{i}", linestyle="--")
    plt.title("HighState IMU (Quaternion + Gyroscope)")
    plt.xlabel("Time [s]")
    plt.ylabel("Value")
    plt.legend(ncol=4, fontsize=8)

    plt.tight_layout()
    plt.savefig(f"/data/figures/{prefix}_highstate.png")
    plt.show()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python vis_data.py <merged_file.npz>")
        sys.exit(1)

    prefix = sys.argv[1].split(".")[0]
    prefix = prefix.split("_")[-2:]
    prefix = "_".join(prefix)
    print(f"Visualizing merged data from prefix: {prefix}")

    merged_path = sys.argv[1]
    time, lowstate, highstate, low_cols, high_cols = load_merged(merged_path)

    visualize_lowstate(time, lowstate, low_cols, prefix)
    visualize_highstate(time, highstate, high_cols, prefix)
