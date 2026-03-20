"""
Merge lowstate and highstate .npy recordings by absolute timestamp.

Usage:
    python merge_data.py <lowstate_dir_or_prefix> <highstate_dir_or_prefix> [output.npz]

Examples:
    # Merge all parts from a specific session:
    python merge_data.py /data/lowstate_data/lowstate_20260213_1430 /data/highstate_data/highstate_20260213_1430

    # Or point at the folders (it will find the latest session in each):
    python merge_data.py /data/lowstate_data /data/highstate_data

    # Specify output path:
    python merge_data.py /data/lowstate_data /data/highstate_data /data/merged_20260213.npz

The script:
 1. Loads all _partN.npy chunks + the final .npy for each source, concatenates them.
 2. Uses nearest-neighbor interpolation on the LOWER-frequency stream (highstate)
    to align it with the HIGHER-frequency stream (lowstate), matching by absolute timestamp.
 3. Saves a single .npz with keys:
      - timestamps     : (N,)   absolute time.time() values
      - lowstate       : (N, 65) foot_force(4) + motors(48) + imu(13)
      - highstate      : (N, 13) position(3) + velocity(3) + quaternion(4) + gyroscope(3)
      - lowstate_cols  : column names for lowstate
      - highstate_cols : column names for highstate
"""

import sys
import os
import glob
import numpy as np


# ─── Column definitions ───────────────────────────────────────────────
LOWSTATE_COLS = (
    ["foot_force_" + s for s in ["FR", "FL", "RR", "RL"]]
    + [f"motor_{i}_{x}" for x in ["q", "dq", "ddq", "tau_est"] for i in range(12)]
    + [f"quat_{i}" for i in range(4)]
    + [f"gyro_{i}" for i in range(3)]
    + [f"accel_{i}" for i in range(3)]
    + [f"rpy_{i}" for i in range(3)]
)

HIGHSTATE_COLS = (
    [f"pos_{i}" for i in range(3)]
    + [f"vel_{i}" for i in range(3)]
    + [f"quat_{i}" for i in range(4)]
    + [f"gyro_{i}" for i in range(3)]
)


def load_session(path_or_prefix: str, kind: str) -> np.ndarray:
    """
    Load and concatenate all .npy parts for a session.

    path_or_prefix can be:
      - A directory  → finds the latest session prefix automatically
      - A file prefix like /data/lowstate_data/lowstate_20260213_1430
        (with or without .npy extension)
    """
    if os.path.isdir(path_or_prefix):
        # Find the latest base file (non-part) in the directory
        base_files = sorted(glob.glob(os.path.join(path_or_prefix, f"{kind}_*.npy")))
        # Filter out part files to find base names
        base_files = [f for f in base_files if "_part" not in f]
        if not base_files:
            # No final file yet, find from parts
            part_files = sorted(glob.glob(os.path.join(path_or_prefix, f"{kind}_*_part*.npy")))
            if not part_files:
                raise FileNotFoundError(f"No {kind} .npy files found in {path_or_prefix}")
            # Extract prefix from first part file
            prefix = part_files[-1].rsplit("_part", 1)[0]
        else:
            prefix = base_files[-1].replace(".npy", "")
    else:
        prefix = path_or_prefix.replace(".npy", "")

    print(f"Loading {kind} from prefix: {prefix}")

    # Gather all part files in order
    part_files = sorted(
        glob.glob(prefix + "_part*.npy"),
        key=lambda f: int(f.rsplit("_part", 1)[1].replace(".npy", ""))
    )

    # Also check for the final (non-part) file
    final_file = prefix + ".npy"
    all_files = part_files + ([final_file] if os.path.exists(final_file) else [])

    if not all_files:
        raise FileNotFoundError(f"No .npy files found for prefix {prefix}")

    arrays = []
    for f in all_files:
        arr = np.load(f)
        arrays.append(arr)
        print(f"  Loaded {f}: {arr.shape}")

    merged = np.concatenate(arrays, axis=0)
    print(f"  Total {kind}: {merged.shape}")
    return merged


def merge(low_data: np.ndarray, high_data: np.ndarray):
    """
    Align highstate to lowstate timestamps using nearest-neighbor matching.

    Both arrays have absolute time.time() in column 0.
    Returns (timestamps, lowstate_features, highstate_aligned).
    """
    low_t = low_data[:, 0]
    high_t = high_data[:, 0]

    # Sort both by timestamp (should already be sorted, but just in case)
    low_data = low_data[np.argsort(low_t)]
    high_data = high_data[np.argsort(high_t)]
    low_t = low_data[:, 0]
    high_t = high_data[:, 0]

    # Determine overlap region
    t_start = max(low_t[0], high_t[0])
    t_end = min(low_t[-1], high_t[-1])
    print(f"\nOverlap window: {t_end - t_start:.2f} seconds")
    print(f"  Low  range: [{low_t[0]:.3f}, {low_t[-1]:.3f}]")
    print(f"  High range: [{high_t[0]:.3f}, {high_t[-1]:.3f}]")

    if t_start >= t_end:
        raise ValueError("No temporal overlap between lowstate and highstate data!")

    # Trim lowstate to overlap region
    mask = (low_t >= t_start) & (low_t <= t_end)
    low_data = low_data[mask]
    low_t = low_data[:, 0]

    # For each lowstate timestamp, find the nearest highstate sample
    indices = np.searchsorted(high_t, low_t, side="left")
    # Clamp to valid range
    indices = np.clip(indices, 1, len(high_t) - 1)
    # Pick the closer of the two neighbors
    left = np.abs(low_t - high_t[indices - 1])
    right = np.abs(low_t - high_t[indices])
    indices = np.where(left <= right, indices - 1, indices)

    high_aligned = high_data[indices]

    # Report alignment quality
    dt = np.abs(low_t - high_aligned[:, 0])
    print(f"\nAlignment quality (nearest-neighbor):")
    print(f"  Median dt: {np.median(dt)*1000:.2f} ms")
    print(f"  Max    dt: {np.max(dt)*1000:.2f} ms")
    print(f"  95th   dt: {np.percentile(dt, 95)*1000:.2f} ms")

    timestamps = low_t
    low_features = low_data[:, 1:]      # drop timestamp column
    high_features = high_aligned[:, 1:]  # drop timestamp column

    return timestamps, low_features, high_features


def main():
    if len(sys.argv) >= 3:
        low_path = sys.argv[1]
        high_path = sys.argv[2]
    else:
        low_path = "/data/lowstate_20260213_1434"
        high_path = "/data/highstate_20260213_1434"

    low_data = load_session(low_path, "lowstate")
    high_data = load_session(high_path, "highstate")

    timestamps, low_features, high_features = merge(low_data, high_data)

    # Output path
    if len(sys.argv) >= 4:
        out_path = sys.argv[3]
    else:
        prefix = low_path.split("_")[-2:]
        prefix = "_".join(prefix)
        out_path = os.path.join("/data/merged", f"merged_{prefix}.npz")

    np.savez_compressed(
        out_path,
        timestamps=timestamps,
        lowstate=low_features,
        highstate=high_features,
        lowstate_cols=LOWSTATE_COLS,
        highstate_cols=HIGHSTATE_COLS,
    )

    print(f"\nSaved merged data to {out_path}")
    print(f"  timestamps : {timestamps.shape}")
    print(f"  lowstate   : {low_features.shape}  cols: {len(LOWSTATE_COLS)}")
    print(f"  highstate  : {high_features.shape}  cols: {len(HIGHSTATE_COLS)}")
    print(f"\nLoad with:")
    print(f"  data = np.load('{out_path}')")
    print(f"  data['timestamps'], data['lowstate'], data['highstate']")
    print(f"  data['lowstate_cols'], data['highstate_cols']")


if __name__ == "__main__":
    main()
