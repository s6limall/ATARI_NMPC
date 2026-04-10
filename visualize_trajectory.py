"""
Visualise recorded MPC trajectory: measured vs planned per joint over time.

Usage:
    python3 visualize_trajectory.py                              # latest log
    python3 visualize_trajectory.py trajectory_logs/traj_123456.npz
"""
import sys
import os
import glob
import numpy as np
import matplotlib.pyplot as plt

# ── locate file ──────────────────────────────────────────────────────────────
LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "trajectory_logs")

if len(sys.argv) > 1:
    npz_path = sys.argv[1]
else:
    files = sorted(glob.glob(os.path.join(LOG_DIR, "traj_*.npz")))
    if not files:
        print(f"No trajectory logs found in {LOG_DIR}")
        sys.exit(1)
    npz_path = files[-1]

print(f"Loading: {npz_path}")
d = np.load(npz_path, allow_pickle=True)

t = d["time"]           # (N,)
meas_q = d["meas_q"]    # (N, 12)
plan_q = d["plan_q"]    # (N, 12)
meas_v = d["meas_v"]    # (N, 12)
plan_v = d["plan_v"]    # (N, 12)
tau_ff = d["tau_ff"]     # (N, 12)
tau_cmd = d["tau_cmd"]   # (N, 12)
joint_names = list(d["joint_names"])

n_joints = meas_q.shape[1]
t0 = t[0]
t_rel = t - t0  # relative time from first MPC step

# ── Group joints by leg ──────────────────────────────────────────────────────
# Pinocchio DOF order: FL(0-2), FR(3-5), RL(6-8), RR(9-11)
legs = {
    "FL": slice(0, 3),
    "FR": slice(3, 6),
    "RL": slice(6, 9),
    "RR": slice(9, 12),
}
joint_types = ["hip", "thigh", "calf"]
colors_meas = ["#1f77b4", "#2ca02c", "#d62728"]   # blue, green, red
colors_plan = ["#aec7e8", "#98df8a", "#ff9896"]   # lighter versions

# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 1: Joint positions — measured vs planned (one subplot per leg)
# ═══════════════════════════════════════════════════════════════════════════════
fig1, axes1 = plt.subplots(2, 2, figsize=(16, 10), sharex=True)
fig1.suptitle(f"Joint Positions: Measured vs Planned — {os.path.basename(npz_path)}",
              fontsize=12, fontweight="bold")

for idx, (leg, sl) in enumerate(legs.items()):
    ax = axes1.flat[idx]
    for j, jt in enumerate(joint_types):
        ji = sl.start + j
        ax.plot(t_rel, meas_q[:, ji], color=colors_meas[j], lw=1.2,
                label=f"{jt} meas")
        ax.plot(t_rel, plan_q[:, ji], color=colors_plan[j], lw=1.2,
                linestyle="--", label=f"{jt} plan")
    ax.set_title(f"{leg}", fontsize=11)
    ax.set_ylabel("position (rad)")
    ax.legend(fontsize=7, ncol=2, loc="upper right")
    ax.grid(alpha=0.3)

for ax in axes1[-1]:
    ax.set_xlabel("time (s)")
fig1.tight_layout(rect=[0, 0, 1, 0.96])

# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 2: Joint velocities — measured vs planned (one subplot per leg)
# ═══════════════════════════════════════════════════════════════════════════════
fig2, axes2 = plt.subplots(2, 2, figsize=(16, 10), sharex=True)
fig2.suptitle(f"Joint Velocities: Measured vs Planned — {os.path.basename(npz_path)}",
              fontsize=12, fontweight="bold")

for idx, (leg, sl) in enumerate(legs.items()):
    ax = axes2.flat[idx]
    for j, jt in enumerate(joint_types):
        ji = sl.start + j
        ax.plot(t_rel, meas_v[:, ji], color=colors_meas[j], lw=1.2,
                label=f"{jt} meas")
        ax.plot(t_rel, plan_v[:, ji], color=colors_plan[j], lw=1.2,
                linestyle="--", label=f"{jt} plan")
    ax.set_title(f"{leg}", fontsize=11)
    ax.set_ylabel("velocity (rad/s)")
    ax.legend(fontsize=7, ncol=2, loc="upper right")
    ax.grid(alpha=0.3)

for ax in axes2[-1]:
    ax.set_xlabel("time (s)")
fig2.tight_layout(rect=[0, 0, 1, 0.96])

# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 3: Torques — ff vs commanded (one subplot per leg)
# ═══════════════════════════════════════════════════════════════════════════════
fig3, axes3 = plt.subplots(2, 2, figsize=(16, 10), sharex=True)
fig3.suptitle(f"Torques: FF vs Commanded — {os.path.basename(npz_path)}",
              fontsize=12, fontweight="bold")

for idx, (leg, sl) in enumerate(legs.items()):
    ax = axes3.flat[idx]
    for j, jt in enumerate(joint_types):
        ji = sl.start + j
        ax.plot(t_rel, tau_ff[:, ji], color=colors_meas[j], lw=1.2,
                label=f"{jt} τ_ff")
        ax.plot(t_rel, tau_cmd[:, ji], color=colors_plan[j], lw=1.2,
                linestyle="--", label=f"{jt} τ_cmd")
    ax.set_title(f"{leg}", fontsize=11)
    ax.set_ylabel("torque (Nm)")
    ax.legend(fontsize=7, ncol=2, loc="upper right")
    ax.grid(alpha=0.3)
    ax.axhline(0, color="black", lw=0.5)

for ax in axes3[-1]:
    ax.set_xlabel("time (s)")
fig3.tight_layout(rect=[0, 0, 1, 0.96])

# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 4: Tracking error per joint (measured - planned)
# ═══════════════════════════════════════════════════════════════════════════════
fig4, axes4 = plt.subplots(2, 2, figsize=(16, 10), sharex=True)
fig4.suptitle(f"Position Tracking Error (meas − plan) — {os.path.basename(npz_path)}",
              fontsize=12, fontweight="bold")

for idx, (leg, sl) in enumerate(legs.items()):
    ax = axes4.flat[idx]
    for j, jt in enumerate(joint_types):
        ji = sl.start + j
        err = meas_q[:, ji] - plan_q[:, ji]
        ax.plot(t_rel, err, color=colors_meas[j], lw=1.2, label=f"{jt}")
    ax.set_title(f"{leg}", fontsize=11)
    ax.set_ylabel("error (rad)")
    ax.legend(fontsize=7, loc="upper right")
    ax.grid(alpha=0.3)
    ax.axhline(0, color="black", lw=0.5)

for ax in axes4[-1]:
    ax.set_xlabel("time (s)")
fig4.tight_layout(rect=[0, 0, 1, 0.96])

# ── Save ─────────────────────────────────────────────────────────────────────
base = npz_path.replace(".npz", "")
for fig, suffix in [(fig1, "_pos"), (fig2, "_vel"), (fig3, "_tau"), (fig4, "_err")]:
    out = base + suffix + ".png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print(f"Saved → {out}")

# ── Console summary ──────────────────────────────────────────────────────────
print(
    f"\nDuration: {t_rel[-1]:.2f}s  |  Steps: {len(t)}  |  dt≈{np.mean(np.diff(t)):.4f}s")
print(f"\nMax |pos error| per joint:")
for i in range(n_joints):
    err = np.abs(meas_q[:, i] - plan_q[:, i])
    print(
        f"  {joint_names[i]:>20}: {np.max(err):.4f} rad  (mean {np.mean(err):.4f})")

plt.show()
