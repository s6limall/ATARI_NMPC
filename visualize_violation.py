"""
Visualise a safety-violation snapshot saved by safety.py.

Usage:
    python3 visualize_violation.py                        # latest snapshot
    python3 visualize_violation.py violation_snapshots/violation_143021.npz
"""
import sys
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── locate file ──────────────────────────────────────────────────────────────
SNAP_DIR = os.path.join(os.path.dirname(
    os.path.abspath(__file__)), "violation_snapshots")

if len(sys.argv) > 1:
    npz_path = sys.argv[1]
else:
    files = sorted(glob.glob(os.path.join(SNAP_DIR, "violation_*.npz")))
    if not files:
        print(f"No snapshots found in {SNAP_DIR}")
        sys.exit(1)
    npz_path = files[-1]

print(f"Loading: {npz_path}")
d = dict(np.load(npz_path, allow_pickle=True))

# ── unpack ───────────────────────────────────────────────────────────────────
names = list(d["joint_names"])
vals = d["joint_values"]
lim_min = d["joint_lim_min"]
lim_max = d["joint_lim_max"]
violated = d["joint_violated"].astype(bool)
torques = d.get("torques", np.array([]))
full_q = d.get("full_q", np.array([]))
q_plan = d.get("q_plan_joints", None)
v_plan = d.get("v_plan_joints", None)
meas_q = d.get("measured_q", full_q)
meas_v = d.get("measured_v", None)
tau_ff = d.get("torques_ff", None)
sim_time = float(d["sim_time"]) if "sim_time" in d else None
plan_step = int(d["plan_step"]) if "plan_step" in d else None

bq = d.get("base_quaternion", np.zeros(4))
w, x, y, z = bq
roll = np.degrees(np.arctan2(2*(y*z + x), 1 - 2*(x**2 + y**2)))
pitch = np.degrees(np.arcsin(np.clip(2*(x*z - y), -1, 1)))

has_plan = q_plan is not None

title = f"Safety Violation Snapshot — {os.path.basename(npz_path)}"
if sim_time is not None:
    title += f"  |  sim_t={sim_time:.3f}s  step={plan_step}"
title += f"\nBase: roll={roll:.1f}°  pitch={pitch:.1f}°  quat={np.round(bq,3).tolist()}"

n = len(names)
x = np.arange(n)


def annotate_pts(ax, xs, ys, color, offset_y, fontsize=7):
    """Place numeric labels next to each point."""
    for xi, yi in zip(xs, ys):
        ax.annotate(f"{yi:.3f}", xy=(xi, yi), xytext=(0, offset_y),
                    textcoords="offset points", ha="center",
                    va="bottom" if offset_y > 0 else "top",
                    fontsize=fontsize, color=color)


n_rows = 3 if (has_plan and meas_v is not None and v_plan is not None) else 2
fig, axes = plt.subplots(n_rows, 1, figsize=(15, 4.5 * n_rows))
fig.suptitle(title, fontsize=11, fontweight="bold")

# ── subplot 0: joint positions — measured, planned, limits ───────────────────
ax = axes[0]
ax.plot(x, vals, "o-", color="steelblue",
        label="measured", markersize=6, zorder=5)
annotate_pts(ax, x, vals, "steelblue", 6)
if has_plan:
    ax.plot(x, q_plan[:n], "s--", color="mediumseagreen",
            label="planned (MPC)", markersize=5, zorder=4)
    annotate_pts(ax, x, q_plan[:n], "green", -10)
ax.step(np.append(x - 0.5, x[-1] + 0.5), np.append(lim_max, lim_max[-1]),
        where="post", color="darkred", linestyle="--", lw=1.2, label="limit max")
ax.step(np.append(x - 0.5, x[-1] + 0.5), np.append(lim_min, lim_min[-1]),
        where="post", color="navy", linestyle="--", lw=1.2, label="limit min")
for i in range(n):
    if violated[i]:
        ax.axvspan(i - 0.4, i + 0.4, color="red", alpha=0.12, zorder=1)
ax.set_xticks(x)
ax.set_xticklabels(names, rotation=40, ha="right", fontsize=8.5)
ax.set_ylabel("position (rad)")
ax.set_title("Joint positions: measured vs planned vs limits", fontsize=10)
ax.legend(fontsize=8, loc="upper right")
ax.grid(axis="y", alpha=0.4)

# ── subplot 1: torques — ff vs cmd ──────────────────────────────────────────
ax = axes[1]
if tau_ff is not None and len(tau_ff) >= n:
    ax.plot(x, tau_ff, "o-", color="darkorange",
            label="τ_ff (MPC)", markersize=6, zorder=5)
    annotate_pts(ax, x, tau_ff, "darkorange", 6)
    t_cmd = torques[:n] if len(torques) >= n else np.zeros(n)
    ax.plot(x, t_cmd, "s--", color="gray",
            label="τ_cmd (clipped)", markersize=5, zorder=4)
    annotate_pts(ax, x, t_cmd, "gray", -10)
    ax.legend(fontsize=8)
elif len(torques) > 0:
    ax.plot(x[:len(torques)], torques[:n], "o-",
            color="darkorange", label="torque", markersize=6)
    annotate_pts(ax, x[:len(torques)], torques[:n], "darkorange", 6)
ax.axhline(0, color="black", lw=0.8)
ax.set_xticks(x)
ax.set_xticklabels(names, rotation=40, ha="right", fontsize=8.5)
ax.set_ylabel("torque (Nm)")
ax.set_title("Torques: feedforward vs commanded", fontsize=10)
ax.grid(axis="y", alpha=0.4)

# ── subplot 2: velocities — measured vs planned ─────────────────────────────
if n_rows == 3:
    ax = axes[2]
    n_v = min(n, len(meas_v), len(v_plan))
    ax.plot(x[:n_v], meas_v[:n_v], "o-", color="steelblue",
            label="measured dq", markersize=6, zorder=5)
    annotate_pts(ax, x[:n_v], meas_v[:n_v], "steelblue", 6)
    ax.plot(x[:n_v], v_plan[:n_v], "s--", color="mediumseagreen",
            label="planned dq", markersize=5, zorder=4)
    annotate_pts(ax, x[:n_v], v_plan[:n_v], "green", -10)
    ax.set_xticks(x[:n_v])
    ax.set_xticklabels(names[:n_v], rotation=40, ha="right", fontsize=8.5)
    ax.set_ylabel("velocity (rad/s)")
    ax.set_title("Joint velocities: measured vs planned", fontsize=10)
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(axis="y", alpha=0.4)

# ── Console print ────────────────────────────────────────────────────────────
print(f"\n  {'Joint':>20}  {'Meas q':>10}  {'Plan q':>10}  {'Δ q':>10}  {'Min':>10}  {'Max':>10}", end="")
if tau_ff is not None and len(tau_ff) >= n:
    print(f"  {'τ_ff':>8}  {'τ_cmd':>8}  {'Δ τ':>8}", end="")
print(f"  {'Viol':>6}")
print("  " + "-" * 120)
for i in range(n):
    pq = q_plan[i] if has_plan and i < len(q_plan) else float('nan')
    delta_q = vals[i] - pq if has_plan and i < len(q_plan) else float('nan')
    line = f"  {names[i]:>20}  {vals[i]:>10.4f}  {pq:>10.4f}  {delta_q:>+10.4f}  {lim_min[i]:>10.4f}  {lim_max[i]:>10.4f}"
    if tau_ff is not None and len(tau_ff) >= n:
        t_ff = tau_ff[i]
        t_cmd = torques[i] if i < len(torques) else 0.0
        line += f"  {t_ff:>8.3f}  {t_cmd:>8.3f}  {t_ff - t_cmd:>+8.3f}"
    line += f"  {'YES' if violated[i] else '':>6}"
    print(line)

fig.tight_layout(rect=[0, 0, 1, 0.95])
out_path = npz_path.replace(".npz", "_vis.png")
fig.savefig(out_path, dpi=130, bbox_inches="tight")
print(f"\nVisualization saved → {out_path}")
plt.show()
