"""
Analyse Phase-1 diagnostic data saved in trajectory NPZ files.

Tests 6 hypotheses:
  H1: runing_time drifts from mj_data.time
  H2: HighState / LowState callback staleness
  H3: Reconstructed base pos/vel error vs ground truth
  H4: plan_step overflow (>= n_interp_plan)
  H5: PD gain mismatch (informational — no data needed here)
  H6: Replan cadence / solver timing

Usage:
    python3 analyze_diagnostics.py                              # latest log
    python3 analyze_diagnostics.py trajectory_logs/traj_XXXXXX.npz
"""
import sys
import os
import glob
import numpy as np

# ── locate file ─────────────────────────────────────────────────────────────
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

# ── Helper ──────────────────────────────────────────────────────────────────


def safe_load(key, default=None):
    if key in d:
        arr = d[key]
        if arr.size > 0:
            return arr
    return default


SEP = "=" * 70

# ── H1: Timing drift ───────────────────────────────────────────────────────
print(f"\n{SEP}")
print("H1: runing_time vs mj_data.time drift")
print(SEP)
mj_time = safe_load("diag_mj_time")
rt = safe_load("diag_runing_time")
if mj_time is not None and rt is not None:
    n = min(len(mj_time), len(rt))
    mj_time, rt = mj_time[:n], rt[:n]
    # Mask out zeros (before smuggling is active)
    valid = mj_time > 0
    if valid.sum() > 2:
        drift = rt[valid] - mj_time[valid]
        drift -= drift[0]  # relative to first valid sample
        print(f"  Samples with valid mj_time: {valid.sum()}/{n}")
        print(f"  Drift (runing_time - mj_time), relative to first sample:")
        print(f"    mean = {drift.mean()*1000:.2f} ms")
        print(f"    max  = {drift.max()*1000:.2f} ms")
        print(f"    min  = {drift.min()*1000:.2f} ms")
        print(f"    std  = {drift.std()*1000:.2f} ms")
        print(f"    final = {drift[-1]*1000:.2f} ms")
        if np.abs(drift).max() > 0.01:
            print("  *** SIGNIFICANT DRIFT DETECTED (>10ms) ***")
        else:
            print("  Drift within 10ms — unlikely to be the cause.")
    else:
        print("  Not enough valid mj_time samples (smuggling may not be active).")
else:
    print("  Missing diag_mj_time or diag_runing_time in NPZ.")

# ── H2: Callback staleness ─────────────────────────────────────────────────
print(f"\n{SEP}")
print("H2: HighState / LowState callback age (staleness)")
print(SEP)
hs_age = safe_load("diag_hs_age")
ls_age = safe_load("diag_ls_age")
if hs_age is not None:
    print(f"  HighState callback age (seconds since last callback):")
    print(f"    mean = {hs_age.mean()*1000:.2f} ms")
    print(f"    max  = {hs_age.max()*1000:.2f} ms")
    print(f"    p95  = {np.percentile(hs_age, 95)*1000:.2f} ms")
    print(f"    p99  = {np.percentile(hs_age, 99)*1000:.2f} ms")
    if hs_age.max() > 0.05:
        print("  *** HighState >50ms stale at some point ***")
else:
    print("  Missing diag_hs_age.")
if ls_age is not None:
    print(f"  LowState callback age (seconds since last callback):")
    print(f"    mean = {ls_age.mean()*1000:.2f} ms")
    print(f"    max  = {ls_age.max()*1000:.2f} ms")
    print(f"    p95  = {np.percentile(ls_age, 95)*1000:.2f} ms")
    print(f"    p99  = {np.percentile(ls_age, 99)*1000:.2f} ms")
    if ls_age.max() > 0.05:
        print("  *** LowState >50ms stale at some point ***")
else:
    print("  Missing diag_ls_age.")

# ── H3: Base pos/vel reconstruction error ───────────────────────────────────
print(f"\n{SEP}")
print("H3: Reconstructed base pos/vel error vs ground truth")
print(SEP)
bp_err = safe_load("diag_base_pos_err")
bv_err = safe_load("diag_base_vel_err")
bp = safe_load("diag_base_pos")
bp_true = safe_load("diag_true_base_pos")
if bp_err is not None and bp_err.shape[0] > 0:
    norms = np.linalg.norm(bp_err, axis=1)
    print(f"  Base POSITION error (reconstructed - ground_truth):")
    print(
        f"    norm: mean={norms.mean()*1000:.2f} mm, max={norms.max()*1000:.2f} mm")
    print(
        f"    x:    mean={np.abs(bp_err[:,0]).mean()*1000:.2f} mm, max={np.abs(bp_err[:,0]).max()*1000:.2f} mm")
    print(
        f"    y:    mean={np.abs(bp_err[:,1]).mean()*1000:.2f} mm, max={np.abs(bp_err[:,1]).max()*1000:.2f} mm")
    print(
        f"    z:    mean={np.abs(bp_err[:,2]).mean()*1000:.2f} mm, max={np.abs(bp_err[:,2]).max()*1000:.2f} mm")
    if norms.max() > 0.01:
        print("  *** POSITION ERROR >10mm ***")
else:
    print("  Missing or empty diag_base_pos_err.")
if bv_err is not None and bv_err.shape[0] > 0:
    norms = np.linalg.norm(bv_err, axis=1)
    print(f"  Base VELOCITY error (reconstructed - ground_truth):")
    print(f"    norm: mean={norms.mean():.4f} m/s, max={norms.max():.4f} m/s")
    print(
        f"    x:    mean={np.abs(bv_err[:,0]).mean():.4f}, max={np.abs(bv_err[:,0]).max():.4f} m/s")
    print(
        f"    y:    mean={np.abs(bv_err[:,1]).mean():.4f}, max={np.abs(bv_err[:,1]).max():.4f} m/s")
    print(
        f"    z:    mean={np.abs(bv_err[:,2]).mean():.4f}, max={np.abs(bv_err[:,2]).max():.4f} m/s")
    if norms.max() > 0.1:
        print("  *** VELOCITY ERROR >0.1 m/s ***")
else:
    print("  Missing or empty diag_base_vel_err.")

# ── H4: plan_step overflow ─────────────────────────────────────────────────
print(f"\n{SEP}")
print("H4: plan_step overflow")
print(SEP)
mpc_ps = safe_load("mpc_diag_plan_step")
mpc_of = safe_load("mpc_diag_plan_overflow")
nip = safe_load("mpc_n_interp_plan")
if mpc_ps is not None:
    nip_val = int(nip[0]) if nip is not None else "?"
    print(f"  n_interp_plan = {nip_val}")
    print(
        f"  plan_step: min={mpc_ps.min()}, max={mpc_ps.max()}, mean={mpc_ps.mean():.1f}")
    if mpc_of is not None:
        n_overflow = mpc_of.sum()
        print(
            f"  Overflow events: {n_overflow}/{len(mpc_of)} ({100*n_overflow/max(len(mpc_of),1):.1f}%)")
        if n_overflow > 0:
            idx = np.where(mpc_of)[0]
            print(
                f"  First overflow at MPC tick {idx[0]}, plan_step = {mpc_ps[idx[0]]}")
            print("  *** PLAN_STEP OVERFLOW CONFIRMED ***")
    else:
        over = mpc_ps >= nip_val if isinstance(nip_val, int) else None
        if over is not None:
            print(
                f"  Manual overflow check: {over.sum()} ticks >= n_interp_plan")
else:
    print("  Missing mpc_diag_plan_step in NPZ.")

# Also check controller-level plan_step
ctrl_ps = safe_load("diag_plan_step")
if ctrl_ps is not None and ctrl_ps.shape[0] > 0 and ctrl_ps.ndim >= 2:
    ps_col = ctrl_ps[:, 0]
    nip_col = ctrl_ps[:, 1] if ctrl_ps.shape[1] > 1 else None
    delay_col = ctrl_ps[:, 2] if ctrl_ps.shape[1] > 2 else None
    print(
        f"  Controller-level plan_step: min={ps_col.min()}, max={ps_col.max()}")
    if nip_col is not None:
        print(f"  n_interp_plan from controller: {nip_col[0]}")

# ── H6: Replan cadence and solver timing ────────────────────────────────────
print(f"\n{SEP}")
print("H6: Replan cadence and solver timing")
print(SEP)
replan = safe_load("mpc_diag_replan_events")
if replan is not None and replan.shape[0] > 0:
    # columns: (sim_step, t, delay, solver_wall_s)
    print(f"  Total replan events: {len(replan)}")
    steps = replan[:, 0]
    times = replan[:, 1]
    delays = replan[:, 2]
    solver_t = replan[:, 3]

    inter_replan = np.diff(times)
    print(f"  Inter-replan interval (seconds):")
    if len(inter_replan) > 0:
        print(f"    mean = {inter_replan.mean()*1000:.1f} ms")
        print(f"    min  = {inter_replan.min()*1000:.1f} ms")
        print(f"    max  = {inter_replan.max()*1000:.1f} ms")
        print(f"    std  = {inter_replan.std()*1000:.1f} ms")
    print(f"  Solver wall time (seconds):")
    print(f"    mean = {solver_t.mean()*1000:.1f} ms")
    print(f"    max  = {solver_t.max()*1000:.1f} ms")
    print(f"    min  = {solver_t.min()*1000:.1f} ms")
    print(f"  Plan delay (steps applied to compensate solver time):")
    print(f"    mean = {delays.mean():.1f}, max = {delays.max():.0f}")
    if delays.max() > 5:
        print("  *** HIGH DELAY — solver too slow for control rate ***")
else:
    print("  Missing or empty mpc_diag_replan_events in NPZ.")

# ── MPC timing (t vs t0) ───────────────────────────────────────────────────
print(f"\n{SEP}")
print("MPC internal timing (sim_time, t0)")
print(SEP)
mpc_t = safe_load("mpc_diag_t")
mpc_t0 = safe_load("mpc_diag_t0")
mpc_opt_node = safe_load("mpc_diag_current_opt_node")
if mpc_t is not None and mpc_t0 is not None:
    t_rel = mpc_t - mpc_t0
    print(f"  sim_time: first={mpc_t[0]:.4f}, last={mpc_t[-1]:.4f}")
    print(f"  t0: first={mpc_t0[0]:.4f}, last={mpc_t0[-1]:.4f}")
    print(f"  t (sim_time - t0): first={t_rel[0]:.4f}, last={t_rel[-1]:.4f}")
    if mpc_opt_node is not None:
        print(
            f"  current_opt_node: min={mpc_opt_node.min()}, max={mpc_opt_node.max()}")

# ── Summary ─────────────────────────────────────────────────────────────────
print(f"\n{SEP}")
print("SUMMARY")
print(SEP)
print(f"  Total controller steps: {len(d['time'])}")
print(f"  Duration: {d['time'][-1] - d['time'][0]:.3f}s (runing_time)")
print()
print("Available keys in NPZ:")
for k in sorted(d.files):
    arr = d[k]
    print(f"  {k:35s}  shape={arr.shape}  dtype={arr.dtype}")
