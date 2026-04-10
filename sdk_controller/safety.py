from mj_pin.utils import mj_joint_name2act_id, mj_joint_name2dof
import matplotlib.pyplot as plt
import numpy as np
import mujoco
import os
import time
import matplotlib
matplotlib.use("Agg")         # works headless / in background threads

_SNAPSHOT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                             "violation_snapshots")
os.makedirs(_SNAPSHOT_DIR, exist_ok=True)


class SafetyLayer:
    def __init__(self, mj_model):
        """
        Safety controller to enforce kinematic and torque limits.

        Args:
            mj_model : MuJoCo model with joint and ctrl limits.
        """
        joint_name2act_id = mj_joint_name2act_id(mj_model)
        joint_name2dof = mj_joint_name2dof(mj_model)
        self.joint_act_id2dof = {
            v: joint_name2dof[k] for k, v in joint_name2act_id.items()}
        self.joint_dof2act_id = {v: k for k,
                                 v in self.joint_act_id2dof.items()}
        # reverse: qpos_addr -> joint name  (for readable prints)
        self.qpos2name = {}
        for jnt_name, act_id in joint_name2act_id.items():
            dof = joint_name2dof[jnt_name]
            self.qpos2name[dof + 1] = jnt_name   # off=1 (nq-nv)
        self.joint_limits = {}
        self.torque_limits = {}
        self.base_orientation_limit = 40 * np.pi / 180.
        self.scale_joint_limit = 0.95
        self.scale_torque_limit = 0.9
        # MuJoCo soft-limit enforcement can transiently push joints slightly
        # past their defined range.  Allow this much slack (rad) beyond the
        # scaled limit before actually triggering a safety stop.
        self._joint_tol = 0.05

        for id in range(mj_model.njnt):
            dof = int(mj_model.joint(id).dofadr)
            if not dof in self.joint_dof2act_id:
                continue
            act_id = self.joint_dof2act_id[dof]
            q_dof = int(mj_model.jnt_qposadr[id])
            if mj_model.jnt_limited[id]:
                min = mj_model.jnt_range[id][0]
                max = mj_model.jnt_range[id][1]
                self.joint_limits[q_dof] = (
                    min * self.scale_joint_limit, max * self.scale_joint_limit)
            if mj_model.actuator_ctrllimited[act_id]:
                self.torque_limits[act_id] = mj_model.actuator_ctrlrange[act_id][1] * \
                    self.scale_torque_limit

        # Extra context set by the controller before each safety check
        self._debug_ctx = {}

    def set_debug_context(self, **kwargs):
        """Call from update_motor_cmd to attach extra data for violation snapshots."""
        self._debug_ctx = kwargs

    def check_joint_limits(self, joint_positions):
        """Check if joint positions exceed limits."""
        if not self.joint_limits:
            return True

        tol = self._joint_tol
        rows = []
        violation = False
        for joint_id, (q_min, q_max) in sorted(self.joint_limits.items()):
            val = joint_positions[joint_id]
            in_range = (q_min - tol) <= val <= (q_max + tol)
            if not in_range:
                violation = True
                name = self.qpos2name.get(joint_id, f"q[{joint_id}]")
                rows.append((joint_id, name, q_min, val, q_max))

        if violation:
            print("\n  [SAFETY] Joint position check (violations only):")
            print(
                f"  {'qpos_idx':>8}  {'name':>18}  {'min_lim':>10}  {'value':>10}  {'max_lim':>10}")
            for joint_id, name, q_min, val, q_max in rows:
                print(
                    f"  {joint_id:>8}  {name:>18}  {q_min:>10.4f}  {val:>10.4f}  {q_max:>10.4f}  <<< VIOLATION")
        return not violation

    def check_torque_limits(self, torques):
        """Check if torques exceed limits."""
        if not self.torque_limits:
            return True

        for act_id, max_torque in self.torque_limits.items():
            if abs(torques[act_id]) > max_torque:
                print(
                    f"Warning: Joint {act_id} torque limit exceeded! ({torques[act_id]:.3f} > {max_torque:.3f})")
                return False
        return True

    def check_base_orientation(self, base_quaternion):
        """Check if base orientation exceeds the safety threshold."""
        if self.base_orientation_limit is None:
            return True

        _, x, y, z = base_quaternion  # Assuming quaternion (w, x, y, z)
        roll = np.arctan2(2 * (y * z + x), 1 - 2 * (x ** 2 + y ** 2))
        pitch = np.arcsin(2 * (x * z - y))

        if abs(roll) > self.base_orientation_limit or abs(pitch) > self.base_orientation_limit:
            print(
                f"Warning: Base orientation limit exceeded! roll={np.degrees(roll):.1f}° pitch={np.degrees(pitch):.1f}° (limit={np.degrees(self.base_orientation_limit):.1f}°)")
            return False
        return True

    def check_safety(self, joint_positions, torques, base_quaternion) -> bool:
        """Enforce safety by zeroing torques if any limit is exceeded."""
        jnt_ok = self.check_joint_limits(joint_positions)
        tau_ok = self.check_torque_limits(torques)
        ori_ok = self.check_base_orientation(base_quaternion)
        if not (jnt_ok and tau_ok and ori_ok):
            print("Safety violation detected! Setting torques to zero.")
            self._save_violation_snapshot(
                joint_positions, torques, base_quaternion)
            return False
        return True

    # ── internal helpers ──────────────────────────────────────────────────────

    def _save_violation_snapshot(self, joint_positions, torques, base_quaternion):
        ts = time.strftime("%H%M%S")
        npz_path = os.path.join(_SNAPSHOT_DIR, f"violation_{ts}.npz")

        # build limit arrays aligned with joint_limits keys
        jnt_ids = sorted(self.joint_limits.keys())
        names = [self.qpos2name.get(i, f"q[{i}]") for i in jnt_ids]
        vals = np.array([joint_positions[i] for i in jnt_ids])
        lim_min = np.array([self.joint_limits[i][0] for i in jnt_ids])
        lim_max = np.array([self.joint_limits[i][1] for i in jnt_ids])
        violated = (vals < lim_min) | (vals > lim_max)

        torques_arr = np.array(torques)

        save_dict = dict(
            joint_names=np.array(names),
            joint_qpos_ids=np.array(jnt_ids),
            joint_values=vals,
            joint_lim_min=lim_min,
            joint_lim_max=lim_max,
            joint_violated=violated,
            base_quaternion=np.array(base_quaternion),
            torques=torques_arr,
            full_q=np.array(joint_positions),
        )
        # merge any extra context from the controller
        for k, v in self._debug_ctx.items():
            save_dict[k] = np.array(v)

        np.savez(npz_path, **save_dict)
        print(f"  [SAFETY] Snapshot saved → {npz_path}")

        # auto-generate plot
        plot_path = npz_path.replace(".npz", ".png")
        self._plot_violation(save_dict, plot_path)
        print(f"  [SAFETY] Plot saved    → {plot_path}")

    @staticmethod
    def _annotate_pts(ax, xs, ys, color, offset_y, fontsize=7):
        for xi, yi in zip(xs, ys):
            ax.annotate(f"{yi:.3f}", xy=(xi, yi), xytext=(0, offset_y),
                        textcoords="offset points", ha="center",
                        va="bottom" if offset_y > 0 else "top",
                        fontsize=fontsize, color=color)

    def _plot_violation(self, d, path):
        names = list(d["joint_names"])
        vals = d["joint_values"]
        lim_min = d["joint_lim_min"]
        lim_max = d["joint_lim_max"]
        violated = d["joint_violated"]
        torques = d["torques"]
        has_plan = "q_plan_joints" in d
        tau_ff = d.get("torques_ff", None)
        v_plan = d.get("v_plan_joints", None)
        meas_v = d.get("measured_v", None)

        n = len(names)
        x = np.arange(n)

        n_rows = 3 if (
            has_plan and meas_v is not None and v_plan is not None) else 2
        fig, axes = plt.subplots(n_rows, 1, figsize=(14, 4 * n_rows))
        fig.suptitle("Safety Violation Snapshot",
                     fontsize=13, fontweight="bold")

        # ── row 0: joint positions — measured, planned, limits ────────────────
        ax = axes[0]
        ax.plot(x, vals, "o-", color="steelblue",
                label="measured", markersize=6, zorder=5)
        self._annotate_pts(ax, x, vals, "steelblue", 6)
        if has_plan:
            pq = d["q_plan_joints"][:n]
            ax.plot(x, pq, "s--", color="mediumseagreen",
                    label="planned (MPC)", markersize=5, zorder=4)
            self._annotate_pts(ax, x, pq, "green", -10)
        ax.step(np.append(x - 0.5, x[-1] + 0.5), np.append(lim_max, lim_max[-1]),
                where="post", color="darkred", linestyle="--", lw=1.2, label="limit max")
        ax.step(np.append(x - 0.5, x[-1] + 0.5), np.append(lim_min, lim_min[-1]),
                where="post", color="navy", linestyle="--", lw=1.2, label="limit min")
        for i in range(n):
            if violated[i]:
                ax.axvspan(i - 0.4, i + 0.4, color="red", alpha=0.12, zorder=1)
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("Joint position (rad)")
        ax.set_title("Joint positions: measured vs planned vs limits")
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(axis="y", alpha=0.4)

        # ── row 1: torques — ff vs cmd ────────────────────────────────────────
        ax = axes[1]
        has_tau = tau_ff is not None and len(tau_ff) >= n
        if has_tau:
            ax.plot(x, tau_ff, "o-", color="darkorange",
                    label="τ_ff (MPC)", markersize=6, zorder=5)
            self._annotate_pts(ax, x, tau_ff, "darkorange", 6)
            t_cmd = torques[:n] if len(torques) >= n else np.zeros(n)
            ax.plot(x, t_cmd, "s--", color="gray",
                    label="τ_cmd (clipped)", markersize=5, zorder=4)
            self._annotate_pts(ax, x, t_cmd, "gray", -10)
            ax.legend(fontsize=8)
        elif len(torques) > 0:
            n_t = min(n, len(torques))
            ax.plot(x[:n_t], torques[:n_t], "o-",
                    color="darkorange", markersize=6)
            self._annotate_pts(ax, x[:n_t], torques[:n_t], "darkorange", 6)
        ax.axhline(0, color="black", lw=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("Torque (Nm)")
        ax.set_title("Torques: feedforward vs commanded")
        ax.grid(axis="y", alpha=0.4)

        # ── row 2: velocities — measured vs planned ───────────────────────────
        if n_rows == 3:
            ax = axes[2]
            n_v = min(n, len(meas_v), len(v_plan))
            ax.plot(x[:n_v], meas_v[:n_v], "o-", color="steelblue",
                    label="measured dq", markersize=6, zorder=5)
            self._annotate_pts(ax, x[:n_v], meas_v[:n_v], "steelblue", 6)
            ax.plot(x[:n_v], v_plan[:n_v], "s--", color="mediumseagreen",
                    label="planned dq", markersize=5, zorder=4)
            self._annotate_pts(ax, x[:n_v], v_plan[:n_v], "green", -10)
            ax.set_xticks(x[:n_v])
            ax.set_xticklabels(names[:n_v], rotation=45,
                               ha="right", fontsize=8)
            ax.set_ylabel("Velocity (rad/s)")
            ax.set_title("Joint velocities: measured vs planned")
            ax.legend(fontsize=8, loc="upper right")
            ax.grid(axis="y", alpha=0.4)

        fig.tight_layout()
        fig.savefig(path, dpi=120, bbox_inches="tight")
        plt.close(fig)
