from unitree_sdk2py.utils.crc import CRC
from unitree_sdk2py.idl.unitree_go.msg.dds_ import WirelessController_
from unitree_sdk2py.idl.default import unitree_go_msg_dds__WirelessController_
from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from mj_pin.utils import get_robot_description
from sdk_controller.abstract import SDKController
from mpc_controller.config.quadruped.utils import get_quadruped_config
from mpc_controller.mpc import LocomotionMPC
import time as time_module
import sys
import os
import numpy as np

import faulthandler
faulthandler.enable()

# ── initialise debug log (overwrite previous run) ─────────────────────────────
_DEBUG_LOG = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "debug_transform.log")
with open(_DEBUG_LOG, "w") as _fh:
    _fh.write(
        f"Transform Debug Log – started {time_module.strftime('%Y-%m-%d %H:%M:%S')}\n")
    _fh.write("Sections: LOWSTATE | HIGHSTATE_SIM_INPUT | HIGHSTATE_SIM_OUTPUT |\n")
    _fh.write("          COMPUTE_FF_INPUT | CONVERT_FROM_MUJOCO_INPUT |\n")
    _fh.write("          CONVERT_FROM_MUJOCO_OUTPUT | COMPUTE_FF_OUTPUT\n")
    _fh.write("="*80 + "\n")
print(f"[DEBUG] Transform debug log: {_DEBUG_LOG}")
# ─────────────────────────────────────────────────────────────────────────────


class MPC_SDK(SDKController):
    def __init__(self,
                 simulate: bool,
                 mpc: LocomotionMPC,
                 robot_config,
                 xml_path="",
                 v_max=0.5,
                 w_max=0.5):
        self.mpc = mpc
        self.mpc.scale_joint = np.tile([2., 1.5, 1.], 4)

        self.v_max = v_max
        self.w_max = w_max
        self.v_des = np.zeros(3)
        self.w_des = 0.

        # Per-step time-series logging (measured vs planned)
        self._log_time = []
        self._log_meas_q = []   # measured joint positions (12,)
        self._log_meas_v = []   # measured joint velocities (12,)
        self._log_plan_q = []   # planned joint positions (12,)
        self._log_plan_v = []   # planned joint velocities (12,)
        self._log_tau_ff = []   # feedforward torques (12,)
        self._log_tau_cmd = []  # clipped commanded torques (12,)
        self._log_start_stamp = time_module.strftime("%H%M%S")

        # ── diagnostic arrays (Phase 1 instrumentation) ─────────────────
        self._log_diag_mj_time = []       # mj_data.time from bridge
        self._log_diag_runing_time = []   # software clock (runing_time)
        self._log_diag_hs_age = []        # HighState callback age (ms)
        self._log_diag_ls_age = []        # LowState callback age (ms)
        self._log_diag_base_pos_err = []  # _q[0:3] - true_base_pos (3,)
        self._log_diag_base_vel_err = []  # _v[0:3] - true_base_vel (3,)
        self._log_diag_plan_step = []     # (plan_step, n_interp_plan, delay)
        self._log_diag_base_pos = []      # _q[0:3] (reconstructed) (3,)
        self._log_diag_true_base_pos = []  # true base pos from bridge (3,)
        # ──────────────────────────────────────────────────────────────

        super().__init__(simulate, robot_config, xml_path)
        self._joint_names = [f"joint_{i}" for i in range(self.nu)]

    def wireless_handler(self, msg: WirelessController_):
        super().wireless_handler(msg)
        v_des = np.round([msg.ly * self.v_max, -msg.lx * self.v_max, 0.], 2)
        w_yaw = -round(msg.rx * self.w_max, 1)

        self.mpc.set_command(v_des=v_des, w_yaw=w_yaw)

    def update_motor_cmd(self, time):
        torques_ff = self.mpc._compute_torques_ff(time, self._q, self._v)
        if self.mpc.first_solve:
            # Stand up
            phase = 1.
            for i in range(self.nu):
                self.cmd.motor_cmd[i].q = phase * self.robot_config.STAND_UP_JOINT_POS[i] + (
                    1 - phase) * self.robot_config.STAND_DOWN_JOINT_POS[i]
                self.cmd.motor_cmd[i].kp = phase * 50.0 + (1 - phase) * 20.0
                self.cmd.motor_cmd[i].dq = 0.0
                self.cmd.motor_cmd[i].kd = 3.5
                self.cmd.motor_cmd[i].tau = 0.0

            # Record during stand-up as well, so periodic log saving starts early.
            self._log_time.append(time)
            self._log_meas_q.append(self._q[7:].copy())
            self._log_meas_v.append(self._v[6:].copy())
            # Build plan arrays in MuJoCo DOF order to match meas_q / meas_v.
            # self._q[7:] is indexed by MuJoCo DOF (via joint_act_id2dof), so we
            # must use the same mapping here instead of raw actuator order.
            stand_plan_q = np.zeros(self.nu)
            stand_plan_v = np.zeros(self.nu)
            stand_tau_cmd = np.zeros(self.nu)
            for i_act in range(self.nu):
                dof = self.joint_act_id2dof[i_act]  # MuJoCo DOF index (6..17)
                j = dof - 6                          # 0-based DOF offset
                stand_plan_q[j] = self.cmd.motor_cmd[i_act].q
                stand_plan_v[j] = self.cmd.motor_cmd[i_act].dq
                stand_tau_cmd[j] = self.cmd.motor_cmd[i_act].tau
            self._log_plan_q.append(stand_plan_q)
            self._log_plan_v.append(stand_plan_v)
            self._log_tau_ff.append(np.zeros(self.nu))
            self._log_tau_cmd.append(stand_tau_cmd)
            # ── diag: record even during stand-up ───────────────────────
            self._record_diagnostics(time, now=time_module.perf_counter())
        else:
            self.mpc.tau_full.append(torques_ff)
            step = min(self.mpc.plan_step, self.mpc.n_interp_plan - 1)
            scale = 1. if self.simulate else self.robot_config.scale_gains
            for i, tau in enumerate(torques_ff, start=6):
                i_act = self.joint_dof2act_id[i]
                self.cmd.motor_cmd[i_act].q = self.mpc.q_plan[step, i]
                self.cmd.motor_cmd[i_act].kp = self.mpc.scale_joint[i -
                                                                    6] * self.robot_config.Kp * scale
                self.cmd.motor_cmd[i_act].dq = self.mpc.v_plan[step, i]
                self.cmd.motor_cmd[i_act].kd = self.mpc.scale_joint[i -
                                                                    6] * self.robot_config.Kd * scale
                max_tau = self.safety.torque_limits[i_act]
                self.cmd.motor_cmd[i_act].tau = np.clip(
                    tau, -max_tau, max_tau)

            # cmd_step is the look-ahead step used for the PD reference (safety ctx).
            cmd_step = min(step, len(self.mpc.q_plan) - 1)
            # log_step: the step the feedforward torque was computed for
            # (plan_step - 1, since _step() already incremented plan_step).
            # Logging at log_step gives an apples-to-apples comparison:
            # "plan at time t" vs "measurement at time t".
            log_step = min(max(self.mpc.plan_step - 1, 0),
                           self.mpc.n_interp_plan - 1)
            q_plan_pin = self.mpc.q_plan[log_step]
            q_plan_cmd = self.mpc.q_plan[cmd_step]
            # map pin DOF 6..17 → MuJoCo qpos 7..18 via joint_dof2act_id
            q_plan_mj_joints = np.zeros(self.nu)
            for i_pin in range(6, 6 + self.nu):
                i_act = self.joint_dof2act_id.get(i_pin)
                if i_act is not None:
                    q_plan_mj_joints[i_pin - 6] = q_plan_cmd[i_pin]
            self.safety.set_debug_context(
                q_plan_joints=q_plan_mj_joints,
                v_plan_joints=self.mpc.v_plan[cmd_step, 6:],
                torques_ff=torques_ff,
                measured_q=self._q.copy(),
                measured_v=self._v.copy(),
                plan_step=np.array([cmd_step]),
                sim_time=np.array([time]),
            )

            # ── record time-series data ──────────────────────────────────────
            self._log_time.append(time)
            self._log_meas_q.append(self._q[7:].copy())        # MuJoCo joint q
            self._log_meas_v.append(
                self._v[6:].copy())        # MuJoCo joint dq
            # Log the plan at the same instant as the measurement (log_step),
            # not the look-ahead step, so visualisation shows true tracking error.
            self._log_plan_q.append(
                q_plan_pin[6:].copy())     # Pinocchio plan q
            self._log_plan_v.append(self.mpc.v_plan[log_step, 6:].copy())
            self._log_tau_ff.append(torques_ff.copy())
            tau_cmd = np.array([self.cmd.motor_cmd[self.joint_dof2act_id[i]].tau
                                for i in range(6, 6 + self.nu)])
            self._log_tau_cmd.append(tau_cmd)
            # ── diag: record during MPC control ─────────────────────────
            self._record_diagnostics(time, now=time_module.perf_counter())

    def _record_diagnostics(self, ctrl_time, now):
        """Record per-step diagnostic data for hypothesis testing."""
        self._log_diag_mj_time.append(self._diag_mj_time)
        self._log_diag_runing_time.append(ctrl_time)
        hs_age = (
            now - self._diag_highstate_wall) if self._diag_highstate_wall > 0 else -1.0
        ls_age = (
            now - self._diag_lowstate_wall) if self._diag_lowstate_wall > 0 else -1.0
        self._log_diag_hs_age.append(hs_age)
        self._log_diag_ls_age.append(ls_age)
        self._log_diag_base_pos_err.append(
            self._q[0:3].copy() - self._diag_true_base_pos)
        self._log_diag_base_vel_err.append(
            self._v[0:3].copy() - self._diag_true_base_vel)
        ps = self.mpc.plan_step if hasattr(self.mpc, 'plan_step') else -1
        nip = self.mpc.n_interp_plan if hasattr(
            self.mpc, 'n_interp_plan') else -1
        dly = self.mpc.delay if hasattr(self.mpc, 'delay') else -1
        self._log_diag_plan_step.append(
            np.array([ps, nip, dly], dtype=np.float64))
        self._log_diag_base_pos.append(self._q[0:3].copy())
        self._log_diag_true_base_pos.append(self._diag_true_base_pos.copy())

    def save_trajectory_log(self):
        """Save recorded time-series to .npz for visualization."""
        if len(self._log_time) < 2:
            print("[LOG] Not enough data to save trajectory log.")
            return
        stamp = self._log_start_stamp
        log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "trajectory_logs")
        os.makedirs(log_dir, exist_ok=True)
        path = os.path.join(log_dir, f"traj_{stamp}.npz")

        np.savez(path,
                 time=np.array(self._log_time),
                 meas_q=np.array(self._log_meas_q),
                 meas_v=np.array(self._log_meas_v),
                 plan_q=np.array(self._log_plan_q),
                 plan_v=np.array(self._log_plan_v),
                 tau_ff=np.array(self._log_tau_ff),
                 tau_cmd=np.array(self._log_tau_cmd),
                 joint_names=np.array(self._joint_names),
                 dt=dt,
                 # ── Phase 1 diagnostics ──────────────────────────────
                 diag_mj_time=np.array(self._log_diag_mj_time),
                 diag_runing_time=np.array(self._log_diag_runing_time),
                 diag_hs_age=np.array(self._log_diag_hs_age),
                 diag_ls_age=np.array(self._log_diag_ls_age),
                 diag_base_pos_err=np.array(
                     self._log_diag_base_pos_err) if self._log_diag_base_pos_err else np.empty((0, 3)),
                 diag_base_vel_err=np.array(
                     self._log_diag_base_vel_err) if self._log_diag_base_vel_err else np.empty((0, 3)),
                 diag_plan_step=np.array(
                     self._log_diag_plan_step) if self._log_diag_plan_step else np.empty((0, 3)),
                 diag_base_pos=np.array(
                     self._log_diag_base_pos) if self._log_diag_base_pos else np.empty((0, 3)),
                 diag_true_base_pos=np.array(
                     self._log_diag_true_base_pos) if self._log_diag_true_base_pos else np.empty((0, 3)),
                 # ── MPC-level diagnostics ────────────────────────────
                 mpc_diag_plan_step=np.array(
                     self.mpc._diag_plan_step) if self.mpc._diag_plan_step else np.empty(0),
                 mpc_diag_plan_overflow=np.array(
                     self.mpc._diag_plan_overflow) if self.mpc._diag_plan_overflow else np.empty(0),
                 mpc_diag_delay=np.array(
                     self.mpc._diag_delay) if self.mpc._diag_delay else np.empty(0),
                 mpc_diag_replan_events=np.array(
                     self.mpc._diag_replan_events) if self.mpc._diag_replan_events else np.empty((0, 4)),
                 mpc_diag_t=np.array(
                     self.mpc._diag_t) if self.mpc._diag_t else np.empty(0),
                 mpc_diag_t0=np.array(
                     self.mpc._diag_t0) if self.mpc._diag_t0 else np.empty(0),
                 mpc_diag_current_opt_node=np.array(
                     self.mpc._diag_current_opt_node) if self.mpc._diag_current_opt_node else np.empty(0),
                 mpc_n_interp_plan=np.array([self.mpc.n_interp_plan]),
                 )
        print(
            f"[LOG] Trajectory log saved → {path}  ({len(self._log_time)} steps)")

    def reset_controller(self):
        print("reset controller")
        time_module.sleep(0.1)
        self.mpc.reset()


runing_time = 0.0

VICON_IP = "131.220.7.195:801"

if __name__ == '__main__':
    from sdk_controller.robots import Go2
    from sdk_controller.joystick import JoystickPublisher
    from sdk_controller.vicon_publisher import ViconHighStatePublisher

    if len(sys.argv) < 2:
        interface = "lo"
        ChannelFactoryInitialize(1, interface)
        print(
            f"Initializing SDK with interface: {interface}, domain: 1 (simulation mode)")
        joystick = JoystickPublisher(device_id=0, js_type="logitech")
        simulate = True
    else:
        interface = sys.argv[1]
        interface = "enx503eaadfddca"
        ChannelFactoryInitialize(0, interface)
        print(
            f"Initializing SDK with interface: {interface}, domain: 0 (hardware mode)")
        joystick = JoystickPublisher(device_id=0, js_type="logitech")
        simulate = False
        vicon = ViconHighStatePublisher(
            vicon_ip=VICON_IP,
            object_name=Go2.OBJECT_NAME,
            publish_freq=Go2.CONTROL_FREQ,
        )

    dt = 1 / Go2.CONTROL_FREQ
    robot_desc = get_robot_description(Go2.ROBOT_NAME)
    feet_frame_names = ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]

    # Build MPC joint reference from STAND_UP pose (actuator → pinocchio order).
    # robot_desc.q0 has thigh=0.9, calf=−1.8 (z≈0.27 m) which conflicts with
    # the trot nom_height=0.35 m.  STAND_UP (thigh≈0.61, calf≈−1.22) gives
    # z≈0.35 m, eliminating the cost‑function mismatch that destabilised the
    # first MPC plan.
    import mujoco as _mj
    from mj_pin.utils import mj_joint_name2act_id as _n2a, mj_joint_name2dof as _n2d
    _mj_model = _mj.MjModel.from_xml_path(robot_desc.xml_path)
    _n2a_map, _n2d_map = _n2a(_mj_model), _n2d(_mj_model)
    _joint_ref_pin = np.zeros(12)
    for _jn, _aid in _n2a_map.items():
        _joint_ref_pin[_n2d_map[_jn] - 6] = Go2.STAND_UP_JOINT_POS[_aid]

    gait_name = "trot"
    config_gait, config_opt, config_cost = get_quadruped_config(
        gait_name, Go2.ROBOT_NAME)
    config_gait.nominal_period = 0.5
    config_opt.recompile = True

    print(f"Initializing MPC with gait: {gait_name}")
    mpc = LocomotionMPC(
        path_urdf=robot_desc.urdf_path,
        feet_frame_names=feet_frame_names,
        config_opt=config_opt,
        config_gait=config_gait,
        config_cost=config_cost,
        joint_ref=_joint_ref_pin,
        sim_dt=dt,
        print_info=True,
        solve_async=True,
    )
    print("MPC initialized.")

    sdk_controller = MPC_SDK(simulate, mpc, Go2)

    print("Starting main control loop. Press Ctrl+C to exit.")
    last_heartbeat = 0.
    last_traj_save = 0.
    loop_count = 0
    mj_time_offset = None  # set on first valid mj_data.time from bridge

    try:
        wall_start = time_module.perf_counter()
        stall_threshold_ms = 50.0  # warn if a single loop takes > 50ms
        while True:
            step_start = time_module.perf_counter()

            try:
                sdk_controller.send_motor_command(runing_time)
            except Exception as e:
                print(
                    f"ERROR in send_motor_command at t={runing_time:.3f}s: {e}", flush=True)
                import traceback
                traceback.print_exc()
                break

            step_time_ms = (time_module.perf_counter() - step_start) * 1000
            loop_count += 1

            # Warn on any loop that took too long (GIL stall / blocking call)
            if step_time_ms > stall_threshold_ms:
                print(
                    f"  STALL loop={loop_count} t={runing_time:.2f}s step={step_time_ms:.1f}ms", flush=True)

            # Periodic heartbeat every 10 seconds (use wall-clock to survive stalls)
            wall_now = time_module.perf_counter() - wall_start

            # Save trajectory log every 10 seconds
            if wall_now - last_traj_save >= 10.0 and len(sdk_controller._log_time) > 1:
                try:
                    sdk_controller.save_trajectory_log()
                except Exception as e:
                    print(f"Error saving trajectory log: {e}", flush=True)
                last_traj_save = wall_now

            if wall_now - last_heartbeat >= 10.0:
                mode = "CONTROLLER" if sdk_controller.controller_running else \
                       "STAND_UP" if sdk_controller.stand_up_running else \
                       "STAND_DOWN" if sdk_controller.stand_down_running else \
                       "DAMPING" if sdk_controller.damping_running else "IDLE"
                print(
                    f"[wall={wall_now:.1f}s sim={runing_time:.1f}s] mode={mode} loops={loop_count} step={step_time_ms:.1f}ms", flush=True)
                last_heartbeat = wall_now

            # Print first few loops to verify it's running
            if loop_count <= 3:
                print(
                    f"  Loop {loop_count}: send_motor_command took {step_time_ms:.1f}ms", flush=True)

            # Sync runing_time: sim → mj_data.time from bridge; hardware → wall-clock
            if simulate:
                mj_t = sdk_controller._diag_mj_time
                if mj_t > 0:
                    if mj_time_offset is None:
                        mj_time_offset = mj_t - runing_time
                    runing_time = mj_t - mj_time_offset
                else:
                    runing_time += dt
            else:
                runing_time = time_module.perf_counter() - wall_start
            time_until_next_step = dt - \
                (time_module.perf_counter() - step_start)
            if time_until_next_step > 0:
                time_module.sleep(time_until_next_step)

    except KeyboardInterrupt:
        print("\nShutting down...")
        try:
            sdk_controller.save_trajectory_log()
        except Exception as e:
            print(f"Error saving trajectory log: {e}")
        try:
            print(mpc.print_timings())
        except Exception as e:
            print(f"Error printing timings: {e}")
        # Force exit — background DDS/Vicon/Joystick threads won't stop on their own
        import os
        os._exit(0)
