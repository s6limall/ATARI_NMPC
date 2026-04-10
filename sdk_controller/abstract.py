from unitree_sdk2py.utils.crc import CRC
from unitree_sdk2py.idl.unitree_go.msg.dds_ import WirelessController_
from unitree_sdk2py.idl.unitree_go.msg.dds_ import SportModeState_
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowState_
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowCmd_
from unitree_sdk2py.idl.default import unitree_go_msg_dds__WirelessController_
from unitree_sdk2py.idl.default import unitree_go_msg_dds__SportModeState_
from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowState_
from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowCmd_
from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from unitree_sdk2py.core.channel import ChannelPublisher, ChannelSubscriber
from mj_pin.utils import get_robot_description, mj_joint_name2act_id, mj_joint_name2dof
from sdk_controller.joystick import KEY_MAP
import time
import os
import numpy as np
import mujoco
from abc import ABC, abstractmethod

from sdk_controller.topics import TOPIC_HIGHSTATE, TOPIC_LOWCMD, TOPIC_LOWSTATE, TOPIC_HIGHSTATE, TOPIC_WIRELESS_CONTROLLER

# ── Transform debugger ────────────────────────────────────────────────────────
_DBG_LOG = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "debug_transform.log")
_dbg_call_count = 0


def _dbg_write(tag: str, **fields):
    """Write one labelled block to the debug log every 80 calls (~1 s at 80 Hz)."""
    global _dbg_call_count
    _dbg_call_count += 1
    if _dbg_call_count % 80 != 1:
        return
    lines = [f"\n{'='*60}",
             f"[{tag}]  call #{_dbg_call_count}  wall={time.time():.3f}"]
    for k, v in fields.items():
        if isinstance(v, np.ndarray):
            lines.append(f"  {k:30s}: {np.round(v.ravel(), 6).tolist()}")
        else:
            lines.append(f"  {k:30s}: {v}")
    with open(_DBG_LOG, "a") as fh:
        fh.write("\n".join(lines) + "\n")

# ─────────────────────────────────────────────────────────────────────────────


try:
    from sdk_controller.safety import SafetyLayer
except:
    from safety import SafetyLayer


class SDKControllerBase(ABC):
    def __init__(self):
        super().__init__()

        self.last_high_state = None
        self.last_low_state = None
        self.last_wireless = None

        # joystick
        self.key_map = {v: k for k, v in KEY_MAP.items()}

        self.crc = CRC()
        self.cmd = unitree_go_msg_dds__LowCmd_()
        self.cmd.head[0] = 0xFE
        self.cmd.head[1] = 0xEF
        self.cmd.level_flag = 0xFF
        self.cmd.gpio = 0

        for i in range(20):
            self.cmd.motor_cmd[i].mode = 0x01  # (PMSM) mode
            self.cmd.motor_cmd[i].q = 0.0
            self.cmd.motor_cmd[i].kp = 0.0
            self.cmd.motor_cmd[i].dq = 0.0
            self.cmd.motor_cmd[i].kd = 0.0
            self.cmd.motor_cmd[i].tau = 0.0

        # Create a publisher to publish the data defined in UserData class
        self.pub = ChannelPublisher(TOPIC_LOWCMD, LowCmd_)
        self.pub.Init()

        # Create subscribers - store as instance attributes to prevent garbage collection
        self.low_state_sub = ChannelSubscriber(TOPIC_LOWSTATE, LowState_)
        self.high_state_sub = ChannelSubscriber(
            TOPIC_HIGHSTATE, SportModeState_)
        self.joystick_sub = ChannelSubscriber(
            TOPIC_WIRELESS_CONTROLLER, WirelessController_)
        self.low_state_sub.Init(self.low_state_handler, 10)
        self.high_state_sub.Init(self.high_state_handler, 10)
        self.joystick_sub.Init(self.wireless_handler, 10)

        print("Subscribers initialized, waiting for messages...")
        self.wait_subscriber()
        print("Controller ready!")

    def wait_subscriber(self) -> bool:
        timeout = 10.
        t = 0.
        sleep = 0.1
        while t < timeout:
            if (self.last_high_state is not None and
                    self.last_low_state is not None and
                    self.last_wireless is not None):
                print(f"All subscribers received messages after {t:.1f}s")
                return True
            status = (
                f"high={'OK' if self.last_high_state is not None else 'WAITING'}, "
                f"low={'OK' if self.last_low_state is not None else 'WAITING'}, "
                f"wireless={'OK' if self.last_wireless is not None else 'WAITING'}"
            )
            print(
                f"  Waiting for subscribers ({t:.1f}s/{timeout:.1f}s): {status}", flush=True)
            t += sleep
            time.sleep(sleep)

        # After timeout, report which subscribers are missing
        missing = []
        if self.last_high_state is None:
            missing.append("highstate")
        if self.last_low_state is None:
            missing.append("lowstate")
        if self.last_wireless is None:
            missing.append("wireless")

        if missing:
            print(
                f"WARNING: Timeout waiting for subscribers: {', '.join(missing)}")
            # lowstate requires robot to be in low-level SDK mode — warn but continue
            if self.last_high_state is None:
                raise TimeoutError(
                    "No highstate msg received. Is the Vicon/simulation running?")
            if self.last_wireless is None:
                raise TimeoutError(
                    "No wireless controller msg received. Is the joystick connected?")
            if self.last_low_state is None:
                print(
                    "WARNING: No lowstate received. Is the robot powered on and in low-level SDK mode?")
                print(
                    "WARNING: Continuing without lowstate — joint data will not be available until robot responds.")
        return True

    def wireless_handler(self, msg: WirelessController_):
        self.last_wireless = msg

    def low_state_handler(self, msg: LowState_):
        self.last_low_state = msg

    def high_state_handler(self, msg: SportModeState_):
        self.last_high_state = msg

    def get_last_key(self):
        if self.last_wireless.keys == 0.:
            return None
        key_id = int(self.last_wireless.keys)
        # print(f"Last key id: {key_id} -> {self.key_map[key_id]}")
        return self.key_map[key_id]


class SDKController(SDKControllerBase):
    def __init__(self,
                 simulate: bool,
                 robot_config,
                 xml_path: str = ""
                 ):
        self.simulate = simulate
        self.use_angular_from_highstate = not self.simulate

        self.robot_config = robot_config
        # Init robot interface, init joint mapping
        if not xml_path:
            desc = get_robot_description(robot_config.ROBOT_NAME)
            xml_path = desc.xml_scene_path

        mj_model = mujoco.MjModel.from_xml_path(xml_path)
        self.nu = mj_model.nu
        self.nq = mj_model.nq
        self.nv = mj_model.nv
        self._q = np.zeros(self.nq)
        self._v = np.zeros(self.nv)
        self.off = self.nq - self.nv
        joint_name2act_id = mj_joint_name2act_id(mj_model)
        joint_name2dof = mj_joint_name2dof(mj_model)
        self.joint_act_id2dof = {
            v: joint_name2dof[k] for k, v in joint_name2act_id.items()}
        self.joint_dof2act_id = {
            joint_name2dof[k]: v for k, v in joint_name2act_id.items()}

        # Safety layer
        self.safety = SafetyLayer(mj_model)

        self.t_last_key = 0.
        self.stand_up_start = 0.
        self.stand_down_start = 0.
        self.stand_up_running = False
        self.stand_down_running = False
        self.controller_running = False
        self.damping_running = False
        self.stand_up_duration = 3.5
        self.stand_down_duration = 3.5

        # ── diagnostic state ──────────────────────────────────────────────
        self._diag_highstate_wall = 0.0   # wall-clock of last HighState cb
        self._diag_lowstate_wall = 0.0    # wall-clock of last LowState cb
        self._diag_mj_time = 0.0          # mj_data.time from bridge
        self._diag_true_base_pos = np.zeros(3)  # true base COM pos
        self._diag_true_base_vel = np.zeros(3)  # true base COM vel
        # ─────────────────────────────────────────────────────────────────

        super().__init__()

    def high_state_handler(self, msg):
        super().high_state_handler(msg)
        self._diag_highstate_wall = time.perf_counter()
        # Extract diagnostic ground-truth from smuggled bridge fields
        # (only valid in simulation — on real robot these fields have real SDK meaning)
        if self.simulate:
            self._diag_mj_time = getattr(msg, 'foot_raise_height', 0.0)
            fpb = getattr(msg, 'foot_position_body', None)
            fsb = getattr(msg, 'foot_speed_body', None)
            if fpb is not None and len(fpb) >= 3:
                self._diag_true_base_pos[:] = fpb[:3]
            if fsb is not None and len(fsb) >= 3:
                self._diag_true_base_vel[:] = fsb[:3]
            self.update_q_v_from_highstate_simulation()
        else:
            self.update_q_v_from_highstate()

    def low_state_handler(self, msg):
        super().low_state_handler(msg)
        self._diag_lowstate_wall = time.perf_counter()
        self.update_q_v_from_lowstate()

    def wireless_handler(self, msg):
        # print("Initial wireless handler in SDKController called.")
        super().wireless_handler(msg)
        last_key = self.get_last_key()
        t = time.time()
        # Cannot change mode if already running
        # Change mode anytime while controller running
        if last_key:
            print(f"Key pressed: {last_key}")
            # Switching off from controller -> reset
            if (self.controller_running and last_key in "ABY"):
                self.controller_running = False
                self.reset_controller()
            # Damping mode
            if last_key == "B":
                if not self.damping_running:
                    print("Damping_mode")
                self.damping_running = True
                self.stand_down_running = False
                self.stand_up_running = False
                self.controller_running = False
            # If stand up/down is done
            elif not (
                (self.stand_down_running and t - self.t_last_key < self.stand_down_duration) or
                (self.stand_up_running and t -
                 self.t_last_key < self.stand_up_duration)
            ):
                self.t_last_key = t
                self.stand_up_start = 0.
                self.stand_down_start = 0.

                if last_key == "A":
                    print("Running stand down")
                    self.stand_down_running = True
                    self.stand_up_running = False
                    self.controller_running = False
                    self.damping_running = False
                elif last_key == "Y":
                    print("Running stand up")
                    self.stand_up_running = True
                    self.stand_down_running = False
                    self.controller_running = False
                    self.damping_running = False
                elif last_key == "X":
                    if not self.controller_running:
                        print("Running controller")
                    self.controller_running = True
                    self.stand_down_running = False
                    self.stand_up_running = False
                    self.damping_running = False
            # Wait if stand up/down not finished
            else:
                if self.stand_down_running and not last_key == "A":
                    print("Waiting for stand down to finish")
                elif self.stand_up_running and not last_key == "Y":
                    print("Waiting for stand up to finish")

    def update_q_v_from_lowstate(self):
        """Extracts q (position) and v (velocity) from a Unitree LowState_ message."""
        if not self.use_angular_from_highstate:
            # Base orientation (quaternion w, x, y, z)
            # Quaternion order: w, x, y, z
            self._q[3:7] = self.last_low_state.imu_state.quaternion
            # Base velocity (linear and angular) - extracted from IMU
            # Angular velocity from IMU
            self._v[3:6] = self.last_low_state.imu_state.gyroscope

        # Joint positions and velocities
        for i, motor in enumerate(self.last_low_state.motor_state[:self.nu]):
            dof = self.joint_act_id2dof[i]
            self._q[dof + self.off] = motor.q  # Joint position
            self._v[dof] = motor.dq  # Joint velocity

        _dbg_write("LOWSTATE",
                   imu_quat=np.array(self.last_low_state.imu_state.quaternion),
                   imu_quat_norm=float(np.linalg.norm(
                       self.last_low_state.imu_state.quaternion)),
                   imu_gyro=np.array(self.last_low_state.imu_state.gyroscope),
                   q_3_7_stored=self._q[3:7].copy(),
                   v_3_6_stored=self._v[3:6].copy(),
                   joint_q=self._q[7:].copy(),
                   )

    def update_q_v_from_highstate_simulation(self):
        """Converts IMU velocity to base frame velocity."""
        if self._q is None or self._v is None:
            return

        p_imu_B = self.robot_config.P_IMU_IN_BASE
        R_imu_B = self.robot_config.R_IMU_IN_BASE

        # ── debug: snapshot inputs before transformation ───────────────────
        _dbg_imu_quat_in = self._q[3:7].copy()
        _dbg_high_pos = np.array(self.last_high_state.position[:3])
        _dbg_high_vel = np.array(self.last_high_state.velocity[:3])
        _dbg_write("HIGHSTATE_SIM_INPUT",
                   imu_quat_from_lowstate=_dbg_imu_quat_in,
                   imu_quat_norm=float(np.linalg.norm(_dbg_imu_quat_in)),
                   high_state_position=_dbg_high_pos,
                   high_state_velocity=_dbg_high_vel,
                   P_IMU_IN_BASE=p_imu_B,
                   R_IMU_IN_BASE=R_imu_B.ravel(),
                   angular_vel_body=self._v[3:6].copy(),
                   )
        # ──────────────────────────────────────────────────────────────────

        # Convert quaternion to rotation matrix
        R_IMU_W_flat = np.zeros(9)
        mujoco.mju_quat2Mat(R_IMU_W_flat, self._q[3:7])
        R_imu_W = R_IMU_W_flat.reshape((3, 3), order='A')
        # Get base position
        self._q[0:3] = R_imu_W @ (- R_imu_B.T @ p_imu_B) + \
            self.last_high_state.position
        R_B_W = R_imu_W @ R_imu_B.T
        mujoco.mju_mat2Quat(self._q[3:7], R_B_W.reshape(-1, order='A'))
        # Linear velocity. Same angular velocity
        self._v[0:3] = self.last_high_state.velocity + \
            np.cross(R_B_W @ self._v[3:6], self._q[0:3] -
                     self.last_high_state.position)

        # ── debug: snapshot outputs after transformation ───────────────────
        _dbg_write("HIGHSTATE_SIM_OUTPUT",
                   base_position_q=self._q[0:3].copy(),
                   base_quat_out=self._q[3:7].copy(),
                   base_quat_out_norm=float(np.linalg.norm(self._q[3:7])),
                   base_lin_vel=self._v[0:3].copy(),
                   base_ang_vel_body=self._v[3:6].copy(),
                   R_W_IMU_row0=R_imu_W[0].copy(),
                   R_W_IMU_row1=R_imu_W[1].copy(),
                   R_W_IMU_row2=R_imu_W[2].copy(),
                   R_W_IMU_det=float(np.linalg.det(R_imu_W)),
                   R_B_W_row0=R_B_W[0].copy(),
                   R_B_W_row1=R_B_W[1].copy(),
                   R_B_W_row2=R_B_W[2].copy(),
                   R_B_W_det=float(np.linalg.det(R_B_W)),
                   lever_arm=self._q[0:3] - _dbg_high_pos,
                   )
        # ──────────────────────────────────────────────────────────────────

    def update_q_v_from_highstate(self):
        """Extracts q (position) and v (velocity) from a Unitree HighState_ message."""
        # Base position
        self._q[0:3] = self.last_high_state.position
        self._v[0:3] = self.last_high_state.velocity
        if self.use_angular_from_highstate:
            # Base orientation (quaternion w, x, y, z)
            self._q[3:7] = self.last_high_state.imu_state.quaternion
            self._v[3:6] = self.last_high_state.imu_state.gyroscope

    def send_motor_command(self, time: float):

        # Stand down
        if self.stand_down_running:
            self.stand_down_motor_cmd(time)
        # Stand up
        elif self.stand_up_running:
            self.stand_up_motor_cmd(time)
        # Controller
        elif self.controller_running:
            self.update_motor_cmd(time)
            # Check state and action
            safe = self.safety.check_safety(
                self._q,
                [motor_cmd.tau for motor_cmd in self.cmd.motor_cmd],
                self._q[3:7],
            )
            # Go to damping mode if not safe
            if not safe:
                self.reset_controller()
                self.damping_running = True
                self.controller_running = False
                self.damping_motor_cmd()
        else:
            self.damping_motor_cmd()

        if self.damping_running:
            self.damping_motor_cmd()

        self.cmd.crc = self.crc.Crc(self.cmd)
        self.pub.Write(self.cmd)

    def stand_up_motor_cmd(self, time: float) -> None:
        if self.stand_up_start == 0. and time > 0.:
            self.stand_up_start = time

        t = time - self.stand_up_start
        phase = np.tanh(t / 1.3)
        for i in range(self.nu):
            self.cmd.motor_cmd[i].q = phase * self.robot_config.STAND_UP_JOINT_POS[i] + (
                1 - phase) * self.robot_config.STAND_DOWN_JOINT_POS[i]
            self.cmd.motor_cmd[i].kp = phase * 50.0 + (1 - phase) * 20.0
            self.cmd.motor_cmd[i].dq = 0.0
            self.cmd.motor_cmd[i].kd = 3.5
            self.cmd.motor_cmd[i].tau = 0.0

    def stand_down_motor_cmd(self, time: float) -> None:
        if self.stand_down_start == 0. and time > 0.:
            self.stand_down_start = time

        t = time - self.stand_down_start

        phase = np.tanh(t / 1.1)
        for i in range(self.nu):
            self.cmd.motor_cmd[i].q = phase * self.robot_config.STAND_DOWN_JOINT_POS[i] + (
                1 - phase) * self.robot_config.STAND_UP_JOINT_POS[i]
            self.cmd.motor_cmd[i].kp = 50.0
            self.cmd.motor_cmd[i].dq = 0.0
            self.cmd.motor_cmd[i].kd = 3.5
            self.cmd.motor_cmd[i].tau = 0.0

    def damping_motor_cmd(self) -> None:
        for i in range(self.nu):
            self.cmd.motor_cmd[i].q = 0.
            self.cmd.motor_cmd[i].kp = 0.
            self.cmd.motor_cmd[i].dq = 0.0
            self.cmd.motor_cmd[i].kd = 2.
            self.cmd.motor_cmd[i].tau = 0.0

    def update_motor_cmd(self, time: float) -> None:
        pass

    def reset_controller(self) -> None:
        pass


if __name__ == "__main__":
    import robots.Go2 as Go2

    ChannelFactoryInitialize(1, "lo")
    controller = SDKController(Go2)

    running_time = 0.
    controller_time = 20.
    freq = 100
    dt = 1 / freq
    while running_time < controller_time:
        controller.send_motor_command(running_time)
        running_time += dt
        time.sleep(dt)
