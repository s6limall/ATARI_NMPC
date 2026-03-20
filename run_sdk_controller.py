from unitree_sdk2py.utils.crc import CRC
from unitree_sdk2py.idl.unitree_go.msg.dds_ import WirelessController_
from unitree_sdk2py.idl.default import unitree_go_msg_dds__WirelessController_
from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from mj_pin.utils import get_robot_description
from sdk_controller.abstract import SDKController
from mpc_controller.config.quadruped.utils import get_quadruped_config
from mpc_controller.mpc import LocomotionMPC
import time
import sys
import numpy as np

import faulthandler
faulthandler.enable()


class MPC_SDK(SDKController):
    def __init__(self,
                 simulate: bool,
                 mpc: LocomotionMPC,
                 robot_config,
                 xml_path="",
                 v_max=0.5,
                 w_max=0.5):
        self.mpc = mpc
        self.mpc.scale_joint = np.repeat([1.4, 1.2, 1], 4)

        self.v_max = v_max
        self.w_max = w_max
        self.v_des = np.zeros(3)
        self.w_des = 0.
        super().__init__(simulate, robot_config, xml_path)

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
        else:
            self.mpc.tau_full.append(torques_ff)
            step = self.mpc.plan_step + 2
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
                self.cmd.motor_cmd[i_act].tau = np.clip(tau, -max_tau, max_tau)

    def reset_controller(self):
        print("reset controller")
        time.sleep(0.1)
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
        joint_ref=robot_desc.q0,
        sim_dt=dt,
        print_info=True,
        solve_async=True,
    )
    print("MPC initialized.")

    sdk_controller = MPC_SDK(simulate, mpc, Go2)

    print("Starting main control loop. Press Ctrl+C to exit.")
    last_heartbeat = 0.
    loop_count = 0

    try:
        wall_start = time.perf_counter()
        stall_threshold_ms = 50.0  # warn if a single loop takes > 50ms
        while True:
            step_start = time.perf_counter()

            try:
                sdk_controller.send_motor_command(runing_time)
            except Exception as e:
                print(
                    f"ERROR in send_motor_command at t={runing_time:.3f}s: {e}", flush=True)
                import traceback
                traceback.print_exc()
                break

            step_time_ms = (time.perf_counter() - step_start) * 1000
            loop_count += 1

            # Warn on any loop that took too long (GIL stall / blocking call)
            if step_time_ms > stall_threshold_ms:
                print(
                    f"  STALL loop={loop_count} t={runing_time:.2f}s step={step_time_ms:.1f}ms", flush=True)

            # Periodic heartbeat every 2 seconds (use wall-clock to survive stalls)
            wall_now = time.perf_counter() - wall_start
            if wall_now - last_heartbeat >= 2.0:
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

            runing_time += dt
            time_until_next_step = dt - (time.perf_counter() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

    except KeyboardInterrupt:
        print("\nShutting down...")
        try:
            print(mpc.print_timings())
        except Exception as e:
            print(f"Error printing timings: {e}")
        # Force exit — background DDS/Vicon/Joystick threads won't stop on their own
        import os
        os._exit(0)
