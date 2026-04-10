from typing import List, Tuple
import os
import time
import pinocchio as pin
import numpy as np
from contact_tamp.traj_opt_acados.models.floating_base_dynamics import FloatingBaseDynamics
from contact_tamp.traj_opt_acados.models.point_contact import PointContact
from contact_tamp.traj_opt_acados.utils.model_utils import toSymModel, loadModelImpl
from contact_tamp.traj_opt_acados.interface.acados_helper import ProblemFormulation, cs
from .transform import local_angular_to_euler_derivative, euler_derivative_to_local_angular

# ── dynamics conversion debugger ─────────────────────────────────────────────
_DYN_DBG_LOG = os.path.join(
    os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__)))),
    "debug_transform.log")
_dyn_call_count = 0


def _dyn_dbg_write(tag: str, **fields):
    global _dyn_call_count
    _dyn_call_count += 1
    if _dyn_call_count % 80 != 1:
        return
    lines = [f"\n{'='*60}",
             f"[{tag}]  call #{_dyn_call_count}  wall={time.time():.3f}"]
    for k, v in fields.items():
        if isinstance(v, np.ndarray):
            lines.append(f"  {k:35s}: {np.round(v.ravel(), 6).tolist()}")
        else:
            lines.append(f"  {k:35s}: {v}")
    with open(_DYN_DBG_LOG, "a") as fh:
        fh.write("\n".join(lines) + "\n")
# ─────────────────────────────────────────────────────────────────────────────


class QuadrupedDynamics(FloatingBaseDynamics):

    def __init__(self,
                 urdf_path,
                 feet_frame_names: List[str],
                 cnt_patch_restriction: bool = False,
                 mu_contact: float = 0.7,
                 torque_limit: bool = True,
                 ):
        # Load pinocchio model
        self.__raw_model = loadModelImpl(urdf_path)
        self.__raw_data = self.__raw_model.createData()
        # separate data for id_torques (thread-safe)
        self.__ff_data = self.__raw_model.createData()
        self.nu = self.__raw_model.nv - 6
        # Load symbolic model
        model, data = toSymModel(self.__raw_model)
        super().__init__(model.name, model, data)

        # Init point contact
        self.feet_frame_id = [model.getFrameId(
            ee_name) for ee_name in feet_frame_names]
        self.feet = [PointContact(
            dyn=self,
            frame=frame_name,
            mu=mu_contact,
            patch_restriction=cnt_patch_restriction) for frame_name in feet_frame_names]

        self.add_contacts(self.feet)
        self.base_cost = self.add_expr(
            name="base_cost", expr=self.get_base_cost())
        self.joint_cost = self.add_expr(
            name="joint_cost", expr=self.get_joint_cost())
        self.acc_cost = self.add_expr(
            name="acc_cost", expr=self.get_acc_cost())
        self.torque_limit = torque_limit

    @property
    def pin_model(self):
        return self.__raw_model

    @property
    def pin_data(self):
        return self.__raw_data

    def update_pin(self, q: np.ndarray, v: np.ndarray):
        pin.framesForwardKinematics(self.pin_model, self.pin_data, q)
        pin.computeCentroidalMomentum(self.pin_model, self.pin_data, q, v)

    @staticmethod
    def convert_from_mujoco(q_mj: np.ndarray, v_mj: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        # Convert state from MuJoCo to Pinocchio model format
        q = np.zeros(len(q_mj) - 1)
        q[:3] = q_mj[:3]
        R_WB = pin.Quaternion(
            w=q_mj[3],
            x=q_mj[4],
            y=q_mj[5],
            z=q_mj[6]).toRotationMatrix()
        q[5:2:-1] = pin.rpy.matrixToRpy(R_WB)
        q[6:] = q_mj[7:]
        # Convert velocties from MuJoCo to Pinocchio model format
        # MuJoCo is v global and w local
        v = v_mj.copy()
        # w local to euler derivatives (z y x)
        # https://github.com/ANYbotics/kindr/blob/master/doc/cheatsheet/cheatsheet_latest.pdf
        v[3:6] = local_angular_to_euler_derivative(q[3:6], v[3:6])

        # ── debug log ────────────────────────────────────────────────────────
        _dyn_dbg_write("CONVERT_FROM_MUJOCO_INPUT",
                       q_mj_pos=q_mj[:3],
                       q_mj_quat_wxyz=q_mj[3:7],
                       q_mj_quat_norm=float(np.linalg.norm(q_mj[3:7])),
                       v_mj_lin_world=v_mj[:3],
                       v_mj_ang_body=v_mj[3:6],
                       q_mj_joints=q_mj[7:],
                       )
        _dyn_dbg_write("CONVERT_FROM_MUJOCO_OUTPUT",
                       q_pos=q[:3],
                       q_yaw_pitch_roll_rad=q[3:6],
                       q_yaw_pitch_roll_deg=(np.degrees(q[3:6])),
                       v_lin_world=v[:3],
                       v_ang_euler_deriv=v[3:6],
                       q_joints=q[6:],
                       R_WB_det=float(np.linalg.det(R_WB)),
                       R_WB_row0=R_WB[0],
                       R_WB_row1=R_WB[1],
                       R_WB_row2=R_WB[2],
                       )
        # ─────────────────────────────────────────────────────────────────────

        return q, v

    @staticmethod
    def convert_to_mujoco(q: np.ndarray, v: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        # Convert state from MuJoCo to Pinocchio model format
        q_mj = np.zeros(len(q) + 1)
        R_WB = pin.rpy.rpyToMatrix(q[3:6][::-1])
        quat = pin.Quaternion(R_WB)
        q_mj[:3] = q[:3]
        q_mj[4:7] = quat.coeffs()[:-1]
        q_mj[3] = quat.coeffs()[-1]
        q_mj[7:] = q[6:]
        # Convert velocties from MuJoCo to Pinocchio model format

        R_WB = pin.Quaternion(
            w=q_mj[3],
            x=q_mj[4],
            y=q_mj[5],
            z=q_mj[6]).toRotationMatrix()
        q[3:6] = pin.rpy.matrixToRpy(R_WB)[::-1]
        q[6:] = q_mj[7:]
        # Convert velocties from MuJoCo to Pinocchio model format
        # MuJoCo is v global and w local
        v_mj = v.copy()
        # w local to euler derivatives (z y x)
        # https://github.com/ANYbotics/kindr/blob/master/doc/cheatsheet/cheatsheet_latest.pdf
        v_mj[3:6] = euler_derivative_to_local_angular(q[3:6], v[3:6])

        return q_mj, v_mj

    def get_feet_position_w(self):
        feet_pos = np.array([
            self.__raw_data.oMf[frame_id].translation
            for frame_id in
            self.feet_frame_id])

        return feet_pos

    def setup(self, problem: ProblemFormulation):
        super().setup(problem)
        if self.torque_limit:
            self.add_torque_limit(problem)

        problem.add_cost(self.base_cost, terminal=True)
        problem.add_cost(self.joint_cost, terminal=True)
        problem.add_cost(self.acc_cost)

    def get_hg(self):
        return self.h

    def get_base_cost(self):
        r = self.q[:3]  # position cost
        euler = self.q[3:6]
        return cs.vcat([r, euler, self.v[:6]])

    def get_joint_cost(self):
        return cs.vcat([self.q[6:], self.v[6:]])

    def get_acc_cost(self):
        return self.a[6:]

    def id_torques(self,
                   q_plan: np.ndarray,
                   v_plan: np.ndarray,
                   a_plan: np.ndarray,
                   f_plan: np.ndarray,
                   ) -> np.ndarray:
        """
        Return torques for desired position, velocity, acceleration
        and external forces plan.

        Args:
            q_plan (np.ndarray): State position plan [px, py, pz, y, p, r, joints]
            v_plan (np.ndarray): State velocity plan [vx, vy, vz, wz, wy, wx, joints]
            a_plan (np.ndarray): Acceleration plan   (idem)
            f_plan (np.ndarray): Contact forces plan

        Return:
            torques:
        """
        # Use a private pinocchio data to avoid racing with the
        # async optimizer that writes to self.pin_data concurrently.
        tau = pin.rnea(self.pin_model, self.__ff_data,
                       q_plan, v_plan, a_plan)[-self.nu:]

        # Loop through each end-effector and accumulate external forces
        for frame_id, f_ee in zip(self.feet_frame_id, f_plan):
            J_ee = pin.computeFrameJacobian(
                self.pin_model, self.__ff_data, q_plan, frame_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)[:3, -self.nu:]
            tau -= f_ee @ J_ee

        return tau
