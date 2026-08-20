#!/usr/bin/env python3
"""Hardware-side full-6x6 GIRAF receiver for final OptiTrack teleoperation.

Runs only inside anymal_custom_control. It never opens NatNet or the pedal.
Fresh, engaged ROS commands are mapped through the physical 6x6 Jacobian to
the three MD80 arm joints and Dynamixel wrist IDs 21-23. ID 24 stays closed.
"""

import argparse
import math
import threading
import time
import traceback

import numpy as np
import rospy
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool, String

from anymal_custom_control.RRP_kinematic_model import (
    get_boom_length_d3,
    get_boom_motor_rad,
)
from anymal_custom_control.RRPRRR_kinematic_model import (
    num_forward_transform,
    num_jacobian,
)
from anymal_custom_control.control.giraf_arm_common import (
    BOOM_MAX,
    BOOM_MIN,
    D3_MIN,
    PITCH_KIN_OFFSET,
    PITCH_MAX,
    PITCH_MIN,
    ROLL_LIMIT,
    THETA4_DXL_SIGN,
    THETA4_KIN_OFFSET,
    THETA5_DXL_SIGN,
    THETA5_KIN_OFFSET,
    THETA6_KIN_OFFSET,
)
from anymal_custom_control.dynamixel import (
    ARM_HOME,
    ARM_IDS,
    ARM_TICK_LIMITS,
    GRIPPER_CLOSED,
    GRIPPER_IDS,
    GRIPPER_OPEN,
    dynamixel_connect,
    dynamixel_disconnect,
    dynamixel_drive,
    dynamixel_read,
    radians_to_ticks,
)
from anymal_custom_control.motor_driver import (
    motor_connect,
    motor_disconnect,
    motor_drive,
)


POSE_TOPIC = "/giraf_final/relative_pose_cmd"
ENGAGED_TOPIC = "/giraf_final/engaged"
GRIPPER_TOPIC = "/giraf_final/gripper_open"
STATUS_TOPIC = "/giraf_final/hardware_status"
MD80_STATE_TOPIC = "/md80/joint_states"

CONTROL_HZ = 100.0
COMMAND_TIMEOUT_SEC = 0.15
MD80_TIMEOUT_SEC = 0.30
POSITION_GAIN = 1.0
ROTATION_GAIN = 1.0
LINEAR_LIMIT = np.array((0.05, 0.05, 0.025), dtype=float)
ANGULAR_LIMIT = np.array((0.125, 0.125, 0.125), dtype=float)
LINEAR_DEADBAND = 0.002
ANGULAR_DEADBAND = 0.01
MAX_JOINT_SPEED = np.array((0.10, 0.10, 0.025, 0.15, 0.15, 0.15))
JACOBIAN_RCOND = 1e-3


def normalize_quaternion(value):
    quaternion = np.asarray(value, dtype=float)
    norm = float(np.linalg.norm(quaternion))
    if quaternion.shape != (4,) or not np.all(np.isfinite(quaternion)) or norm <= 1e-12:
        raise ValueError("invalid quaternion")
    return quaternion / norm


def quaternion_to_matrix(value):
    x, y, z, w = normalize_quaternion(value)
    return np.array(
        (
            (1 - 2*(y*y + z*z), 2*(x*y - z*w), 2*(x*z + y*w)),
            (2*(x*y + z*w), 1 - 2*(x*x + z*z), 2*(y*z - x*w)),
            (2*(x*z - y*w), 2*(y*z + x*w), 1 - 2*(x*x + y*y)),
        ),
        dtype=float,
    )


def rotation_vector(rotation):
    cosine = float(np.clip((np.trace(rotation) - 1.0) * 0.5, -1.0, 1.0))
    angle = math.acos(cosine)
    skew = np.array(
        (
            rotation[2, 1] - rotation[1, 2],
            rotation[0, 2] - rotation[2, 0],
            rotation[1, 0] - rotation[0, 1],
        ),
        dtype=float,
    )
    if angle < 1e-7:
        return 0.5 * skew
    if math.pi - angle < 1e-5:
        raise RuntimeError("orientation error is too close to 180 degrees")
    return angle * skew / (2.0 * math.sin(angle))


def velocity_from_error(error, gain, deadband, limit):
    magnitude = float(np.linalg.norm(error))
    if magnitude <= deadband:
        return np.zeros(3, dtype=float)
    effective = error * ((magnitude - deadband) / magnitude)
    return np.clip(gain * effective, -limit, limit)


def joint_coordinates(roll, pitch, d3, wrist):
    return np.array(
        (
            roll,
            pitch + PITCH_KIN_OFFSET,
            d3,
            wrist[0] + THETA4_KIN_OFFSET,
            wrist[1] + THETA5_KIN_OFFSET,
            wrist[2] + THETA6_KIN_OFFSET,
        ),
        dtype=float,
    )


def end_effector_pose(roll, pitch, d3, wrist):
    transform = np.asarray(
        num_forward_transform(joint_coordinates(roll, pitch, d3, wrist)),
        dtype=float,
    )
    if transform.shape != (4, 4) or not np.all(np.isfinite(transform)):
        raise RuntimeError("kinematics returned an invalid transform")
    return transform[:3, 3].copy(), transform[:3, :3].copy()


def wrist_ticks(wrist):
    return [
        ARM_HOME[ARM_IDS[0]] + radians_to_ticks(THETA4_DXL_SIGN * wrist[0]),
        ARM_HOME[ARM_IDS[1]] + radians_to_ticks(THETA5_DXL_SIGN * wrist[1]),
        ARM_HOME[ARM_IDS[2]] + radians_to_ticks(wrist[2]),
    ]


def validate_dynamixel_configuration(context):
    if tuple(ARM_IDS) != (21, 22, 23) or tuple(GRIPPER_IDS) != (24,):
        raise RuntimeError(
            "unexpected Dynamixel IDs: ARM_IDS=%r GRIPPER_IDS=%r"
            % (tuple(ARM_IDS), tuple(GRIPPER_IDS))
        )
    expected_home = {21: 3075, 22: 3075, 23: 2050}
    if (
        dict(ARM_HOME) != expected_home
        or GRIPPER_CLOSED[24] != 3900
        or GRIPPER_OPEN[24] != 6000
    ):
        raise RuntimeError("Dynamixel home/gripper calibration does not match final setup")
    state = dynamixel_read(context)
    for motor_id in tuple(ARM_IDS) + tuple(GRIPPER_IDS):
        if motor_id not in state or state[motor_id].get("position") is None:
            raise RuntimeError("no measured position from Dynamixel ID %d" % motor_id)
    for motor_id, target in expected_home.items():
        low, high = ARM_TICK_LIMITS[motor_id]
        if not low <= target <= high:
            raise RuntimeError("home for ID %d is outside calibrated limits" % motor_id)



class Inputs:
    def __init__(self):
        self.lock = threading.Lock()
        self.engaged = False
        self.gripper_open = False
        self.position = np.zeros(3, dtype=float)
        self.quaternion = np.array((0.0, 0.0, 0.0, 1.0), dtype=float)
        self.pose_receipt = 0.0
        self.md80_receipt = 0.0

    def engaged_cb(self, message):
        with self.lock:
            self.engaged = bool(message.data)

    def gripper_cb(self, message):
        with self.lock:
            self.gripper_open = bool(message.data)

    def pose_cb(self, message):
        position = np.array(
            (message.pose.position.x, message.pose.position.y, message.pose.position.z),
            dtype=float,
        )
        quaternion = np.array(
            (
                message.pose.orientation.x,
                message.pose.orientation.y,
                message.pose.orientation.z,
                message.pose.orientation.w,
            ),
            dtype=float,
        )
        try:
            quaternion = normalize_quaternion(quaternion)
        except ValueError as exc:
            rospy.logwarn_throttle(1.0, "Rejected pose command: %s", exc)
            return
        if not np.all(np.isfinite(position)):
            rospy.logwarn_throttle(1.0, "Rejected non-finite position command")
            return
        with self.lock:
            self.position = position
            self.quaternion = quaternion
            self.pose_receipt = time.monotonic()

    def md80_cb(self, message):
        names = {str(name).lower().replace(" ", "") for name in message.name}
        if {"joint21", "joint22", "joint23"}.issubset(names):
            with self.lock:
                self.md80_receipt = time.monotonic()

    def snapshot(self):
        with self.lock:
            return (
                self.engaged,
                self.gripper_open,
                self.position.copy(),
                self.quaternion.copy(),
                self.pose_receipt,
                self.md80_receipt,
            )


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dxl-port", default="/dev/ttyACM0")
    parser.add_argument("--dxl-baud", type=int, default=1_000_000)
    parser.add_argument("--kp", type=float, default=100.0)
    parser.add_argument("--kd", type=float, default=5.0)
    parser.add_argument("--max-torque", type=float, default=12.0)
    return parser.parse_args()


def main():
    args = parse_args()
    if not all(
        math.isfinite(value) and value > 0.0
        for value in (args.kp, args.kd, args.max_torque)
    ) or args.dxl_baud <= 0:
        raise ValueError("gains, torque, and baud must be positive")

    rospy.init_node("giraf_final_optitrak_hardware_receiver", anonymous=False)
    inputs = Inputs()
    rospy.Subscriber(ENGAGED_TOPIC, Bool, inputs.engaged_cb, queue_size=1, tcp_nodelay=True)
    rospy.Subscriber(
        GRIPPER_TOPIC, Bool, inputs.gripper_cb, queue_size=1, tcp_nodelay=True
    )
    rospy.Subscriber(POSE_TOPIC, PoseStamped, inputs.pose_cb, queue_size=1, tcp_nodelay=True)
    rospy.Subscriber(
        MD80_STATE_TOPIC, JointState, inputs.md80_cb, queue_size=2, tcp_nodelay=True
    )
    status_pub = rospy.Publisher(STATUS_TOPIC, String, queue_size=1, latch=True)

    minimum_d3 = float(get_boom_length_d3(BOOM_MAX))
    roll = 0.0
    pitch = 0.0
    d3 = max(D3_MIN, minimum_d3)
    boom = float(np.clip(get_boom_motor_rad(d3), BOOM_MIN, BOOM_MAX))
    d3 = float(get_boom_length_d3(boom))
    wrist = np.zeros(3, dtype=float)
    motor_context = None
    dxl_context = None
    tracking = False
    armed_by_release = False
    robot_anchor_position = None
    robot_anchor_rotation = None
    last_loop = time.monotonic()
    last_status = -math.inf

    print("FINAL FULL-6X6 HARDWARE RECEIVER")
    print("Confirm MD80 joints are physically at established home before continuing.")
    print("Start with pedal RELEASED. Physical estop must be accessible.")

    try:
        dxl_context = dynamixel_connect(port=args.dxl_port, baudrate=args.dxl_baud)
        validate_dynamixel_configuration(dxl_context)
        initial_dxl_targets = [ARM_HOME[motor_id] for motor_id in ARM_IDS] + [
            GRIPPER_CLOSED[GRIPPER_IDS[0]]
        ]
        if not dynamixel_drive(dxl_context, initial_dxl_targets):
            raise RuntimeError("failed to command wrist home and gripper closed")

        motor_context = motor_connect(
            kp=args.kp,
            kd=args.kd,
            max_torque=args.max_torque,
            gain_overrides={},
        )
        motor_drive(motor_context, roll, pitch, boom)
        rate = rospy.Rate(CONTROL_HZ)

        while not rospy.is_shutdown():
            now = time.monotonic()
            dt = min(max(now - last_loop, 0.0), 0.02)
            last_loop = now
            engaged, gripper_open, relative_position, relative_quaternion, pose_time, md80_time = (
                inputs.snapshot()
            )
            pose_fresh = pose_time > 0.0 and now - pose_time <= COMMAND_TIMEOUT_SEC
            md80_fresh = md80_time > 0.0 and now - md80_time <= MD80_TIMEOUT_SEC

            if not engaged:
                tracking = False
                armed_by_release = True
                robot_anchor_position = None
                robot_anchor_rotation = None
            elif tracking and (not pose_fresh or not md80_fresh):
                tracking = False
                armed_by_release = False
                robot_anchor_position = None
                robot_anchor_rotation = None
            elif not tracking and armed_by_release and pose_fresh and md80_fresh:
                robot_anchor_position, robot_anchor_rotation = end_effector_pose(
                    roll, pitch, d3, wrist
                )
                tracking = True
                armed_by_release = False

            linear_velocity = np.zeros(3, dtype=float)
            angular_velocity = np.zeros(3, dtype=float)
            qdot = np.zeros(6, dtype=float)
            if tracking:
                target_position = robot_anchor_position + relative_position
                target_rotation = robot_anchor_rotation @ quaternion_to_matrix(
                    relative_quaternion
                )
                current_position, current_rotation = end_effector_pose(
                    roll, pitch, d3, wrist
                )
                linear_velocity = velocity_from_error(
                    target_position - current_position,
                    POSITION_GAIN,
                    LINEAR_DEADBAND,
                    LINEAR_LIMIT,
                )
                angular_velocity = velocity_from_error(
                    rotation_vector(target_rotation @ current_rotation.T),
                    ROTATION_GAIN,
                    ANGULAR_DEADBAND,
                    ANGULAR_LIMIT,
                )
                task_velocity = np.concatenate((linear_velocity, angular_velocity))
                jacobian = np.asarray(
                    num_jacobian(joint_coordinates(roll, pitch, d3, wrist)),
                    dtype=float,
                )
                if jacobian.shape != (6, 6) or not np.all(np.isfinite(jacobian)):
                    raise RuntimeError("physical Jacobian is invalid")
                qdot = np.linalg.pinv(jacobian, rcond=JACOBIAN_RCOND).dot(task_velocity)
                if not np.all(np.isfinite(qdot)):
                    raise RuntimeError("Jacobian produced non-finite joint velocity")
                speed_ratio = float(np.max(np.abs(qdot) / MAX_JOINT_SPEED))
                if speed_ratio > 1.0:
                    qdot /= speed_ratio

                roll = float(np.clip(roll + dt*qdot[0], -ROLL_LIMIT, ROLL_LIMIT))
                pitch = float(np.clip(pitch + dt*qdot[1], PITCH_MIN, PITCH_MAX))
                proposed_d3 = max(minimum_d3, d3 + dt*float(qdot[2]))
                boom = float(
                    np.clip(get_boom_motor_rad(proposed_d3), BOOM_MIN, BOOM_MAX)
                )
                d3 = float(get_boom_length_d3(boom))
                proposed_wrist = wrist + dt*qdot[3:6]
                proposed_ticks = wrist_ticks(proposed_wrist)
                for index, (motor_id, tick) in enumerate(zip(ARM_IDS, proposed_ticks)):
                    low, high = ARM_TICK_LIMITS[motor_id]
                    if not low <= tick <= high:
                        proposed_wrist[index] = wrist[index]
                wrist = proposed_wrist

            motor_drive(motor_context, roll, pitch, boom)
            gripper_target = (
                GRIPPER_OPEN[GRIPPER_IDS[0]]
                if gripper_open
                else GRIPPER_CLOSED[GRIPPER_IDS[0]]
            )
            dxl_targets = wrist_ticks(wrist) + [gripper_target]
            if not dynamixel_drive(dxl_context, dxl_targets):
                raise RuntimeError("Dynamixel command transmission failed")

            if now - last_status >= 0.25:
                last_status = now
                status = (
                    "tracking=%s engaged=%s pose_fresh=%s md80_fresh=%s "
                    "|v|=%.4f |w|=%.4f |qdot|=%.4f"
                    % (
                        tracking,
                        engaged,
                        pose_fresh,
                        md80_fresh,
                        np.linalg.norm(linear_velocity),
                        np.linalg.norm(angular_velocity),
                        np.linalg.norm(qdot),
                    )
                )
                status_pub.publish(String(data=status))
                print("\r" + status + "   ", end="", flush=True)
            rate.sleep()
        return 0
    except Exception as exc:
        print("\nFATAL: %s" % exc)
        traceback.print_exc()
        return 1
    finally:
        if motor_context is not None:
            try:
                motor_drive(motor_context, roll, pitch, boom)
                time.sleep(0.02)
            finally:
                if not motor_disconnect():
                    print("WARNING: not every MD80 acknowledged disable")
        if dxl_context is not None:
            dynamixel_disconnect(dxl_context)
        print("\nFinal hardware receiver stopped; motor torque disabled.")


if __name__ == "__main__":
    raise SystemExit(main())
