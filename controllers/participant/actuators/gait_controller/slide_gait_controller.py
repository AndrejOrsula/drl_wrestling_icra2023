# Copyright 1996-2023 Cyberbotics Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import constants.joint_limits as joint_limits

from .kinematics import Kinematics
from .slide_gait_generator import SlideGaitGenerator


class SlideGaitController:
    """Connects the Kinematics class and the SlideGaitGenerator class together to have a simple gait interface."""

    def __init__(self, robot, time_step):
        self.gait_generator = SlideGaitGenerator(robot, time_step)
        self.kinematics = Kinematics()
        joints = [
            "HipYawPitch",
            "HipRoll",
            "HipPitch",
            "KneePitch",
            "AnklePitch",
            "AnkleRoll",
        ]
        self.L_leg_motors = [robot.getDevice(f"L{joint}") for joint in joints]
        self.R_leg_motors = [robot.getDevice(f"R{joint}") for joint in joints]
        self.L_limits = [
            (
                getattr(joint_limits, f"L{joint}Low"),
                getattr(joint_limits, f"L{joint}High"),
            )
            for joint in joints
        ]
        self.R_limits = [
            (
                getattr(joint_limits, f"R{joint}Low"),
                getattr(joint_limits, f"R{joint}High"),
            )
            for joint in joints
        ]

    def reset(self):
        self.gait_generator.reset()
        self.kinematics.reset()

    def update_theta(self):
        self.gait_generator.update_theta()

    def set_step_amplitude(self, amplitude: float):
        self.gait_generator.set_step_amplitude(amplitude)

    def command_to_motors(self, desired_radius=0, heading_angle=0):
        """
        Compute the desired positions of the robot's legs for a desired radius (R > 0 is a right turn)
        and a desired heading angle (in radians. 0 is straight on, > 0 is turning left).
        Send the commands to the motors.
        """
        self.update_theta()
        x, y, z, yaw = self.gait_generator.compute_leg_position(
            is_left=False, desired_radius=desired_radius, heading_angle=heading_angle
        )
        right_target_commands = self.kinematics.inverse_leg(
            x * 1e3, y * 1e3, z * 1e3, 0, 0, yaw, is_left=False
        )
        for command, motor, limits in zip(
            right_target_commands, self.R_leg_motors, self.R_limits
        ):
            motor.setPosition(min(max(command, limits[0]), limits[1]))

        x, y, z, yaw = self.gait_generator.compute_leg_position(
            is_left=True, desired_radius=desired_radius, heading_angle=heading_angle
        )
        left_target_commands = self.kinematics.inverse_leg(
            x * 1e3, y * 1e3, z * 1e3, 0, 0, yaw, is_left=True
        )
        for command, motor, limits in zip(
            left_target_commands, self.L_leg_motors, self.L_limits
        ):
            motor.setPosition(min(max(command, limits[0]), limits[1]))
