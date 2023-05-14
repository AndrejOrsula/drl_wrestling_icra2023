import numpy as np


class SlideGaitGenerator:
    MAX_STEP_LENGTH_FRONT: float = 0.04
    MAX_STEP_LENGTH_SIDE: float = 0.02
    MIN_Z: float = -0.327

    ROBOT_HEIGHT_OFFSET: float = 0.2525
    LATERAL_LEG_OFFSET: float = 0.075
    STEP_HEIGHT: float = 0.005
    STEP_PENETRATION: float = 0.000625

    STEP_PERIOD: float = 1.0

    def __init__(self, robot, time_step):
        self.robot = robot
        self.time_step = time_step
        self.theta: float = 0.0

        self.step_length_front = self.MAX_STEP_LENGTH_FRONT
        self.step_length_side = self.MAX_STEP_LENGTH_SIDE

    def reset(self):
        self.theta = 0.0

    def update_theta(self):
        self.theta = (
            -(2 * np.pi * self.robot.getTime() / self.STEP_PERIOD) % (2 * np.pi) - np.pi
        )

    def compute_leg_position(self, is_left, desired_radius, heading_angle):
        factor = -1 if is_left else 1
        amplitude_x = (
            self.adapt_step_length(heading_angle)
            * (desired_radius - factor * self.LATERAL_LEG_OFFSET)
            / desired_radius
        )
        x = factor * amplitude_x * np.cos(self.theta)
        yaw = -x / (desired_radius - factor * self.LATERAL_LEG_OFFSET)
        y = -(1 - np.cos(yaw)) * (desired_radius - factor * self.LATERAL_LEG_OFFSET)
        if heading_angle != 0:
            x, y = self.__rotate(x, y, heading_angle)
        y += -factor * self.LATERAL_LEG_OFFSET
        amplitude_z = (
            self.STEP_PENETRATION if factor * self.theta < 0 else self.STEP_HEIGHT
        )
        z = max(
            self.MIN_Z,
            factor * amplitude_z * np.sin(self.theta) - self.ROBOT_HEIGHT_OFFSET,
        )
        return x, y, z, yaw

    def adapt_step_length(self, heading_angle):
        if heading_angle < 0:
            heading_angle = -heading_angle
        if heading_angle > np.pi / 2:
            heading_angle = np.pi - heading_angle
        factor = heading_angle / (np.pi / 2)
        amplitude = (
            self.step_length_front * (1 - factor) + self.step_length_side * factor
        )
        return amplitude

    def set_step_amplitude(self, amount):
        self.step_length_front = self.MAX_STEP_LENGTH_FRONT * amount
        self.step_length_side = self.MAX_STEP_LENGTH_SIDE * amount

    @staticmethod
    def __rotate(x, y, angle):
        return (
            x * np.cos(angle) - y * np.sin(angle),
            x * np.sin(angle) + y * np.cos(angle),
        )
