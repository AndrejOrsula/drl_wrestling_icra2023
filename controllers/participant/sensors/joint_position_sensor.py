from typing import Optional, Tuple

import constants.joint_limits as joint_limits
import numpy as np
from constants.joint_limits import *
from controller import Motor, PositionSensor, Robot


class _PositionSensor:
    def __init__(
        self,
        robot: Robot,
        time_step: float,
        device_name: Optional[str] = None,
        motor: Optional[Motor] = None,
    ):
        if device_name is not None:
            self.device: PositionSensor = robot.getDevice(device_name)
        elif motor is not None:
            self.device = motor.getPositionSensor()
        else:
            raise ValueError("Either device_name or motor must be set.")
        self.device.enable(time_step)

        self._low = getattr(joint_limits, f"{self.device.getName()[:-1]}Low")
        self._high = getattr(joint_limits, f"{self.device.getName()[:-1]}High")

    @property
    def low(self) -> float:
        return self._low

    @property
    def high(self) -> float:
        return self._high

    def get_joint_position(self) -> float:
        return self.device.getValue()

    def get_joint_position_normalized(self) -> float:
        return np.interp(
            self.get_joint_position(),
            [self.low, self.high],
            [-1, 1],
        )


class PositionHeadYaw(_PositionSensor):
    def __init__(self, robot: Robot, time_step: float, device_name: str = "HeadYawS"):
        super().__init__(robot=robot, time_step=time_step, device_name=device_name)


class PositionHeadPitch(_PositionSensor):
    def __init__(self, robot: Robot, time_step: float, device_name: str = "HeadPitchS"):
        super().__init__(robot=robot, time_step=time_step, device_name=device_name)


class PositionLShoulderPitch(_PositionSensor):
    def __init__(
        self, robot: Robot, time_step: float, device_name: str = "LShoulderPitchS"
    ):
        super().__init__(robot=robot, time_step=time_step, device_name=device_name)


class PositionLShoulderRoll(_PositionSensor):
    def __init__(
        self, robot: Robot, time_step: float, device_name: str = "LShoulderRollS"
    ):
        super().__init__(robot=robot, time_step=time_step, device_name=device_name)


class PositionLElbowYaw(_PositionSensor):
    def __init__(self, robot: Robot, time_step: float, device_name: str = "LElbowYawS"):
        super().__init__(robot=robot, time_step=time_step, device_name=device_name)


class PositionLElbowRoll(_PositionSensor):
    def __init__(
        self, robot: Robot, time_step: float, device_name: str = "LElbowRollS"
    ):
        super().__init__(robot=robot, time_step=time_step, device_name=device_name)


class PositionLWristYaw(_PositionSensor):
    def __init__(self, robot: Robot, time_step: float, device_name: str = "LWristYawS"):
        super().__init__(robot=robot, time_step=time_step, device_name=device_name)


class PositionLHipYawPitch(_PositionSensor):
    def __init__(
        self, robot: Robot, time_step: float, device_name: str = "LHipYawPitchS"
    ):
        super().__init__(robot=robot, time_step=time_step, device_name=device_name)


class PositionLHipRoll(_PositionSensor):
    def __init__(self, robot: Robot, time_step: float, device_name: str = "LHipRollS"):
        super().__init__(robot=robot, time_step=time_step, device_name=device_name)


class PositionLHipPitch(_PositionSensor):
    def __init__(self, robot: Robot, time_step: float, device_name: str = "LHipPitchS"):
        super().__init__(robot=robot, time_step=time_step, device_name=device_name)


class PositionLKneePitch(_PositionSensor):
    def __init__(
        self, robot: Robot, time_step: float, device_name: str = "LKneePitchS"
    ):
        super().__init__(robot=robot, time_step=time_step, device_name=device_name)


class PositionLAnklePitch(_PositionSensor):
    def __init__(
        self, robot: Robot, time_step: float, device_name: str = "LAnklePitchS"
    ):
        super().__init__(robot=robot, time_step=time_step, device_name=device_name)


class PositionLAnkleRoll(_PositionSensor):
    def __init__(
        self, robot: Robot, time_step: float, device_name: str = "LAnkleRollS"
    ):
        super().__init__(robot=robot, time_step=time_step, device_name=device_name)


class PositionRShoulderPitch(_PositionSensor):
    def __init__(
        self, robot: Robot, time_step: float, device_name: str = "RShoulderPitchS"
    ):
        super().__init__(robot=robot, time_step=time_step, device_name=device_name)


class PositionRShoulderRoll(_PositionSensor):
    def __init__(
        self, robot: Robot, time_step: float, device_name: str = "RShoulderRollS"
    ):
        super().__init__(robot=robot, time_step=time_step, device_name=device_name)


class PositionRElbowYaw(_PositionSensor):
    def __init__(self, robot: Robot, time_step: float, device_name: str = "RElbowYawS"):
        super().__init__(robot=robot, time_step=time_step, device_name=device_name)


class PositionRElbowRoll(_PositionSensor):
    def __init__(
        self, robot: Robot, time_step: float, device_name: str = "RElbowRollS"
    ):
        super().__init__(robot=robot, time_step=time_step, device_name=device_name)


class PositionRWristYaw(_PositionSensor):
    def __init__(self, robot: Robot, time_step: float, device_name: str = "RWristYawS"):
        super().__init__(robot=robot, time_step=time_step, device_name=device_name)


class PositionRHipYawPitch(_PositionSensor):
    def __init__(
        self, robot: Robot, time_step: float, device_name: str = "RHipYawPitchS"
    ):
        super().__init__(robot=robot, time_step=time_step, device_name=device_name)


class PositionRHipRoll(_PositionSensor):
    def __init__(self, robot: Robot, time_step: float, device_name: str = "RHipRollS"):
        super().__init__(robot=robot, time_step=time_step, device_name=device_name)


class PositionRHipPitch(_PositionSensor):
    def __init__(self, robot: Robot, time_step: float, device_name: str = "RHipPitchS"):
        super().__init__(robot=robot, time_step=time_step, device_name=device_name)


class PositionRKneePitch(_PositionSensor):
    def __init__(
        self, robot: Robot, time_step: float, device_name: str = "RKneePitchS"
    ):
        super().__init__(robot=robot, time_step=time_step, device_name=device_name)


class PositionRAnklePitch(_PositionSensor):
    def __init__(
        self, robot: Robot, time_step: float, device_name: str = "RAnklePitchS"
    ):
        super().__init__(robot=robot, time_step=time_step, device_name=device_name)


class PositionRAnkleRoll(_PositionSensor):
    def __init__(
        self, robot: Robot, time_step: float, device_name: str = "RAnkleRollS"
    ):
        super().__init__(robot=robot, time_step=time_step, device_name=device_name)


class PositionLPhalanx1(_PositionSensor):
    def __init__(self, robot: Robot, time_step: float, device_name: str = "LPhalanx1S"):
        super().__init__(robot=robot, time_step=time_step, device_name=device_name)


class PositionLPhalanx2(_PositionSensor):
    def __init__(self, robot: Robot, time_step: float, device_name: str = "LPhalanx2S"):
        super().__init__(robot=robot, time_step=time_step, device_name=device_name)


class PositionLPhalanx3(_PositionSensor):
    def __init__(self, robot: Robot, time_step: float, device_name: str = "LPhalanx3S"):
        super().__init__(robot=robot, time_step=time_step, device_name=device_name)


class PositionLPhalanx4(_PositionSensor):
    def __init__(self, robot: Robot, time_step: float, device_name: str = "LPhalanx4S"):
        super().__init__(robot=robot, time_step=time_step, device_name=device_name)


class PositionLPhalanx5(_PositionSensor):
    def __init__(self, robot: Robot, time_step: float, device_name: str = "LPhalanx5S"):
        super().__init__(robot=robot, time_step=time_step, device_name=device_name)


class PositionLPhalanx6(_PositionSensor):
    def __init__(self, robot: Robot, time_step: float, device_name: str = "LPhalanx6S"):
        super().__init__(robot=robot, time_step=time_step, device_name=device_name)


class PositionLPhalanx7(_PositionSensor):
    def __init__(self, robot: Robot, time_step: float, device_name: str = "LPhalanx7S"):
        super().__init__(robot=robot, time_step=time_step, device_name=device_name)


class PositionLPhalanx8(_PositionSensor):
    def __init__(self, robot: Robot, time_step: float, device_name: str = "LPhalanx8S"):
        super().__init__(robot=robot, time_step=time_step, device_name=device_name)


class PositionRPhalanx1(_PositionSensor):
    def __init__(self, robot: Robot, time_step: float, device_name: str = "RPhalanx1S"):
        super().__init__(robot=robot, time_step=time_step, device_name=device_name)


class PositionRPhalanx2(_PositionSensor):
    def __init__(self, robot: Robot, time_step: float, device_name: str = "RPhalanx2S"):
        super().__init__(robot=robot, time_step=time_step, device_name=device_name)


class PositionRPhalanx3(_PositionSensor):
    def __init__(self, robot: Robot, time_step: float, device_name: str = "RPhalanx3S"):
        super().__init__(robot=robot, time_step=time_step, device_name=device_name)


class PositionRPhalanx4(_PositionSensor):
    def __init__(self, robot: Robot, time_step: float, device_name: str = "RPhalanx4S"):
        super().__init__(robot=robot, time_step=time_step, device_name=device_name)


class PositionRPhalanx5(_PositionSensor):
    def __init__(self, robot: Robot, time_step: float, device_name: str = "RPhalanx5S"):
        super().__init__(robot=robot, time_step=time_step, device_name=device_name)


class PositionRPhalanx6(_PositionSensor):
    def __init__(self, robot: Robot, time_step: float, device_name: str = "RPhalanx6S"):
        super().__init__(robot=robot, time_step=time_step, device_name=device_name)


class PositionRPhalanx7(_PositionSensor):
    def __init__(self, robot: Robot, time_step: float, device_name: str = "RPhalanx7S"):
        super().__init__(robot=robot, time_step=time_step, device_name=device_name)


class PositionRPhalanx8(_PositionSensor):
    def __init__(self, robot: Robot, time_step: float, device_name: str = "RPhalanx8S"):
        super().__init__(robot=robot, time_step=time_step, device_name=device_name)


class JointPositionSensorsCombined:
    def __init__(
        self,
        robot: Robot,
        time_step: float,
        head_yaw: bool = False,
        head_pitch: bool = False,
        simplified_scheme_legs: bool = False,
        simplified_scheme_arms: bool = False,
        include_fingers: bool = False,
    ):
        self.__position_sensors = [
            PositionLShoulderPitch(robot=robot, time_step=time_step),
            PositionLShoulderRoll(robot=robot, time_step=time_step),
            PositionLElbowRoll(robot=robot, time_step=time_step),
            PositionLHipRoll(robot=robot, time_step=time_step),
            PositionLHipPitch(robot=robot, time_step=time_step),
            PositionLKneePitch(robot=robot, time_step=time_step),
            PositionLAnklePitch(robot=robot, time_step=time_step),
            PositionRShoulderPitch(robot=robot, time_step=time_step),
            PositionRShoulderRoll(robot=robot, time_step=time_step),
            PositionRElbowRoll(robot=robot, time_step=time_step),
            PositionRHipRoll(robot=robot, time_step=time_step),
            PositionRHipPitch(robot=robot, time_step=time_step),
            PositionRKneePitch(robot=robot, time_step=time_step),
            PositionRAnklePitch(robot=robot, time_step=time_step),
        ]
        if head_yaw:
            self.__position_sensors.append(
                PositionHeadYaw(robot=robot, time_step=time_step)
            )
        if head_pitch:
            self.__position_sensors.append(
                PositionHeadPitch(robot=robot, time_step=time_step)
            )

        if not simplified_scheme_legs:
            self.__position_sensors.extend(
                [
                    PositionLHipYawPitch(robot=robot, time_step=time_step),
                    PositionLAnkleRoll(robot=robot, time_step=time_step),
                    PositionRHipYawPitch(robot=robot, time_step=time_step),
                    PositionRAnkleRoll(robot=robot, time_step=time_step),
                ]
            )
        if not simplified_scheme_arms:
            self.__position_sensors.extend(
                [
                    PositionLElbowYaw(robot=robot, time_step=time_step),
                    PositionLWristYaw(robot=robot, time_step=time_step),
                    PositionRElbowYaw(robot=robot, time_step=time_step),
                    PositionRWristYaw(robot=robot, time_step=time_step),
                ]
            )
        if include_fingers:
            self.__position_sensors.extend(
                [
                    PositionLPhalanx1(robot=robot, time_step=time_step),
                    PositionLPhalanx2(robot=robot, time_step=time_step),
                    PositionLPhalanx3(robot=robot, time_step=time_step),
                    PositionLPhalanx4(robot=robot, time_step=time_step),
                    PositionLPhalanx5(robot=robot, time_step=time_step),
                    PositionLPhalanx6(robot=robot, time_step=time_step),
                    PositionLPhalanx7(robot=robot, time_step=time_step),
                    PositionLPhalanx8(robot=robot, time_step=time_step),
                    PositionRPhalanx1(robot=robot, time_step=time_step),
                    PositionRPhalanx2(robot=robot, time_step=time_step),
                    PositionRPhalanx3(robot=robot, time_step=time_step),
                    PositionRPhalanx4(robot=robot, time_step=time_step),
                    PositionRPhalanx5(robot=robot, time_step=time_step),
                    PositionRPhalanx6(robot=robot, time_step=time_step),
                    PositionRPhalanx7(robot=robot, time_step=time_step),
                    PositionRPhalanx8(robot=robot, time_step=time_step),
                ]
            )
        self.__position_sensors = tuple(self.__position_sensors)

    @property
    def position_sensors(self) -> Tuple[_PositionSensor, ...]:
        return self.__position_sensors

    @property
    def output_size(self) -> int:
        return self.n_joints

    @property
    def n_joints(self) -> int:
        return len(self.__position_sensors)

    def get_joint_position(self) -> np.ndarray:
        return np.array(
            [
                position_sensor.get_joint_position()
                for position_sensor in self.__position_sensors
            ],
            dtype=np.float32,
        )

    def get_joint_position_normalized(self) -> np.ndarray:
        return np.array(
            [
                position_sensor.get_joint_position_normalized()
                for position_sensor in self.__position_sensors
            ],
            dtype=np.float32,
        )
