import constants.joint_limits as joint_limits
import numpy as np
from constants.joint_limits import *
from controller import Motor, Robot
from sensors.joint_position_sensor import _PositionSensor


class __JointController:
    def __init__(self, robot: Robot, time_step: float, device_name: str):
        self.device: Motor = robot.getDevice(device_name)
        self.position_sensor = _PositionSensor(
            robot=robot, time_step=time_step, motor=self.device
        )

    @property
    def low(self) -> float:
        return getattr(joint_limits, f"{self.device.getName()[:-1]}Low")

    @property
    def high(self) -> float:
        return getattr(joint_limits, f"{self.device.getName()[:-1]}High")

    def set_joint_position(self, position: float):
        self.device.setPosition(position)

    def set_joint_position_normalized(self, position: float):
        self.set_joint_position(
            np.interp(
                position,
                [-1, 1],
                [self.low, self.high],
            )
        )

    def set_joint_position_relative(self, offset: float):
        self.device.setPosition(self.position_sensor.get_joint_position() + offset)

    def set_joint_position_relative_normalized(self, offset: float):
        self.set_joint_position_normalized(
            np.clip(
                self.position_sensor.get_joint_position_normalized() + offset,
                -1,
                1,
            )
        )

    def reassign_position_sensor(self, position_sensor: _PositionSensor):
        self.position_sensor = position_sensor


class ControllerHeadYaw(__JointController):
    def __init__(self, robot: Robot, time_step: float, device_name: str = "HeadYaw"):
        super().__init__(robot=robot, time_step=time_step, device_name=device_name)

    @property
    def low(self) -> float:
        return HeadYawLow

    @property
    def high(self) -> float:
        return HeadYawHigh


class ControllerHeadPitch(__JointController):
    def __init__(self, robot: Robot, time_step: float, device_name: str = "HeadPitch"):
        super().__init__(robot=robot, time_step=time_step, device_name=device_name)

    @property
    def low(self) -> float:
        return HeadPitchLow

    @property
    def high(self) -> float:
        return HeadPitchHigh


class ControllerLShoulderPitch(__JointController):
    def __init__(
        self, robot: Robot, time_step: float, device_name: str = "LShoulderPitch"
    ):
        super().__init__(robot=robot, time_step=time_step, device_name=device_name)

    @property
    def low(self) -> float:
        return LShoulderPitchLow

    @property
    def high(self) -> float:
        return LShoulderPitchHigh


class ControllerLShoulderRoll(__JointController):
    def __init__(
        self, robot: Robot, time_step: float, device_name: str = "LShoulderRoll"
    ):
        super().__init__(robot=robot, time_step=time_step, device_name=device_name)

    @property
    def low(self) -> float:
        return LShoulderRollLow

    @property
    def high(self) -> float:
        return LShoulderRollHigh


class ControllerLElbowYaw(__JointController):
    def __init__(self, robot: Robot, time_step: float, device_name: str = "LElbowYaw"):
        super().__init__(robot=robot, time_step=time_step, device_name=device_name)

    @property
    def low(self) -> float:
        return LElbowYawLow

    @property
    def high(self) -> float:
        return LElbowYawHigh


class ControllerLElbowRoll(__JointController):
    def __init__(self, robot: Robot, time_step: float, device_name: str = "LElbowRoll"):
        super().__init__(robot=robot, time_step=time_step, device_name=device_name)

    @property
    def low(self) -> float:
        return LElbowRollLow

    @property
    def high(self) -> float:
        return LElbowRollHigh


class ControllerLWristYaw(__JointController):
    def __init__(self, robot: Robot, time_step: float, device_name: str = "LWristYaw"):
        super().__init__(robot=robot, time_step=time_step, device_name=device_name)

    @property
    def low(self) -> float:
        return LWristYawLow

    @property
    def high(self) -> float:
        return LWristYawHigh


class ControllerLHipYawPitch(__JointController):
    def __init__(
        self, robot: Robot, time_step: float, device_name: str = "LHipYawPitch"
    ):
        super().__init__(robot=robot, time_step=time_step, device_name=device_name)

    @property
    def low(self) -> float:
        return LHipYawPitchLow

    @property
    def high(self) -> float:
        return LHipYawPitchHigh


class ControllerLHipRoll(__JointController):
    def __init__(self, robot: Robot, time_step: float, device_name: str = "LHipRoll"):
        super().__init__(robot=robot, time_step=time_step, device_name=device_name)

    @property
    def low(self) -> float:
        return LHipRollLow

    @property
    def high(self) -> float:
        return LHipRollHigh


class ControllerLHipPitch(__JointController):
    def __init__(self, robot: Robot, time_step: float, device_name: str = "LHipPitch"):
        super().__init__(robot=robot, time_step=time_step, device_name=device_name)

    @property
    def low(self) -> float:
        return LHipPitchLow

    @property
    def high(self) -> float:
        return LHipPitchHigh


class ControllerLKneePitch(__JointController):
    def __init__(self, robot: Robot, time_step: float, device_name: str = "LKneePitch"):
        super().__init__(robot=robot, time_step=time_step, device_name=device_name)

    @property
    def low(self) -> float:
        return LKneePitchLow

    @property
    def high(self) -> float:
        return LKneePitchHigh


class ControllerLAnklePitch(__JointController):
    def __init__(
        self, robot: Robot, time_step: float, device_name: str = "LAnklePitch"
    ):
        super().__init__(robot=robot, time_step=time_step, device_name=device_name)

    @property
    def low(self) -> float:
        return LAnklePitchLow

    @property
    def high(self) -> float:
        return LAnklePitchHigh


class ControllerLAnkleRoll(__JointController):
    def __init__(self, robot: Robot, time_step: float, device_name: str = "LAnkleRoll"):
        super().__init__(robot=robot, time_step=time_step, device_name=device_name)

    @property
    def low(self) -> float:
        return LAnkleRollLow

    @property
    def high(self) -> float:
        return LAnkleRollHigh


class ControllerRShoulderPitch(__JointController):
    def __init__(
        self, robot: Robot, time_step: float, device_name: str = "RShoulderPitch"
    ):
        super().__init__(robot=robot, time_step=time_step, device_name=device_name)

    @property
    def low(self) -> float:
        return RShoulderPitchLow

    @property
    def high(self) -> float:
        return RShoulderPitchHigh


class ControllerRShoulderRoll(__JointController):
    def __init__(
        self, robot: Robot, time_step: float, device_name: str = "RShoulderRoll"
    ):
        super().__init__(robot=robot, time_step=time_step, device_name=device_name)

    @property
    def low(self) -> float:
        return RShoulderRollLow

    @property
    def high(self) -> float:
        return RShoulderRollHigh


class ControllerRElbowYaw(__JointController):
    def __init__(self, robot: Robot, time_step: float, device_name: str = "RElbowYaw"):
        super().__init__(robot=robot, time_step=time_step, device_name=device_name)

    @property
    def low(self) -> float:
        return RElbowYawLow

    @property
    def high(self) -> float:
        return RElbowYawHigh


class ControllerRElbowRoll(__JointController):
    def __init__(self, robot: Robot, time_step: float, device_name: str = "RElbowRoll"):
        super().__init__(robot=robot, time_step=time_step, device_name=device_name)

    @property
    def low(self) -> float:
        return RElbowRollLow

    @property
    def high(self) -> float:
        return RElbowRollHigh


class ControllerRWristYaw(__JointController):
    def __init__(self, robot: Robot, time_step: float, device_name: str = "RWristYaw"):
        super().__init__(robot=robot, time_step=time_step, device_name=device_name)

    @property
    def low(self) -> float:
        return RWristYawLow

    @property
    def high(self) -> float:
        return RWristYawHigh


class ControllerRHipYawPitch(__JointController):
    def __init__(
        self, robot: Robot, time_step: float, device_name: str = "RHipYawPitch"
    ):
        super().__init__(robot=robot, time_step=time_step, device_name=device_name)

    @property
    def low(self) -> float:
        return RHipYawPitchLow

    @property
    def high(self) -> float:
        return RHipYawPitchHigh


class ControllerRHipRoll(__JointController):
    def __init__(self, robot: Robot, time_step: float, device_name: str = "RHipRoll"):
        super().__init__(robot=robot, time_step=time_step, device_name=device_name)

    @property
    def low(self) -> float:
        return RHipRollLow

    @property
    def high(self) -> float:
        return RHipRollHigh


class ControllerRHipPitch(__JointController):
    def __init__(self, robot: Robot, time_step: float, device_name: str = "RHipPitch"):
        super().__init__(robot=robot, time_step=time_step, device_name=device_name)

    @property
    def low(self) -> float:
        return RHipPitchLow

    @property
    def high(self) -> float:
        return RHipPitchHigh


class ControllerRKneePitch(__JointController):
    def __init__(self, robot: Robot, time_step: float, device_name: str = "RKneePitch"):
        super().__init__(robot=robot, time_step=time_step, device_name=device_name)

    @property
    def low(self) -> float:
        return RKneePitchLow

    @property
    def high(self) -> float:
        return RKneePitchHigh


class ControllerRAnklePitch(__JointController):
    def __init__(
        self, robot: Robot, time_step: float, device_name: str = "RAnklePitch"
    ):
        super().__init__(robot=robot, time_step=time_step, device_name=device_name)

    @property
    def low(self) -> float:
        return RAnklePitchLow

    @property
    def high(self) -> float:
        return RAnklePitchHigh


class ControllerRAnkleRoll(__JointController):
    def __init__(self, robot: Robot, time_step: float, device_name: str = "RAnkleRoll"):
        super().__init__(robot=robot, time_step=time_step, device_name=device_name)

    @property
    def low(self) -> float:
        return RAnkleRollLow

    @property
    def high(self) -> float:
        return RAnkleRollHigh


class ControllerLPhalanx1(__JointController):
    def __init__(self, robot: Robot, time_step: float, device_name: str = "LPhalanx1"):
        super().__init__(robot=robot, time_step=time_step, device_name=device_name)

    @property
    def low(self) -> float:
        return LPhalanx1Low

    @property
    def high(self) -> float:
        return LPhalanx1High


class ControllerLPhalanx2(__JointController):
    def __init__(self, robot: Robot, time_step: float, device_name: str = "LPhalanx2"):
        super().__init__(robot=robot, time_step=time_step, device_name=device_name)

    @property
    def low(self) -> float:
        return LPhalanx2Low

    @property
    def high(self) -> float:
        return LPhalanx2High


class ControllerLPhalanx3(__JointController):
    def __init__(self, robot: Robot, time_step: float, device_name: str = "LPhalanx3"):
        super().__init__(robot=robot, time_step=time_step, device_name=device_name)

    @property
    def low(self) -> float:
        return LPhalanx3Low

    @property
    def high(self) -> float:
        return LPhalanx3High


class ControllerLPhalanx4(__JointController):
    def __init__(self, robot: Robot, time_step: float, device_name: str = "LPhalanx4"):
        super().__init__(robot=robot, time_step=time_step, device_name=device_name)

    @property
    def low(self) -> float:
        return LPhalanx4Low

    @property
    def high(self) -> float:
        return LPhalanx4High


class ControllerLPhalanx5(__JointController):
    def __init__(self, robot: Robot, time_step: float, device_name: str = "LPhalanx5"):
        super().__init__(robot=robot, time_step=time_step, device_name=device_name)

    @property
    def low(self) -> float:
        return LPhalanx5Low

    @property
    def high(self) -> float:
        return LPhalanx5High


class ControllerLPhalanx6(__JointController):
    def __init__(self, robot: Robot, time_step: float, device_name: str = "LPhalanx6"):
        super().__init__(robot=robot, time_step=time_step, device_name=device_name)

    @property
    def low(self) -> float:
        return LPhalanx6Low

    @property
    def high(self) -> float:
        return LPhalanx6High


class ControllerLPhalanx7(__JointController):
    def __init__(self, robot: Robot, time_step: float, device_name: str = "LPhalanx7"):
        super().__init__(robot=robot, time_step=time_step, device_name=device_name)

    @property
    def low(self) -> float:
        return LPhalanx7Low

    @property
    def high(self) -> float:
        return LPhalanx7High


class ControllerLPhalanx8(__JointController):
    def __init__(self, robot: Robot, time_step: float, device_name: str = "LPhalanx8"):
        super().__init__(robot=robot, time_step=time_step, device_name=device_name)

    @property
    def low(self) -> float:
        return LPhalanx8Low

    @property
    def high(self) -> float:
        return LPhalanx8High


class ControllerRPhalanx1(__JointController):
    def __init__(self, robot: Robot, time_step: float, device_name: str = "RPhalanx1"):
        super().__init__(robot=robot, time_step=time_step, device_name=device_name)

    @property
    def low(self) -> float:
        return RPhalanx1Low

    @property
    def high(self) -> float:
        return RPhalanx1High


class ControllerRPhalanx2(__JointController):
    def __init__(self, robot: Robot, time_step: float, device_name: str = "RPhalanx2"):
        super().__init__(robot=robot, time_step=time_step, device_name=device_name)

    @property
    def low(self) -> float:
        return RPhalanx2Low

    @property
    def high(self) -> float:
        return RPhalanx2High


class ControllerRPhalanx3(__JointController):
    def __init__(self, robot: Robot, time_step: float, device_name: str = "RPhalanx3"):
        super().__init__(robot=robot, time_step=time_step, device_name=device_name)

    @property
    def low(self) -> float:
        return RPhalanx3Low

    @property
    def high(self) -> float:
        return RPhalanx3High


class ControllerRPhalanx4(__JointController):
    def __init__(self, robot: Robot, time_step: float, device_name: str = "RPhalanx4"):
        super().__init__(robot=robot, time_step=time_step, device_name=device_name)

    @property
    def low(self) -> float:
        return RPhalanx4Low

    @property
    def high(self) -> float:
        return RPhalanx4High


class ControllerRPhalanx5(__JointController):
    def __init__(self, robot: Robot, time_step: float, device_name: str = "RPhalanx5"):
        super().__init__(robot=robot, time_step=time_step, device_name=device_name)

    @property
    def low(self) -> float:
        return RPhalanx5Low

    @property
    def high(self) -> float:
        return RPhalanx5High


class ControllerRPhalanx6(__JointController):
    def __init__(self, robot: Robot, time_step: float, device_name: str = "RPhalanx6"):
        super().__init__(robot=robot, time_step=time_step, device_name=device_name)

    @property
    def low(self) -> float:
        return RPhalanx6Low

    @property
    def high(self) -> float:
        return RPhalanx6High


class ControllerRPhalanx7(__JointController):
    def __init__(self, robot: Robot, time_step: float, device_name: str = "RPhalanx7"):
        super().__init__(robot=robot, time_step=time_step, device_name=device_name)

    @property
    def low(self) -> float:
        return RPhalanx7Low

    @property
    def high(self) -> float:
        return RPhalanx7High


class ControllerRPhalanx8(__JointController):
    def __init__(self, robot: Robot, time_step: float, device_name: str = "RPhalanx8"):
        super().__init__(robot=robot, time_step=time_step, device_name=device_name)

    @property
    def low(self) -> float:
        return RPhalanx8Low

    @property
    def high(self) -> float:
        return RPhalanx8High


class JointControllersCombined:
    def __init__(
        self,
        robot: Robot,
        time_step: float,
        simplified_scheme: bool = False,
        include_fingers: bool = False,
    ):
        self._joint_controllers = [
            ControllerLShoulderPitch(robot=robot, time_step=time_step),
            ControllerLShoulderRoll(robot=robot, time_step=time_step),
            ControllerLElbowRoll(robot=robot, time_step=time_step),
            ControllerLHipRoll(robot=robot, time_step=time_step),
            ControllerLHipPitch(robot=robot, time_step=time_step),
            ControllerLKneePitch(robot=robot, time_step=time_step),
            ControllerLAnklePitch(robot=robot, time_step=time_step),
            ControllerRShoulderPitch(robot=robot, time_step=time_step),
            ControllerRShoulderRoll(robot=robot, time_step=time_step),
            ControllerRElbowRoll(robot=robot, time_step=time_step),
            ControllerRHipRoll(robot=robot, time_step=time_step),
            ControllerRHipPitch(robot=robot, time_step=time_step),
            ControllerRKneePitch(robot=robot, time_step=time_step),
            ControllerRAnklePitch(robot=robot, time_step=time_step),
        ]
        if not simplified_scheme:
            self._joint_controllers.extend(
                [
                    ControllerHeadYaw(robot=robot, time_step=time_step),
                    ControllerHeadPitch(robot=robot, time_step=time_step),
                    ControllerLElbowYaw(robot=robot, time_step=time_step),
                    ControllerLWristYaw(robot=robot, time_step=time_step),
                    ControllerLHipYawPitch(robot=robot, time_step=time_step),
                    ControllerLAnkleRoll(robot=robot, time_step=time_step),
                    ControllerRElbowYaw(robot=robot, time_step=time_step),
                    ControllerRWristYaw(robot=robot, time_step=time_step),
                    ControllerRHipYawPitch(robot=robot, time_step=time_step),
                    ControllerRAnkleRoll(robot=robot, time_step=time_step),
                ]
            )
        if include_fingers:
            self._joint_controllers.extend(
                [
                    ControllerLPhalanx1(robot=robot, time_step=time_step),
                    ControllerLPhalanx2(robot=robot, time_step=time_step),
                    ControllerLPhalanx3(robot=robot, time_step=time_step),
                    ControllerLPhalanx4(robot=robot, time_step=time_step),
                    ControllerLPhalanx5(robot=robot, time_step=time_step),
                    ControllerLPhalanx6(robot=robot, time_step=time_step),
                    ControllerLPhalanx7(robot=robot, time_step=time_step),
                    ControllerLPhalanx8(robot=robot, time_step=time_step),
                    ControllerRPhalanx1(robot=robot, time_step=time_step),
                    ControllerRPhalanx2(robot=robot, time_step=time_step),
                    ControllerRPhalanx3(robot=robot, time_step=time_step),
                    ControllerRPhalanx4(robot=robot, time_step=time_step),
                    ControllerRPhalanx5(robot=robot, time_step=time_step),
                    ControllerRPhalanx6(robot=robot, time_step=time_step),
                    ControllerRPhalanx7(robot=robot, time_step=time_step),
                    ControllerRPhalanx8(robot=robot, time_step=time_step),
                ]
            )

        self._joint_controllers = tuple(self._joint_controllers)

    @property
    def input_size(self) -> int:
        return self.n_joints

    @property
    def n_joints(self) -> int:
        return len(self._joint_controllers)

    def set_joint_position(self, positions: np.ndarray):
        for i, controller in enumerate(self._joint_controllers):
            controller.set_joint_position(positions[i])

    def set_joint_position_normalized(self, positions: np.ndarray):
        for i, controller in enumerate(self._joint_controllers):
            controller.set_joint_position_normalized(positions[i])

    def set_joint_position_relative(
        self,
        offsets: np.ndarray,
        relative_scaling_factor: float = 0.05,
    ):
        offsets *= relative_scaling_factor

        for i, controller in enumerate(self._joint_controllers):
            controller.set_joint_position_relative(offsets[i])

    def set_joint_position_relative_normalized(
        self,
        offsets: np.ndarray,
        relative_scaling_factor: float = 0.1,
    ):
        offsets *= relative_scaling_factor

        for i, controller in enumerate(self._joint_controllers):
            controller.set_joint_position_relative_normalized(offsets[i])


class HeadController(JointControllersCombined):
    def __init__(
        self,
        robot: Robot,
        time_step: float,
        yaw: bool = True,
        pitch: bool = True,
    ):
        self._joint_controllers = []
        if yaw:
            self._joint_controllers.extend(
                [
                    ControllerHeadYaw(robot=robot, time_step=time_step),
                ]
            )
        if pitch:
            self._joint_controllers.extend(
                [
                    ControllerHeadPitch(robot=robot, time_step=time_step),
                ]
            )
        self._joint_controllers = tuple(self._joint_controllers)


class ArmControllers(JointControllersCombined):
    def __init__(
        self,
        robot: Robot,
        time_step: float,
        simplified_scheme: bool = False,
    ):
        self._joint_controllers = [
            ControllerLShoulderPitch(robot=robot, time_step=time_step),
            ControllerLShoulderRoll(robot=robot, time_step=time_step),
            ControllerLElbowRoll(robot=robot, time_step=time_step),
            ControllerRShoulderPitch(robot=robot, time_step=time_step),
            ControllerRShoulderRoll(robot=robot, time_step=time_step),
            ControllerRElbowRoll(robot=robot, time_step=time_step),
        ]
        if not simplified_scheme:
            self._joint_controllers.extend(
                [
                    ControllerLElbowYaw(robot=robot, time_step=time_step),
                    ControllerLWristYaw(robot=robot, time_step=time_step),
                    ControllerRElbowYaw(robot=robot, time_step=time_step),
                    ControllerRWristYaw(robot=robot, time_step=time_step),
                ]
            )
        self._joint_controllers = tuple(self._joint_controllers)


class LegControllers(JointControllersCombined):
    def __init__(
        self,
        robot: Robot,
        time_step: float,
        simplified_scheme: bool = False,
    ):
        self._joint_controllers = [
            ControllerLHipRoll(robot=robot, time_step=time_step),
            ControllerLHipPitch(robot=robot, time_step=time_step),
            ControllerLKneePitch(robot=robot, time_step=time_step),
            ControllerLAnklePitch(robot=robot, time_step=time_step),
            ControllerRHipRoll(robot=robot, time_step=time_step),
            ControllerRHipPitch(robot=robot, time_step=time_step),
            ControllerRKneePitch(robot=robot, time_step=time_step),
            ControllerRAnklePitch(robot=robot, time_step=time_step),
        ]
        if not simplified_scheme:
            self._joint_controllers.extend(
                [
                    ControllerLHipYawPitch(robot=robot, time_step=time_step),
                    ControllerLAnkleRoll(robot=robot, time_step=time_step),
                    ControllerRHipYawPitch(robot=robot, time_step=time_step),
                    ControllerRAnkleRoll(robot=robot, time_step=time_step),
                ]
            )
        self._joint_controllers = tuple(self._joint_controllers)


class HandControllers(JointControllersCombined):
    def __init__(
        self,
        robot: Robot,
        time_step: float,
        enable: bool = True,
    ):
        self.enable = enable
        if self.enable:
            self._joint_controllers_left = (
                ControllerLPhalanx1(robot=robot, time_step=time_step),
                ControllerLPhalanx2(robot=robot, time_step=time_step),
                ControllerLPhalanx3(robot=robot, time_step=time_step),
                ControllerLPhalanx4(robot=robot, time_step=time_step),
                ControllerLPhalanx5(robot=robot, time_step=time_step),
                ControllerLPhalanx6(robot=robot, time_step=time_step),
                ControllerLPhalanx7(robot=robot, time_step=time_step),
                ControllerLPhalanx8(robot=robot, time_step=time_step),
            )
            self._joint_controllers_right = (
                ControllerRPhalanx1(robot=robot, time_step=time_step),
                ControllerRPhalanx2(robot=robot, time_step=time_step),
                ControllerRPhalanx3(robot=robot, time_step=time_step),
                ControllerRPhalanx4(robot=robot, time_step=time_step),
                ControllerRPhalanx5(robot=robot, time_step=time_step),
                ControllerRPhalanx6(robot=robot, time_step=time_step),
                ControllerRPhalanx7(robot=robot, time_step=time_step),
                ControllerRPhalanx8(robot=robot, time_step=time_step),
            )
            self._joint_controllers = tuple(
                self._joint_controllers_left + self._joint_controllers_right
            )
        else:
            self._joint_controllers_left = ()
            self._joint_controllers_right = ()
            self._joint_controllers = ()

    @property
    def input_size_combined(self) -> int:
        return 2 if self.enable else 0

    @property
    def n_joints_left(self) -> int:
        return len(self._joint_controllers_left)

    @property
    def n_joints_right(self) -> int:
        return len(self._joint_controllers_right)

    def set_joint_positions_left(self, positions: np.ndarray):
        for i, controller in enumerate(self._joint_controllers_left):
            controller.set_joint_position(positions[i])

    def set_joint_positions_right(self, positions: np.ndarray):
        for i, controller in enumerate(self._joint_controllers_right):
            controller.set_joint_position(positions[i])

    def set_joint_position_left(self, position: float):
        for controller in self._joint_controllers_left:
            controller.set_joint_position(position)

    def set_joint_position_right(self, position: float):
        for controller in self._joint_controllers_right:
            controller.set_joint_position(position)

    def set_joint_positions_left_normalized(self, positions: np.ndarray):
        for i, controller in enumerate(self._joint_controllers_left):
            controller.set_joint_position_normalized(positions[i])

    def set_joint_positions_right_normalized(self, positions: np.ndarray):
        for i, controller in enumerate(self._joint_controllers_right):
            controller.set_joint_position_normalized(positions[i])

    def set_joint_position_left_normalized(self, position: float):
        for controller in self._joint_controllers_left:
            controller.set_joint_position_normalized(position)

    def set_joint_position_right_normalized(self, position: float):
        for controller in self._joint_controllers_right:
            controller.set_joint_position_normalized(position)
