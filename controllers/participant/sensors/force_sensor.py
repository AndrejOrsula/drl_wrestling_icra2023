import numpy as np
from controller import Robot, TouchSensor


class __ForceSensor:
    def __init__(self, robot: Robot, time_step: float, device_name: str):
        self.device: TouchSensor = robot.getDevice(device_name)
        self.device.enable(time_step)

    def get_force(self) -> np.ndarray:
        return np.array(self.device.getValues()[:3], dtype=np.float32)


class ForceLeftFoot(__ForceSensor):
    def __init__(self, robot: Robot, time_step: float, device_name: str = "LFsr"):
        super().__init__(robot=robot, time_step=time_step, device_name=device_name)


class ForceRightFoot(__ForceSensor):
    def __init__(self, robot: Robot, time_step: float, device_name: str = "RFsr"):
        super().__init__(robot=robot, time_step=time_step, device_name=device_name)


class ForceSensorsCombined:
    MAX_FORCE: float = 200.0

    def __init__(self, robot: Robot, time_step: float):
        self.force_left_foot = ForceLeftFoot(robot=robot, time_step=time_step)
        self.force_right_foot = ForceRightFoot(robot=robot, time_step=time_step)

    @property
    def output_size(self) -> int:
        return 6

    def get_force(self) -> np.ndarray:
        return np.concatenate(
            [self.force_left_foot.get_force(), self.force_right_foot.get_force()],
            dtype=np.float32,
        )

    def get_force_normalized(self) -> np.ndarray:
        return np.clip(
            np.concatenate(
                [
                    self.force_left_foot.get_force(),
                    self.force_right_foot.get_force(),
                ],
                dtype=np.float32,
            )
            / self.MAX_FORCE,
            -1.0,
            1.0,
        )
