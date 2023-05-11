import numpy as np
from controller import Robot, TouchSensor


class __TouchSensor:
    def __init__(self, robot: Robot, time_step: float, device_name: str):
        self.device: TouchSensor = robot.getDevice(device_name)
        self.device.enable(time_step)

    def get_touch(self) -> float:
        return self.device.getValue()


class TouchLeftFootLeftBumper(__TouchSensor):
    def __init__(
        self, robot: Robot, time_step: float, device_name: str = "LFoot/Bumper/Left"
    ):
        super().__init__(robot=robot, time_step=time_step, device_name=device_name)


class TouchLeftFootRightBumper(__TouchSensor):
    def __init__(
        self, robot: Robot, time_step: float, device_name: str = "LFoot/Bumper/Right"
    ):
        super().__init__(robot=robot, time_step=time_step, device_name=device_name)


class TouchRightFootLeftBumper(__TouchSensor):
    def __init__(
        self, robot: Robot, time_step: float, device_name: str = "RFoot/Bumper/Left"
    ):
        super().__init__(robot=robot, time_step=time_step, device_name=device_name)


class TouchRightFootRightBumper(__TouchSensor):
    def __init__(
        self, robot: Robot, time_step: float, device_name: str = "RFoot/Bumper/Right"
    ):
        super().__init__(robot=robot, time_step=time_step, device_name=device_name)


class TouchSensorsCombined:
    def __init__(self, robot: Robot, time_step: float):
        self.touch_left_foot_left_bumper = TouchLeftFootLeftBumper(
            robot=robot, time_step=time_step
        )
        self.touch_left_foot_right_bumper = TouchLeftFootRightBumper(
            robot=robot, time_step=time_step
        )
        self.touch_right_foot_left_bumper = TouchRightFootLeftBumper(
            robot=robot, time_step=time_step
        )
        self.touch_right_foot_right_bumper = TouchRightFootRightBumper(
            robot=robot, time_step=time_step
        )

    @property
    def output_size(self) -> int:
        return 4

    def get_touch(self) -> np.ndarray:
        return np.array(
            [
                self.touch_left_foot_left_bumper.get_touch(),
                self.touch_left_foot_right_bumper.get_touch(),
                self.touch_right_foot_left_bumper.get_touch(),
                self.touch_right_foot_right_bumper.get_touch(),
            ],
            dtype=np.float32,
        )

    def get_touch_normalized(self) -> np.ndarray:
        return (2.0 * self.get_touch()) - 1.0
