import numpy as np
from controller import DistanceSensor, Robot


class __SonarSensor:
    def __init__(self, robot: Robot, time_step: float, device_name: str):
        self.device: DistanceSensor = robot.getDevice(device_name)
        self.device.enable(time_step)

    def get_distance(self) -> float:
        return self.device.getValue()


class SonarLeft(__SonarSensor):
    def __init__(self, robot: Robot, time_step: float, device_name: str = "Sonar/Left"):
        super().__init__(robot=robot, time_step=time_step, device_name=device_name)


class SonarRight(__SonarSensor):
    def __init__(
        self, robot: Robot, time_step: float, device_name: str = "Sonar/Right"
    ):
        super().__init__(robot=robot, time_step=time_step, device_name=device_name)


class SonarSensorsCombined:
    # TODO: Adjust limits of sonars
    MAX_DISTANCE: float = 2.55

    def __init__(self, robot: Robot, time_step: float):
        self.sonar_left = SonarLeft(robot=robot, time_step=time_step)
        self.sonar_right = SonarRight(robot=robot, time_step=time_step)

    @property
    def output_size(self) -> int:
        return 2

    def get_distance(self) -> np.ndarray:
        return np.array(
            [self.sonar_left.get_distance(), self.sonar_right.get_distance()],
            dtype=np.float32,
        )

    def get_distance_normalized(self) -> np.ndarray:
        return np.array(
            [
                self.sonar_left.get_distance() / self.MAX_DISTANCE,
                self.sonar_right.get_distance() / self.MAX_DISTANCE,
            ],
            dtype=np.float32,
        )
        # TODO: Adjust normalization of out-of-range values
        # out = (
        #     np.array(
        #         [
        #             self.sonar_left.get_distance(),
        #             self.sonar_right.get_distance(),
        #         ],
        #         dtype=np.float32,
        #     )
        #     / self.MAX_DISTANCE
        # )
        # out[out >= 1.0] = -1.0
        # return out
