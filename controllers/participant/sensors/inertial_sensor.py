import numpy as np
from controller import Accelerometer, Gyro, Robot


class AccelerometerSensor:
    def __init__(
        self, robot: Robot, time_step: float, device_name: str = "accelerometer"
    ):
        self.device: Accelerometer = robot.getDevice(device_name)
        self.device.enable(time_step)

    def get_linear_acceleration(self) -> np.ndarray:
        return np.array(self.device.getValues(), dtype=np.float32)


class GyroscopeSensor:
    def __init__(self, robot: Robot, time_step: float, device_name: str = "gyro"):
        self.device: Gyro = robot.getDevice(device_name)
        self.device.enable(time_step)

    def get_angular_velocity(self) -> np.ndarray:
        return np.array(self.device.getValues(), dtype=np.float32)


class InertialMeasurementUnit:
    # TODO: Adjust limits of IMU
    CLIP_MAX_ACC: float = 10.0
    CLIP_MAX_GYRO: float = 1.0

    def __init__(self, robot: Robot, time_step: float):
        self.accelerometer = AccelerometerSensor(robot=robot, time_step=time_step)
        self.gyroscope = GyroscopeSensor(robot=robot, time_step=time_step)

    @property
    def output_size(self) -> int:
        return 6

    def get_inertial_measurements(self) -> np.ndarray:
        return np.concatenate(
            (
                self.accelerometer.get_linear_acceleration(),
                self.gyroscope.get_angular_velocity(),
            ),
            dtype=np.float32,
        )

    def get_inertial_measurements_normalized(
        self,
    ) -> np.ndarray:
        return np.concatenate(
            (
                np.clip(
                    self.accelerometer.get_linear_acceleration() / self.CLIP_MAX_ACC,
                    -1.0,
                    1.0,
                ),
                np.clip(
                    self.gyroscope.get_angular_velocity() / self.CLIP_MAX_GYRO,
                    -1.0,
                    1.0,
                ),
            ),
            dtype=np.float32,
        )
