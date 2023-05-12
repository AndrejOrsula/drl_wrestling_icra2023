import os
import sys
import time

import numpy as np

sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from controller import Robot
from sensors.camera_sensor import *
from sensors.distance_sensor import SonarSensorsCombined
from sensors.force_sensor import ForceSensorsCombined
from sensors.inertial_sensor import InertialMeasurementUnit
from sensors.joint_position_sensor import JointPositionSensorsCombined
from sensors.touch_sensor import TouchSensorsCombined

TRAIN: bool = False
RANDOM_AGENT: bool = False and not TRAIN
DEBUG: bool = False and not TRAIN

if os.environ.get("CI"):
    TRAIN = False
    RANDOM_AGENT = False
    DEBUG: bool = False


class SpaceBot(Robot):

    def __init__(
        self,
        ## Observations
        # Vector
        observation_vector_enable: bool = True,
        observation_vector_enable_joint_positions: bool = True,
        observation_vector_enable_imu: bool = True,
        observation_vector_enable_sonars: bool = True,
        observation_vector_enable_force: bool = False,
        observation_vector_enable_touch: bool = False,
        # Camera
        observation_image_enable: bool = True,
        camera_height: int = 32,
        camera_width: int = 32,
        camera_crop_left: int = 20,
        camera_crop_right: int = 20,
        camera_crop_top: int = 0,
        camera_crop_bottom: int = 0,
    ):
        super().__init__()
        self.time_step = int(self.getBasicTimeStep())
        self.TIME_STEP: float = 0.001 * self.time_step

        ## Observations
        self.observation_vector_enable = observation_vector_enable
        self.observation_image_enable = observation_image_enable
        assert (
            self.observation_image_enable or self.observation_vector_enable
        ), "At least one of the observation types must be enabled"

        # Vector observations
        if self.observation_vector_enable:
            self.observation_vector_size: int = 0

            self.observation_vector_enable_joint_positions = (
                observation_vector_enable_joint_positions
            )
            if self.observation_vector_enable_joint_positions:
                self.joint_pos_sensors = JointPositionSensorsCombined(
                    robot=self,
                    time_step=self.time_step,
                    # TODO: Observe all joint positions
                    simplified_scheme=True,
                    include_fingers=False,
                )
                self.observation_vector_size += self.joint_pos_sensors.output_size

            self.observation_vector_enable_imu = observation_vector_enable_imu
            if self.observation_vector_enable_imu:
                self.imu = InertialMeasurementUnit(robot=self, time_step=self.time_step)
                self.observation_vector_size += self.imu.output_size

            self.observation_vector_enable_sonars = observation_vector_enable_sonars
            if self.observation_vector_enable_sonars:
                self.sonars = SonarSensorsCombined(robot=self, time_step=self.time_step)
                self.observation_vector_size += self.sonars.output_size

            self.observation_vector_enable_force = observation_vector_enable_force
            if self.observation_vector_enable_force:
                self.force_sensors = ForceSensorsCombined(
                    robot=self, time_step=self.time_step
                )
                self.observation_vector_size += self.force_sensors.output_size

            self.observation_vector_enable_touch = observation_vector_enable_touch
            if self.observation_vector_enable_touch:
                self.touch_sensors = TouchSensorsCombined(
                    robot=self, time_step=self.time_step
                )
                self.observation_vector_size += self.touch_sensors.output_size

        # Image observations
        if self.observation_image_enable:
            self.cameras = CameraTop(
                robot=self,
                time_step=self.time_step,
                crop_left=camera_crop_left,
                crop_right=camera_crop_right,
                crop_top=camera_crop_top,
                crop_bottom=camera_crop_bottom,
                resize_height=camera_height,
                resize_width=camera_width,
            )
            self.observation_image_height = self.cameras.height
            self.observation_image_width = self.cameras.width
            self.observation_image_channels = 3

    def low_level_controller(self):
        overtime: float = 0.0
        while True:
            time_before: float = time.time()

            if self.step(self.time_step) == -1:
                break

            ## Wait for the time step to finish
            time_to_sleep = self.TIME_STEP - (time.time() - time_before)
            if time_to_sleep > 0.0:
                time_to_sleep += overtime
                overtime = 0.0
                if time_to_sleep > 0.0:
                    time.sleep(time_to_sleep)
            else:
                overtime = time_to_sleep

    def get_observations(self) -> np.ndarray:
        if self.observation_vector_enable:
            joint_pos_obs_norm = np.empty(0)
            imu_obs_norm = np.empty(0)
            sonar_obs_norm = np.empty(0)
            force_obs_norm = np.empty(0)
            touch_obs_norm = np.empty(0)
            if self.observation_vector_enable_joint_positions:
                joint_pos_obs_norm = (
                    self.joint_pos_sensors.get_joint_position_normalized()
                )
            if self.observation_vector_enable_imu:
                imu_obs_norm = self.imu.get_inertial_measurements_normalized()
            if self.observation_vector_enable_sonars:
                sonar_obs_norm = self.sonars.get_distance_normalized()
            if self.observation_vector_enable_force:
                force_obs_norm = self.force_sensors.get_force_normalized()
            if self.observation_vector_enable_touch:
                touch_obs_norm = self.touch_sensors.get_touch_normalized()
            vector_obs = np.hstack(
                (
                    joint_pos_obs_norm,
                    imu_obs_norm,
                    sonar_obs_norm,
                    force_obs_norm,
                    touch_obs_norm,
                    self._is_action_being_replayed,
                )
            )
        if self.observation_image_enable:
            image_obs = self.cameras.get_image_rgb()[:, :, ::-1]

        if self.observation_vector_enable and self.observation_image_enable:
            return {
                "vector": vector_obs,
                "image": image_obs,
            }
        elif self.observation_vector_enable:
            return vector_obs
        elif self.observation_image_enable:
            return image_obs

