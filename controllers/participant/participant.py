import os
import sys
import threading
import time

import numpy as np

sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from actuators.gait_controller import GaitController
from actuators.joint_controller import *
from actuators.led_controller import LedControllersCombined
from actuators.replay_motion_controller import ReplayMotionController
from controller import Robot
from sensors.camera_sensor import *
from sensors.distance_sensor import SonarSensorsCombined
from sensors.force_sensor import ForceSensorsCombined
from sensors.inertial_sensor import InertialMeasurementUnit
from sensors.joint_position_sensor import JointPositionSensorsCombined
from sensors.rolling_average import RollingAverage
from sensors.touch_sensor import TouchSensorsCombined

TRAIN: bool = False
RANDOM_AGENT: bool = False and not TRAIN
DEBUG: bool = False and not TRAIN

if os.environ.get("CI"):
    TRAIN = False
    RANDOM_AGENT = False
    DEBUG: bool = False


class SpaceBot(Robot):
    MAX_TURNING_RADIUS: float = 2.0
    MIN_TURNING_RADIUS: float = 0.1
    GET_UP_TRIGGER_THRESHOLD: float = 0.75
    LED_HUE_DELTA: float = 0.015625

    def __init__(
        self,
        ## Observations
        # Vector
        observation_vector_enable: bool = True,
        observation_vector_enable_joint_positions: bool = True,
        observation_vector_enable_imu: bool = True,
        observation_vector_enable_sonars: bool = True,
        # TODO: Enable all sensors
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
        ## Actions
        # TODO: Test the other control schemes if there is enough time
        action_use_combined_scheme: bool = True,
        action_use_only_replay_motion: bool = False,
        action_joints_relative: bool = True,
        action_joints_relative_scaling: float = 0.15,
        include_fingers: bool = False,
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

            # Additional vector observations
            #   1 - self._is_action_being_replayed
            self.observation_vector_size += 1

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

        # Filters
        self.rolling_average_acceleration = RollingAverage(window_size=5)

        ## Actions
        # Enable static joints that are not controlled by the agent
        self._init_static_joints()

        # LEDs
        self.led_controllers = LedControllersCombined(
            robot=self, time_step=self.time_step
        )
        self.led_hue: float = 0.0

        # Configure action scheme
        self.action_use_combined_scheme = action_use_combined_scheme
        self.action_use_only_replay_motion = action_use_only_replay_motion
        assert not (
            self.action_use_combined_scheme and self.action_use_only_replay_motion
        ), "Only one of the special action schemes can be used at a time"
        if self.action_use_combined_scheme:
            # Gait controller
            #   4 Actions: forward_backward, left_right, turn, step_amplitude
            self.gait_controller = GaitController(
                robot=self, time_step=self.time_step, imu=self.imu
            )
            self.gait_desired_radius = 0.0
            self.gait_heading_angle = 0.0
            self.gait_step_amplitude = 0.5
            self.action_gait_controller_input_size = 4
            self.action_gait_indices = [
                i for i in range(self.action_gait_controller_input_size)
            ]

            # Replay controller
            #   1 Action: (positive values: get_up_front, negative values: get_up_back)
            self.replay_controller = ReplayMotionController()
            self._is_action_being_replayed = 0.0
            self.action_replay_controller_input_size = 1
            self.action_replay_controller_indices = [
                i
                for i in range(
                    self.action_gait_controller_input_size,
                    self.action_gait_controller_input_size
                    + self.action_replay_controller_input_size,
                )
            ]

            # Start the background thread of the low-level controller
            self._is_agent_ready = False
            self._thread = threading.Thread(target=self.low_level_controller)
            self._thread.start()

            # Joint controllers
            #   N Actions: 1 per joint
            # TODO: Consider re-enabling head controller (yaw only)
            self.head_controllers = HeadController(
                robot=self, time_step=self.time_step, yaw=False, pitch=False
            )
            self.action_head_controllers_indices = [
                i
                for i in range(
                    self.action_gait_controller_input_size
                    + self.action_replay_controller_input_size,
                    self.action_gait_controller_input_size
                    + self.action_replay_controller_input_size
                    + self.head_controllers.input_size,
                )
            ]

            # TODO: Consider disabling hand controllers
            self.hand_controllers = HandControllers(
                robot=self, time_step=self.time_step, enable=True
            )
            self.action_hand_controllers_indices = [
                i
                for i in range(
                    self.action_gait_controller_input_size
                    + self.action_replay_controller_input_size
                    + self.head_controllers.input_size,
                    self.action_gait_controller_input_size
                    + self.action_replay_controller_input_size
                    + self.head_controllers.input_size
                    + self.hand_controllers.input_size_combined,
                )
            ]

            # TODO: Consider using simplified scheme for arm controllers
            self.arm_controllers = ArmControllers(
                robot=self, time_step=self.time_step, simplified_scheme=False
            )
            self.action_arm_controllers_indices = [
                i
                for i in range(
                    self.action_gait_controller_input_size
                    + self.action_replay_controller_input_size
                    + self.head_controllers.input_size
                    + self.hand_controllers.input_size_combined,
                    self.action_gait_controller_input_size
                    + self.action_replay_controller_input_size
                    + self.head_controllers.input_size
                    + self.hand_controllers.input_size_combined
                    + self.arm_controllers.input_size,
                )
            ]

            self.action_joints_relative = action_joints_relative
            self.action_joints_relative_scaling = action_joints_relative_scaling

        elif self.action_use_only_replay_motion:
            self.replay_motion_controller = ReplayMotionController()
            self.action_replay_motion_size = self.replay_motion_controller.input_size

        else:
            self.joint_controllers = JointControllersCombined(
                robot=self,
                time_step=self.time_step,
                simplified_scheme=False,
                include_fingers=include_fingers,
            )
            self.action_joints_size = self.joint_controllers.input_size
            self.action_joints_relative = action_joints_relative
            self.action_joints_relative_scaling = action_joints_relative_scaling

    def low_level_controller(self):
        overtime: float = 0.0
        while True:
            time_before: float = time.time()

            if self.step(self.time_step) == -1:
                break

            ## Only play gait if the agent is ready and no action is being replayed
            if self._is_action_being_replayed != 0.0:
                if self.replay_controller.current_motion[1].isOver():
                    self._is_action_being_replayed = 0.0
            else:
                self.gait_controller.set_step_amplitude(self.gait_step_amplitude)
                self.gait_controller.command_to_motors(
                    desired_radius=self.gait_desired_radius,
                    heading_angle=self.gait_heading_angle,
                )

                self._set_default_pose_of_passive_joints()

            ## Change hue of the LEDs
            self.led_hue = (self.led_hue + self.LED_HUE_DELTA) % 1.0
            self.led_controllers.set_color_hsv(self.led_hue, 1.0, 1.0)

            ## Wait for the time step to finish
            time_to_sleep = self.TIME_STEP - (time.time() - time_before)
            if time_to_sleep > 0.0:
                time_to_sleep += overtime
                overtime = 0.0
                if time_to_sleep > 0.0:
                    time.sleep(time_to_sleep)
            else:
                overtime = time_to_sleep

    def apply_action(self, action: np.ndarray):
        self._is_agent_ready = True
        if self.action_use_combined_scheme:
            if self.step(self.time_step) == -1:
                return

            if self._is_action_being_replayed != 0.0:
                return

            (
                action_forward_backward,
                action_left_right,
                action_turn,
                action_step_amplitude,
            ) = action[self.action_gait_indices]
            action_get_up = action[self.action_replay_controller_indices]
            action_head = action[self.action_head_controllers_indices]
            action_hands = action[self.action_hand_controllers_indices]
            action_arms = action[self.action_arm_controllers_indices]

            ## Convert get_up to action
            if (
                np.abs(
                    self.rolling_average_acceleration(
                        self.imu.accelerometer.get_linear_acceleration()
                    )
                ).argmax()
                != 2
            ):
                if action_get_up > self.GET_UP_TRIGGER_THRESHOLD:
                    self._is_action_being_replayed = 1.0
                    self.replay_controller.play_motion_by_name("GetUpFront")
                elif action_get_up < -self.GET_UP_TRIGGER_THRESHOLD:
                    self._is_action_being_replayed = -1.0
                    self.replay_controller.play_motion_by_name("GetUpBack")
                return

            ## Combine forward_backward and left_right into self.gait_heading_angle
            self.gait_heading_angle = np.arctan2(
                -action_left_right, action_forward_backward
            )

            ## Convert turn to self.gait_desired_radius
            if abs(action_turn) < self.MIN_TURNING_RADIUS:
                self.gait_desired_radius = 100000.0
            elif action_turn > 0.0:
                self.gait_desired_radius = np.interp(
                    action_turn,
                    [self.MIN_TURNING_RADIUS, 1.0],
                    [self.MAX_TURNING_RADIUS, self.MIN_TURNING_RADIUS],
                )
            else:
                self.gait_desired_radius = np.interp(
                    action_turn,
                    [-1.0, -self.MIN_TURNING_RADIUS],
                    [-self.MIN_TURNING_RADIUS, -self.MAX_TURNING_RADIUS],
                )

            ## Convert gait_step_amplitude to self.gait_step_amplitude
            self.gait_step_amplitude = np.interp(
                action_step_amplitude,
                [-1.0, 1.0],
                [0.0, 1.0],
            )

            ## Convert gripper_left and gripper_right to gripper action
            if len(action_hands) == 2:
                self.hand_controllers.set_joint_position_left_normalized(
                    action_hands[0]
                )
                self.hand_controllers.set_joint_position_right_normalized(
                    action_hands[1]
                )

            ## Convert head to head action
            if self.action_joints_relative:
                self.head_controllers.set_joint_position_relative_normalized(
                    action_head,
                    relative_scaling_factor=self.action_joints_relative_scaling,
                )
                self.arm_controllers.set_joint_position_relative_normalized(
                    action_arms,
                    relative_scaling_factor=self.action_joints_relative_scaling,
                )
            else:
                self.head_controllers.set_joint_position_normalized(action_head)
                self.arm_controllers.set_joint_position_normalized(action_arms)
        else:
            if self.action_use_only_replay_motion:
                self.replay_motion_controller.play_motion_by_index(action)
            else:
                if self.action_joints_relative:
                    self.joint_controllers.set_joint_position_relative_normalized(
                        action,
                        relative_scaling_factor=self.action_joints_relative_scaling,
                    )
                else:
                    self.joint_controllers.set_joint_position_normalized(action)

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
