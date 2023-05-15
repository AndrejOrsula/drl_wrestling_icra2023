import os
import sys
import threading
import time
from typing import Tuple

import gym
import numpy as np
from controller import Robot, Supervisor

sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from actuators.gait_controller import SlideGaitController
from actuators.joint_controller import *
from actuators.led_controller import LedControllersCombined
from actuators.replay_motion_controller import ReplayMotionController
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
    MIN_TURNING_RADIUS: float = 0.25

    GET_UP_TRIGGER_THRESHOLD: float = 0.75
    GET_UP_MIN_XY_ACCELERATION_MAGNITUDE: float = 5.0

    LED_HUE_DELTA: float = 0.015625

    EMULATE_STARTUP_DELAY_DURING_TRAINING: bool = True
    STARTUP_DELAY_MIN: float = 0.0
    STARTUP_DELAY_MAX: float = 10.0

    def __init__(
        self,
        ## Observations
        # Vector
        observation_vector_enable: bool = True,
        observation_vector_enable_joint_positions: bool = True,
        observation_vector_enable_imu: bool = True,
        observation_vector_enable_sonars: bool = True,
        observation_vector_enable_force: bool = True,
        observation_vector_enable_touch: bool = False,
        # Camera
        observation_image_enable: bool = True,
        camera_height: int = 24,
        camera_width: int = 24,
        camera_crop_left: int = 20,
        camera_crop_right: int = 20,
        camera_crop_top: int = 0,
        camera_crop_bottom: int = 0,
        ## Actions
        # TODO: Test the other control schemes if there is enough time
        action_use_combined_scheme: bool = True,
        action_use_only_replay_motion: bool = False,
        action_joints_relative: bool = True,
        action_joints_relative_scaling: float = 0.5,
        include_fingers: bool = False,
        ## Training
        train: bool = TRAIN,
        reward_for_each_step_standing: float = 1.0,
        reward_knockout_trainee: float = 50.0,
        reward_knockout_opponent: float = 2.5,
        reward_knockout_opponent_without_trainee_knockout: float = 20.0,
        max_distance_knockout_opponent: float = 0.8,
        reward_distance_from_centre: float = 1.0,
        reward_coverage_delta_trainee: float = 40.0,
        reward_coverage_total_trainee: float = 1.0,
        reward_coverage_delta_opponent: float = 1.0,
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
                    head_pitch=False,
                    head_yaw=True,
                    simplified_scheme_legs=False,
                    simplified_scheme_arms=False,
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
        self.rolling_average_acceleration = RollingAverage(window_size=4)

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
            #   ~4~ 3 Actions: forward_backward, left_right, turn, ~step_amplitude~
            self.gait_controller = SlideGaitController(
                robot=self, time_step=self.time_step
            )
            self.gait_desired_radius = 100000.0
            self.gait_heading_angle = 0.0
            # self.gait_step_amplitude = 1.0
            self.action_gait_controller_input_size = 3
            self.action_gait_indices = [
                i for i in range(self.action_gait_controller_input_size)
            ]

            # Replay controller
            #   1 Action: (positive values: get_up_front, negative values: get_up_back)
            self.replay_controller = ReplayMotionController(
                motion_list=("Prepare", "GetUpFrontFast", "GetUpBackFast"),
                motion_list_reverse=(),
            )
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
            self._is_agent_ready_stance_taken = False
            self._thread = threading.Thread(target=self.low_level_controller)
            self._thread.start()

            # Joint controllers
            #   N Actions: 1 per joint
            self.head_controllers = HeadController(
                robot=self, time_step=self.time_step, yaw=True, pitch=False
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

            self.hand_controllers = HandControllers(
                robot=self, time_step=self.time_step, enable=False
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
            self.replay_controller = ReplayMotionController()
            self.action_replay_motion_size = self.replay_controller.input_size

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

        # Trainer (if training is enabled and supervisor is available)
        self.is_training = train and self.getSupervisor()
        if self.is_training:
            self.trainer = Trainer(
                reward_for_each_step_standing=reward_for_each_step_standing,
                reward_knockout_trainee=reward_knockout_trainee,
                reward_knockout_opponent=reward_knockout_opponent,
                reward_knockout_opponent_without_trainee_knockout=reward_knockout_opponent_without_trainee_knockout,
                max_distance_knockout_opponent=max_distance_knockout_opponent,
                reward_distance_from_centre=reward_distance_from_centre,
                reward_coverage_delta_trainee=reward_coverage_delta_trainee,
                reward_coverage_total_trainee=reward_coverage_total_trainee,
                reward_coverage_delta_opponent=reward_coverage_delta_opponent,
            )

    def low_level_controller(self):
        overtime: float = 0.0
        while True:
            time_before: float = time.time()

            if self.step(self.time_step) == -1:
                break

            ## Until the agent is ready, perform manual actions
            # self.replay_controller.set_wait_until_finished(self._is_agent_ready)
            if not self._is_agent_ready:
                if not self._is_agent_ready_stance_taken:
                    if self._is_action_being_replayed == 0.0:
                        self.replay_controller.play_motion_by_name("Prepare")
                        self._is_action_being_replayed = 1.0
                    elif self.replay_controller.current_motion[1].isOver():
                        self._is_agent_ready_stance_taken = True
                        self._is_action_being_replayed = 0.0
                else:
                    self.gait_controller.set_step_amplitude(0.5)
                    self.gait_controller.command_to_motors(
                        desired_radius=100000.0,
                        heading_angle=0.0,
                    )

            else:
                ## Only play gait if the agent is ready and no action is being replayed
                if self._is_action_being_replayed != 0.0:
                    if self.replay_controller.current_motion[1].isOver():
                        self._is_action_being_replayed = 0.0
                else:
                    self.gait_controller.set_step_amplitude(1.0)
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
        if self._is_agent_ready_stance_taken:
            self._is_agent_ready = True
        else:
            return
        if self.action_use_combined_scheme:
            if self.is_training:
                if self.step(self.time_step) == -1:
                    return

            if self._is_action_being_replayed != 0.0:
                return

            (
                action_forward_backward,
                action_left_right,
                action_turn,
                # action_step_amplitude,
            ) = action[self.action_gait_indices]
            action_get_up = action[self.action_replay_controller_indices]
            action_head = action[self.action_head_controllers_indices]
            action_hands = action[self.action_hand_controllers_indices]
            action_arms = action[self.action_arm_controllers_indices]

            ## Convert get_up to action
            if (
                np.linalg.norm(
                    np.abs(
                        self.rolling_average_acceleration(
                            self.imu.accelerometer.get_linear_acceleration()
                        )[:2]
                    )
                )
                > self.GET_UP_MIN_XY_ACCELERATION_MAGNITUDE
            ):
                if action_get_up > self.GET_UP_TRIGGER_THRESHOLD:
                    self._is_action_being_replayed = 1.0
                    self.replay_controller.play_motion_by_name("GetUpFrontFast")
                elif action_get_up < -self.GET_UP_TRIGGER_THRESHOLD:
                    self._is_action_being_replayed = -1.0
                    self.replay_controller.play_motion_by_name("GetUpBackFast")
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

            # ## Convert gait_step_amplitude to self.gait_step_amplitude
            # self.gait_step_amplitude = np.interp(
            #     action_step_amplitude,
            #     [-1.0, 1.0],
            #     [0.0, 1.0],
            # )

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
                self.replay_controller.play_motion_by_index(action)
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

    def get_reward(self, **kwargs) -> Tuple[float, bool]:
        if self.is_training:
            return self.trainer.get_reward()
        else:
            return 0.0, False

    def reset(self):
        self.rolling_average_acceleration.reset()

        if self.action_use_combined_scheme:
            self.gait_controller.reset()
            self._is_action_being_replayed = 0.0
            self._is_agent_ready = False
            self._is_agent_ready_stance_taken = False

        if self.is_training:
            self.trainer.reset()
            for _ in range(2):
                self.replay_controller.stop_current_motion()
                self.replay_controller.play_motion_by_name("Prepare")
                self.step(self.time_step)

            if self.EMULATE_STARTUP_DELAY_DURING_TRAINING:
                time.sleep(
                    np.random.uniform(self.STARTUP_DELAY_MIN, self.STARTUP_DELAY_MAX)
                )

    def _init_static_joints(self):
        self.__ControllerHeadPitch = ControllerHeadPitch(
            robot=self, time_step=self.time_step
        )
        self.__ControllerLPhalanx1 = ControllerLPhalanx1(
            robot=self, time_step=self.time_step
        )
        self.__ControllerLPhalanx2 = ControllerLPhalanx2(
            robot=self, time_step=self.time_step
        )
        self.__ControllerLPhalanx3 = ControllerLPhalanx3(
            robot=self, time_step=self.time_step
        )
        self.__ControllerLPhalanx4 = ControllerLPhalanx4(
            robot=self, time_step=self.time_step
        )
        self.__ControllerLPhalanx5 = ControllerLPhalanx5(
            robot=self, time_step=self.time_step
        )
        self.__ControllerLPhalanx6 = ControllerLPhalanx6(
            robot=self, time_step=self.time_step
        )
        self.__ControllerLPhalanx7 = ControllerLPhalanx7(
            robot=self, time_step=self.time_step
        )
        self.__ControllerLPhalanx8 = ControllerLPhalanx8(
            robot=self, time_step=self.time_step
        )
        self.__ControllerRPhalanx1 = ControllerRPhalanx1(
            robot=self, time_step=self.time_step
        )
        self.__ControllerRPhalanx2 = ControllerRPhalanx2(
            robot=self, time_step=self.time_step
        )
        self.__ControllerRPhalanx3 = ControllerRPhalanx3(
            robot=self, time_step=self.time_step
        )
        self.__ControllerRPhalanx4 = ControllerRPhalanx4(
            robot=self, time_step=self.time_step
        )
        self.__ControllerRPhalanx5 = ControllerRPhalanx5(
            robot=self, time_step=self.time_step
        )
        self.__ControllerRPhalanx6 = ControllerRPhalanx6(
            robot=self, time_step=self.time_step
        )
        self.__ControllerRPhalanx7 = ControllerRPhalanx7(
            robot=self, time_step=self.time_step
        )
        self.__ControllerRPhalanx8 = ControllerRPhalanx8(
            robot=self, time_step=self.time_step
        )
        self._set_default_pose_of_passive_joints()

    def _set_default_pose_of_passive_joints(self):
        self.__ControllerHeadPitch.set_joint_position(0.12)
        self.__ControllerLPhalanx1.set_joint_position(1.0)
        self.__ControllerLPhalanx2.set_joint_position(1.0)
        self.__ControllerLPhalanx3.set_joint_position(1.0)
        self.__ControllerLPhalanx4.set_joint_position(1.0)
        self.__ControllerLPhalanx5.set_joint_position(1.0)
        self.__ControllerLPhalanx6.set_joint_position(1.0)
        self.__ControllerLPhalanx7.set_joint_position(0.0)
        self.__ControllerLPhalanx8.set_joint_position(0.0)
        self.__ControllerRPhalanx1.set_joint_position(1.0)
        self.__ControllerRPhalanx2.set_joint_position(1.0)
        self.__ControllerRPhalanx3.set_joint_position(1.0)
        self.__ControllerRPhalanx4.set_joint_position(1.0)
        self.__ControllerRPhalanx5.set_joint_position(1.0)
        self.__ControllerRPhalanx6.set_joint_position(1.0)
        self.__ControllerRPhalanx7.set_joint_position(0.0)
        self.__ControllerRPhalanx8.set_joint_position(0.0)


class Trainer(Supervisor):
    ## Constants taken from the supervisor
    ROBOT_MIN_Z: float = 0.9
    RING_MAX_XY: float = 1.0
    EXPLOSION_MAX_XYZ: float = 1.5

    ## Additional constants
    ARENA_CENTER_XY: np.ndarray = np.zeros(2, dtype=np.float32)
    MAX_REWARDING_DISTANCE_FROM_CENTRE: float = 0.85
    MIN_DELTA_COVERAGE: float = 0.0001
    MAX_DELTA_COVERAGE: float = 1.0

    def __init__(
        self,
        reward_for_each_step_standing: float,
        reward_knockout_trainee: float,
        reward_knockout_opponent: float,
        reward_knockout_opponent_without_trainee_knockout: float,
        max_distance_knockout_opponent: float,
        reward_distance_from_centre: float,
        reward_coverage_delta_trainee: float,
        reward_coverage_total_trainee: float,
        reward_coverage_delta_opponent: float,
    ):
        from controller import Node

        subject_name = self.getName()
        if subject_name == "participant":
            self.id_trainee = 0
            self.id_opponent = 1
        elif subject_name == "opponent":
            self.id_trainee = 1
            self.id_opponent = 0
        else:
            raise Exception(f"Unknown subject of training (name: '{subject_name}')")

        self.robot: Tuple[Node, Node] = (
            self.getFromDef("WRESTLER_RED"),
            self.getFromDef("WRESTLER_BLUE"),
        )
        self.robot_head: Tuple[Node, Node] = (
            self.robot[0].getFromProtoDef("HEAD_SLOT"),
            self.robot[1].getFromProtoDef("HEAD_SLOT"),
        )

        self.time_step = int(self.getBasicTimeStep())
        self.current_time = self.getTime()

        self.reset()

        self.reward_for_each_step_standing = reward_for_each_step_standing
        self.reward_knockout_trainee = reward_knockout_trainee
        self.reward_knockout_opponent = reward_knockout_opponent
        self.reward_knockout_opponent_without_trainee_knockout = (
            reward_knockout_opponent_without_trainee_knockout
        )
        self.max_distance_knockout_opponent = max_distance_knockout_opponent
        self.reward_distance_from_centre = reward_distance_from_centre
        self.reward_coverage_delta_trainee = reward_coverage_delta_trainee
        self.reward_coverage_total_trainee = reward_coverage_total_trainee
        self.reward_coverage_delta_opponent = reward_coverage_delta_opponent

    def reset(self):
        self._current_position_body = np.array(
            [self.robot[i].getPosition() for i in range(2)], dtype=np.float32
        )
        self._current_position_head = np.array(
            [self.robot_head[i].getPosition() for i in range(2)], dtype=np.float32
        )
        self._is_knocked = np.zeros(2, dtype=bool)
        self._is_near_opponent = False
        self._was_near_opponent = False
        self._previous_reward = 0.0

        self._coverage_min_position = np.zeros((2, 3), dtype=np.float32)
        self._coverage_min_position[0, 0] = -self.MAX_REWARDING_DISTANCE_FROM_CENTRE
        self._coverage_min_position[1, 0] = self.MAX_REWARDING_DISTANCE_FROM_CENTRE
        self._coverage_max_position = self._coverage_min_position.copy()
        self._coverage = np.zeros(2, dtype=np.float32)
        self._coverage_delta = np.zeros(2, dtype=np.float32)

    def get_reward(
        self,
    ) -> Tuple[float, bool]:
        # Update time and reset the episode if needed (detected as a step back in time)
        new_time = self.getTime()
        delta_time = new_time - self.current_time
        self.current_time = new_time
        if delta_time < 0.0:
            return self._previous_reward, True

        # Update position of the robots and their heads
        self._current_position_body = np.array(
            [self.robot[i].getPosition() for i in range(2)], dtype=np.float32
        )
        self._current_position_head = np.array(
            [self.robot_head[i].getPosition() for i in range(2)], dtype=np.float32
        )

        # Update ring coverage and knock-out counter for each robot (based on head position)
        for i in range(2):
            # Update knock-out counter if the robot is below the height threshold
            #                          or the robot exploded (any coordinate above a threshold)
            self._is_knocked[i] = (
                self._current_position_head[i, 2] < self.ROBOT_MIN_Z
                or np.abs(self._current_position_head[i, 0]) > self.EXPLOSION_MAX_XYZ
                or np.abs(self._current_position_head[i, 1]) > self.EXPLOSION_MAX_XYZ
                or self._current_position_head[i, 2] > self.EXPLOSION_MAX_XYZ
            )

            # Update ring coverage if the robot is inside the ring
            if (
                np.abs(self._current_position_head[i, 0]) < self.RING_MAX_XY
                and np.abs(self._current_position_head[i, 1]) < self.RING_MAX_XY
            ):
                new_coverage = 0.0
                for j in range(2):
                    if (
                        self._current_position_head[i, j]
                        < self._coverage_min_position[i, j]
                    ):
                        self._coverage_min_position[i, j] = self._current_position_head[
                            i, j
                        ]
                    elif (
                        self._current_position_head[i, j]
                        > self._coverage_max_position[i, j]
                    ):
                        self._coverage_max_position[i, j] = self._current_position_head[
                            i, j
                        ]
                    box = (
                        self._coverage_max_position[i, j]
                        - self._coverage_min_position[i, j]
                    )
                    new_coverage += box * box
                new_coverage = np.sqrt(new_coverage)
                self._coverage_delta[i] = new_coverage - self._coverage[i]
                if (
                    self._coverage_delta[i] < self.MIN_DELTA_COVERAGE
                    or self._coverage_delta[i] > self.MAX_DELTA_COVERAGE
                ):
                    self._coverage_delta[i] = 0.0
                self._coverage[i] += self._coverage_delta[i]

        # Determine if the trainee is close enough to the opponent for a knock-out (based on robot/body position)
        # It is enough to be near only once per knock-out to be considered near enough for the
        # whole duration of the specific knock-out (reset if the opponent stands up)
        if self._was_near_opponent:
            if self._is_knocked[self.id_opponent]:
                self._is_near_opponent = True
            else:
                self._is_near_opponent = False
                self._was_near_opponent = False
        elif self._is_knocked[self.id_opponent]:
            distance_between_robots = np.linalg.norm(
                self._current_position_body[self.id_trainee, :2]
                - self._current_position_body[self.id_opponent, :2]
            )
            self._is_near_opponent = (
                distance_between_robots < self.max_distance_knockout_opponent
            )
            self._was_near_opponent = self._is_near_opponent

        ## Compute reward
        reward = 0.0

        ## If the trainee is knocked, subtract reward (continuous, down to -self.reward_knockout_trainee)
        if self._is_knocked[self.id_trainee]:
            if (
                np.abs(self._current_position_head[self.id_trainee, 0])
                > self.EXPLOSION_MAX_XYZ
                or np.abs(self._current_position_head[self.id_trainee, 1])
                > self.EXPLOSION_MAX_XYZ
                or self._current_position_head[self.id_trainee, 2]
                > self.EXPLOSION_MAX_XYZ
            ):
                reward -= self.reward_knockout_trainee
            elif self._current_position_head[self.id_trainee, 2] < self.ROBOT_MIN_Z:
                trainee_distance_from_min_z = (
                    self.ROBOT_MIN_Z - self._current_position_head[self.id_trainee, 2]
                )
                trainee_distance_from_min_z_normalized = (
                    trainee_distance_from_min_z / self.ROBOT_MIN_Z
                )
                reward -= (
                    self.reward_knockout_trainee
                    * trainee_distance_from_min_z_normalized
                )
            else:
                raise ValueError("Unexpected position of the trainee")
        else:
            # If the trainee is not knocked, add reward (+self.scale_standing)
            reward += self.reward_for_each_step_standing
            # Furthermore, add reward based on the total coverage so far
            reward += (
                self.reward_coverage_total_trainee * self._coverage[self.id_trainee]
            )

        ## If the opponent is knocked and the trainee is near enough, add reward
        if self._is_knocked[self.id_opponent] and self._is_near_opponent:
            opponent_knocked_multiplier = (
                1.0
                if self._is_knocked[self.id_trainee]
                else self.reward_knockout_opponent_without_trainee_knockout
            )
            reward += opponent_knocked_multiplier * self.reward_knockout_opponent

        ## Reward for the closer the trainee is from the centre of the arena (if not knocked)
        if not self._is_knocked[self.id_trainee]:
            trainee_distance_to_centre = np.linalg.norm(
                self._current_position_body[self.id_trainee, :2] - self.ARENA_CENTER_XY,
            )
            trainee_distance_to_centre_normalized = (
                min(self.MAX_REWARDING_DISTANCE_FROM_CENTRE, trainee_distance_to_centre)
                / self.MAX_REWARDING_DISTANCE_FROM_CENTRE
            )
            reward += (
                1.0 - trainee_distance_to_centre_normalized
            ) * self.reward_distance_from_centre

        # Update reward for ring coverage (positive for trainee, negative for opponent)
        # The reward is proportional to the increase in ring coverage
        reward += (
            self.reward_coverage_delta_trainee * self._coverage_delta[self.id_trainee]
        )
        reward -= (
            self.reward_coverage_delta_opponent * self._coverage_delta[self.id_opponent]
        )

        self._previous_reward = reward
        return reward, False


class ParticipantEnv(gym.Env):
    def __init__(self, train: bool = TRAIN, **kwargs):
        self.robot = SpaceBot(train=train, **kwargs)

    @property
    def observation_space(self):
        if self.robot.observation_vector_enable:
            vector_obs = gym.spaces.Box(
                low=-1.0,
                high=1.0,
                shape=(self.robot.observation_vector_size,),
                dtype=np.float32,
            )
        if self.robot.observation_image_enable:
            image_obs = gym.spaces.Box(
                low=0,
                high=255,
                shape=(
                    self.robot.observation_image_height,
                    self.robot.observation_image_width,
                    self.robot.observation_image_channels,
                ),
                dtype=np.uint8,
            )

        if self.robot.observation_vector_enable and self.robot.observation_image_enable:
            return gym.spaces.Dict(
                spaces={
                    "vector": vector_obs,
                    "image": image_obs,
                }
            )
        elif self.robot.observation_vector_enable:
            return vector_obs
        elif self.robot.observation_image_enable:
            return image_obs

    @property
    def action_space(self):
        if self.robot.action_use_combined_scheme:
            n_actions = (
                self.robot.action_gait_controller_input_size
                + self.robot.action_replay_controller_input_size
                + self.robot.head_controllers.input_size
                + self.robot.hand_controllers.input_size_combined
                + self.robot.arm_controllers.input_size
            )
            return gym.spaces.Box(
                low=-1.0,
                high=1.0,
                shape=(n_actions,),
                dtype=np.float32,
            )
        elif self.robot.action_use_only_replay_motion:
            return gym.spaces.Discrete(self.robot.action_replay_motion_size)
        else:
            if self.robot.action_leds_enable:
                n_actions = (
                    self.robot.joint_controllers.input_size
                    + self.robot.led_controllers.input_size
                )
            else:
                n_actions = self.robot.joint_controllers.input_size
            return gym.spaces.Box(
                low=-1.0,
                high=1.0,
                shape=(n_actions,),
                dtype=np.float32,
            )

    @property
    def reward_range(self):
        # Note: Approximate values, +-50.0 is more realistic with the current defaults
        return (-100.0, 100.0)

    def step(self, action):
        self.robot.apply_action(action)
        obs = self.robot.get_observations()
        reward, is_done = self.robot.get_reward()
        info = {}
        return obs, reward, is_done, info

    def reset(self):
        self.robot.reset()
        return self.robot.get_observations()

    def render(self, mode="human"):
        if mode == "human":
            pass
        elif mode == "rgb_array":
            return self.robot.cameras.get_image_rgb()
        else:
            raise NotImplementedError

    def _random_agent(self, debug: bool = DEBUG):
        observations = self.reset()
        while True:
            if self.robot.action_use_only_replay_motion:
                action = np.random.randint(0, self.robot.action_replay_motion_size)
            else:
                action = np.random.uniform(
                    -1,
                    1,
                    size=self.action_space.shape,
                )

            observations, reward, done, info = self.step(action)

            if done:
                self.reset()

            if debug:
                np.set_printoptions(precision=3, suppress=True, floatmode="fixed")
                print(
                    "\n------------------------------------------------------------------------"
                )
                print(f"action.shape: {action.shape}")
                if isinstance(observations, dict):
                    if "vector" in observations:
                        print(f"observations_vector:\n{observations['vector']}")
                        print(
                            f"observations_vector.shape: {observations['vector'].shape}"
                        )
                    if "image" in observations:
                        import cv2

                        cv2.imshow("image_obs", observations["image"][..., ::-1])
                        cv2.waitKey(1)
                        print(
                            f"observations_image.shape: {observations['image'].shape}"
                        )
                else:
                    print(f"observations: {observations}")
                print(f"reward: {reward}")
                print(f"done: {done}")


def dreamerv3(train: bool = TRAIN, **kwargs):
    import dreamerv3
    from dreamerv3 import embodied
    from embodied.envs import from_gym

    ## Apply monkey patch to accommodate multiple agents running in parallel
    if train:
        XLA_PYTHON_CLIENT_MEM_FRACTION: str = "0.38"
        __monkey_patch__setup_original = dreamerv3.Agent._setup

        def __monkey_patch__setup(self):
            __monkey_patch__setup_original(self)
            os.environ[
                "XLA_PYTHON_CLIENT_MEM_FRACTION"
            ] = XLA_PYTHON_CLIENT_MEM_FRACTION

        dreamerv3.Agent._setup = __monkey_patch__setup
        ##

    config = embodied.Config(dreamerv3.configs["defaults"])
    config = config.update(
        {
            "logdir": os.path.join(
                os.path.abspath(os.path.dirname(__file__)), "logdir"
            ),
            "replay_size": 5e6,
            # "jax.platform": "cpu",
            # "jax.jit": False,
            "jax.precision": "float16",
            "jax.prealloc": train,
            "run.steps": 1e8,
            "run.log_every": 600,
            "run.train_ratio": 1024,
            "batch_size": 32,
            "batch_length": 64,
            "imag_horizon": 15,
            # rssm
            "rssm.deter": 512,
            "rssm.units": 512,
            "rssm.stoch": 32,
            "rssm.classes": 32,
            # encoder/decoder
            "encoder.mlp_keys": "vector",
            "encoder.cnn_keys": "image",
            "encoder.mlp_layers": 2,
            "encoder.mlp_units": 256,
            "encoder.cnn_depth": 8,
            "encoder.minres": 6,
            "decoder.mlp_keys": "vector",
            "decoder.cnn_keys": "image",
            "decoder.mlp_layers": 2,
            "decoder.mlp_units": 256,
            "decoder.cnn_depth": 8,
            "decoder.minres": 6,
            # actor/critic
            "actor.layers": 2,
            "actor.units": 256,
            "critic.layers": 2,
            "critic.units": 512,
            # reward
            "reward_head.layers": 2,
            "reward_head.units": 512,
            # cont
            "cont_head.layers": 2,
            "cont_head.units": 512,
            # disag
            "disag_head.layers": 2,
            "disag_head.units": 512,
        }
    )
    if not train:
        config = config.update(
            {
                "run.from_checkpoint": os.path.join(
                    os.path.abspath(os.path.dirname(__file__)),
                    "models",
                    "model01.ckpt",
                ),
            }
        )

    config = embodied.Flags(config).parse()
    logdir = embodied.Path(config.logdir)
    step = embodied.Counter()
    logger = embodied.Logger(
        step,
        [
            # embodied.logger.TerminalOutput(),
            # embodied.logger.JSONLOutput(logdir, "metrics.jsonl"),
            embodied.logger.TensorBoardOutput(logdir),
            # embodied.logger.WandBOutput(logdir.name, config),
            # embodied.logger.MLFlowOutput(logdir.name),
        ],
    )

    env = ParticipantEnv(train=True, **kwargs)
    env = from_gym.FromGym(env, obs_key="vector")
    env = dreamerv3.wrap_env(env, config)
    env = embodied.BatchEnv([env], parallel=False)
    agent = dreamerv3.Agent(env.obs_space, env.act_space, step, config)
    replay = embodied.replay.Uniform(
        config.batch_length, config.replay_size, logdir / "replay"
    )
    args = embodied.Config(
        **config.run,
        logdir=config.logdir,
        batch_steps=config.batch_size * config.batch_length,
    )

    if train:
        embodied.run.train(agent, env, replay, logger, args)
    else:
        driver = embodied.Driver(env)
        checkpoint = embodied.Checkpoint()
        checkpoint.agent = agent
        checkpoint.load(args.from_checkpoint, keys=["agent"])
        policy = lambda *args: agent.policy(*args, mode="eval")

        while True:
            _time_before = time.time()
            driver._step(policy, 0, 0)
            print(f"{time.time() - _time_before:.3f}", flush=True)


def __get_cmd_stdout(cmd) -> str:
    import subprocess

    try:
        info = subprocess.check_output(cmd, stderr=subprocess.STDOUT)
        info = info.decode("utf8")
    except Exception as err:
        info = f"Executing '{cmd}' failed: {err}"
    return info.strip()


if __name__ == "__main__":
    print(__get_cmd_stdout(["nvidia-smi"]))

    if RANDOM_AGENT:
        wrestler = ParticipantEnv(train=TRAIN)
        wrestler._random_agent()
    else:
        dreamerv3(train=TRAIN)
