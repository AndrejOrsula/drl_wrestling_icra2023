import numpy as np
from ahrs.filters import AngularRate, Madgwick, Mahony
from scipy.spatial.transform import Rotation as R
from sensors.rolling_average import RollingAverage


class PoseEstimator:
    def __init__(
        self,
        accelerometer,
        gyroscope,
        time_step,
        algorithm="madgwick",
        rolling_average_window_size=2,
    ):
        # TODO: Adjust parameters of the pose estimator
        self.accelerometer = accelerometer
        self.gyroscope = gyroscope
        self.time_step = time_step / 1000.0
        self.algorithm = algorithm

        self.accelerometer_rolling_average = RollingAverage(
            window_size=rolling_average_window_size
        )
        self.gyroscope_rolling_average = RollingAverage(
            window_size=rolling_average_window_size
        )

        self.reset()

    def reset(self):
        self.accelerometer_rolling_average.reset()
        self.gyroscope_rolling_average.reset()

        if self.algorithm == "mahony":
            self.mahony = Mahony(Dt=self.time_step, q0=[1.0, 0.0, 0.0, 0.0])
        elif self.algorithm == "madgwick":
            self.madgwick = Madgwick(Dt=self.time_step, q0=[1.0, 0.0, 0.0, 0.0])
        elif self.algorithm == "angular_rate":
            self.angular_rate = AngularRate(Dt=self.time_step, q0=[1.0, 0.0, 0.0, 0.0])

        self.Q = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        self.euler_angles = np.array([0.0, 0.0, 0.0], dtype=np.float32)

    def update_pose_estimation(self):
        acc = self.correct_accelerometer_orientation(
            self.accelerometer_rolling_average(
                self.accelerometer.get_linear_acceleration()
            )
        )
        gyro = self.gyroscope_rolling_average(self.gyroscope.get_angular_velocity())
        if self.algorithm == "tilt":
            self.euler_angles = self.get_tilt(acc)
            self.Q = self.roll_pitch_yaw_to_quaternion(self.euler_angles)
        elif self.algorithm == "mahony":
            self.Q = self.mahony.updateIMU(self.Q, gyr=gyro, acc=acc)
        elif self.algorithm == "madgwick":
            self.Q = self.madgwick.updateIMU(self.Q, gyr=gyro, acc=acc)
        elif self.algorithm == "angular_rate":
            self.Q = self.angular_rate.update(self.Q, gyr=gyro)
        elif self.algorithm == "manual_angular_rate":
            self.Q = self.integrate_gyro(self.Q, gyro)
        else:
            raise Exception("Unknown algorithm: " + self.algorithm)
        self.euler_angles = self.quaternion_to_roll_pitch_yaw(self.Q)

    def get_roll_pitch_yaw(self):
        self.update_pose_estimation()
        return self.euler_angles

    def get_quaternion(self):
        self.update_pose_estimation()
        return self.Q

    def correct_accelerometer_orientation(self, acc):
        acc[1] = -acc[1]
        acc[2] = -acc[2]
        return acc

    def from_ahrs_quaternion_convention_to_scipy(self, Q):
        Q_prime = np.array([Q[1], Q[2], Q[3], Q[0]])
        return Q_prime

    def from_scipy_to_ahrs_quaternion_convention(self, Q):
        Q_prime = np.array([Q[3], Q[0], Q[1], Q[2]])
        return Q_prime

    def integrate_gyro(self, current_value, gyro_values):
        identity_matrix = np.eye(4, 4)
        gyro_matrix = np.array(
            [
                [0.0, -gyro_values[0], -gyro_values[1], -gyro_values[2]],
                [gyro_values[0], 0.0, gyro_values[2], -gyro_values[1]],
                [gyro_values[1], -gyro_values[2], 0.0, gyro_values[0]],
                [gyro_values[2], gyro_values[1], -gyro_values[0], 0.0],
            ]
        )

        rotation_matrix = self.time_step_s / 2 * gyro_matrix + identity_matrix
        new_quaternion = rotation_matrix @ current_value
        return new_quaternion / np.linalg.norm(new_quaternion)

    def get_tilt(self, acc):
        ax, ay, az = acc
        roll = np.arctan2(ay, az)
        pitch = np.arctan2(-ax, np.sqrt(ay**2 + az**2))
        yaw = 0.0
        return np.array([roll, pitch, yaw])

    def quaternion_to_roll_pitch_yaw(self, Q):
        Q = self.from_ahrs_quaternion_convention_to_scipy(Q)
        R_orientation = R.from_quat(Q)
        return R_orientation.as_euler("xyz")

    def roll_pitch_yaw_to_quaternion(self, angles):
        R_orientation = R.from_euler("xyz", angles)
        Q = R_orientation.as_quat()
        return self.from_scipy_to_ahrs_quaternion_convention(Q)
