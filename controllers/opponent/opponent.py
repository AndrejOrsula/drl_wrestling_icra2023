import os
import sys

MIRROR_PARTICIPANT: bool = False

if MIRROR_PARTICIPANT:
    sys.path.append(
        os.path.join(
            os.path.dirname(os.path.abspath(os.path.dirname(__file__))),
            "participant",
        )
    )
    exec(
        open(
            os.path.join(
                os.path.dirname(os.path.abspath(os.path.dirname(__file__))),
                "participant",
                "participant.py",
            )
        ).read()
    )
else:
    from controller import Robot

    sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
    from utils.image_processing import ImageProcessing as IP
    from utils.fall_detection import FallDetection
    from utils.gait_manager import GaitManager
    from utils.camera import Camera

    class ScriptedController(Robot):
        SMALLEST_TURNING_RADIUS = 0.1
        SAFE_ZONE = 0.75
        TIME_BEFORE_DIRECTION_CHANGE = 200

        def __init__(self):
            Robot.__init__(self)
            self.time_step = int(self.getBasicTimeStep())

            self.camera = Camera(self)
            self.fall_detector = FallDetection(self.time_step, self)
            self.gait_manager = GaitManager(self, self.time_step)
            self.heading_angle = 3.14 / 2
            self.counter = 0

        def run(self):
            while self.step(self.time_step) != -1:
                t = self.getTime()
                self.gait_manager.update_theta()
                if t < 4:
                    self.start_sequence()
                elif t > 4:
                    self.fall_detector.check()
                    self.walk()

        def start_sequence(self):
            self.gait_manager.command_to_motors(heading_angle=0)

        def walk(self):
            normalized_x = self._get_normalized_opponent_x()
            desired_radius = (
                (self.SMALLEST_TURNING_RADIUS / normalized_x)
                if abs(normalized_x) > 1e-3
                else None
            )
            if self.counter > self.TIME_BEFORE_DIRECTION_CHANGE:
                self.heading_angle = -self.heading_angle
                self.counter = 0
            self.counter += 1
            self.gait_manager.command_to_motors(
                desired_radius=desired_radius, heading_angle=self.heading_angle
            )

        def _get_normalized_opponent_x(self):
            img = self.camera.get_image()
            _, _, horizontal_coordinate = IP.locate_opponent(img)
            if horizontal_coordinate is None:
                return 0
            return horizontal_coordinate * 2 / img.shape[1] - 1

    wrestler = ScriptedController()
    wrestler.run()
