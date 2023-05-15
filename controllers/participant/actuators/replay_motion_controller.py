import os
import sys

from controller import Motion

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
)
from typing import Optional, Tuple

from utils.motion_library import MotionLibrary


class ReplayMotionController:
    def __init__(
        self,
        wait_until_finished: bool = True,
        motion_list: Optional[Tuple[str, ...]] = None,
        motion_list_reverse: Optional[Tuple[str, ...]] = None,
    ):
        self.library = MotionLibrary()
        self.__motion_list = (
            (
                "Flip",
                "ForwardLoop",
                "GetUpBack",
                "GetUpBackFast",
                "GetUpFront",
                "GetUpFrontFast",
                "Prepare",
                "PunchL",
                "PunchR",
                "Shove",
                "SideStepLeftLoop",
                "SideStepRightLoop",
                "Stand",
                "StandUpFromFront",
                "TurnLeft20",
                "TurnRight20",
            )
            if motion_list is None
            else motion_list
        )
        self.__motion_list_reverse = (
            (
                "_R_Flip",
                "_R_ForwardLoop",
            )
            if motion_list_reverse is None
            else motion_list_reverse
        )

        self.__wait_until_finished = wait_until_finished

        self.__current_motion: Optional[Tuple[str, Motion]] = None

    @property
    def input_size(self) -> int:
        return self.n_motions

    @property
    def n_motions(self) -> int:
        return len(self.__motion_list) + len(self.__motion_list_reverse)

    @property
    def n_motions_forward(self) -> int:
        return len(self.__motion_list)

    @property
    def n_motions_reverse(self) -> int:
        return len(self.__motion_list_reverse)

    @property
    def current_motion(self) -> Optional[Tuple[str, Motion]]:
        return self.__current_motion

    @property
    def wait_until_finished(self) -> bool:
        return self.__wait_until_finished

    def set_wait_until_finished(self, wait_until_finished: bool):
        self.__wait_until_finished = wait_until_finished

    def _get_motion_by_name(self, name: str) -> Motion:
        return self.library.get(name.replace("_R_", ""))

    def _get_motion_by_index(self, index: int) -> Motion:
        if index < self.n_motions_forward:
            return self._get_motion_by_name(self.__motion_list[index])
        else:
            return self._get_motion_by_name(
                self.__motion_list_reverse[index - self.n_motions_forward]
            )

    def play_motion_by_name(self, name: str):
        if self.__current_motion is not None:
            is_playing = not self.__current_motion[1].isOver()
            if is_playing and (
                self.__wait_until_finished or self.__current_motion[0] == name
            ):
                return
        self.__current_motion = (name, self._get_motion_by_name(name))
        self.__current_motion[1].setReverse(self.__current_motion[0].startswith("_R_"))
        self.__current_motion[1].setLoop(False)
        self.__current_motion[1].play()

    def play_motion_by_index(self, index: int):
        if index < self.n_motions_forward:
            return self.play_motion_by_name(self.__motion_list[index])
        else:
            return self.play_motion_by_name(
                self.__motion_list_reverse[index - self.n_motions_forward]
            )

    def stop_current_motion(self):
        if self.__current_motion is not None:
            self.__current_motion[1].stop()
            self.__current_motion = None
