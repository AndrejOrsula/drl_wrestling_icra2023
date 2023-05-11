from typing import Iterable, Tuple, Union

import numpy as np
from controller import LED, Robot


def _hsv_to_rgb(h: float, s: float, v: float) -> Tuple[float, float, float]:
    if s == 0.0:
        return v, v, v
    i = int(h * 6.0)
    f = (h * 6.0) - i
    p = v * (1.0 - s)
    q = v * (1.0 - s * f)
    t = v * (1.0 - s * (1.0 - f))
    i = i % 6
    if 0 == i:
        return v, t, p
    elif 1 == i:
        return q, v, p
    elif 2 == i:
        return p, v, t
    elif 3 == i:
        return p, q, v
    elif 4 == i:
        return t, p, v
    elif 5 == i:
        return v, p, q


class __LedController:
    def __init__(self, robot: Robot, time_step: float, device_name: str):
        self.device: LED = robot.getDevice(device_name)

    def set_color(self, color: int):
        self.device.set(color)

    def set_color_rgb(
        self, red: Union[int, float], green: Union[int, float], blue: Union[int, float]
    ):
        if isinstance(red, float):
            red = int(red * 255)
        if isinstance(green, float):
            green = int(green * 255)
        if isinstance(blue, float):
            blue = int(blue * 255)
        color = red * 65536 + green * 256 + blue
        self.set_color(color)

    def set_color_hsv(self, hue: float, saturation: float, value: float):
        red, green, blue = _hsv_to_rgb(hue, saturation, value)
        self.set_color_rgb(red, green, blue)

    def set_color_array(self, color: np.ndarray):
        if (
            color.dtype == np.float
            or color.dtype == np.float64
            or color.dtype == np.float32
            or color.dtype == np.float16
        ):
            color = (color * 255).astype(int)
        self.set_color(int(color[0] * 65536 + color[1] * 256 + color[2]))


class LedChestBoard(__LedController):
    def __init__(
        self, robot: Robot, time_step: float, device_name: str = "ChestBoard/Led"
    ):
        super().__init__(robot=robot, time_step=time_step, device_name=device_name)


class LedFaceLeft(__LedController):
    def __init__(
        self, robot: Robot, time_step: float, device_name: str = "Face/Led/Left"
    ):
        super().__init__(robot=robot, time_step=time_step, device_name=device_name)


class LedFaceRight(__LedController):
    def __init__(
        self, robot: Robot, time_step: float, device_name: str = "Face/Led/Right"
    ):
        super().__init__(robot=robot, time_step=time_step, device_name=device_name)


class LedEarsLeft(__LedController):
    def __init__(
        self, robot: Robot, time_step: float, device_name: str = "Ears/Led/Left"
    ):
        super().__init__(robot=robot, time_step=time_step, device_name=device_name)


class LedEarsRight(__LedController):
    def __init__(
        self, robot: Robot, time_step: float, device_name: str = "Ears/Led/Right"
    ):
        super().__init__(robot=robot, time_step=time_step, device_name=device_name)


class LedLFoot(__LedController):
    def __init__(self, robot: Robot, time_step: float, device_name: str = "LFoot/Led"):
        super().__init__(robot=robot, time_step=time_step, device_name=device_name)


class LedRFoot(__LedController):
    def __init__(self, robot: Robot, time_step: float, device_name: str = "RFoot/Led"):
        super().__init__(robot=robot, time_step=time_step, device_name=device_name)


class LedControllersCombined:
    def __init__(self, robot: Robot, time_step: float):
        self.__led_controllers = (
            LedChestBoard(robot=robot, time_step=time_step),
            LedFaceLeft(robot=robot, time_step=time_step),
            LedFaceRight(robot=robot, time_step=time_step),
            # LedEarsLeft(robot=robot, time_step=time_step),
            # LedEarsRight(robot=robot, time_step=time_step),
            LedLFoot(robot=robot, time_step=time_step),
            LedRFoot(robot=robot, time_step=time_step),
        )

    @property
    def input_size(self) -> int:
        return self.n_leds

    @property
    def n_leds(self) -> int:
        return len(self.__led_controllers)

    def set_color(self, color: int):
        for led_controller in self.__led_controllers:
            led_controller.set_color(color=color)

    def set_color_rgb(
        self, red: Union[int, float], green: Union[int, float], blue: Union[int, float]
    ):
        for led_controller in self.__led_controllers:
            led_controller.set_color_rgb(red=red, green=green, blue=blue)

    def set_color_hsv(self, hue: float, saturation: float, value: float):
        red, green, blue = _hsv_to_rgb(hue, saturation, value)
        self.set_color_rgb(red, green, blue)

    def set_color_rgb_array(self, rgb_array: np.ndarray):
        for led_controller in self.__led_controllers:
            led_controller.set_color_array(color=rgb_array)

    def set_colors(self, colors: Iterable[int]):
        for i, led_controller in enumerate(self.__led_controllers):
            led_controller.set_color(color=int(colors[i]))

    def set_colors_rgb(
        self,
        reds: Iterable[Union[int, float]],
        greens: Iterable[Union[int, float]],
        blues: Iterable[Union[int, float]],
    ):
        for i, led_controller in enumerate(self.__led_controllers):
            led_controller.set_color_rgb(red=reds[i], green=greens[i], blue=blues[i])

    def set_colors_rgb_array(self, rgb_arrays: np.ndarray):
        for i, led_controller in enumerate(self.__led_controllers):
            led_controller.set_color_array(color=rgb_arrays[i])
